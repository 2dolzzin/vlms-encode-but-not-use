import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ben_openflamingo_blind.py
# ben_openflamingo.py 에서 채점(evaluate_one) 을 뺀 blind 벤치 전용 추론 스크립트.
#   blind 9 task 추론 + 임베딩 추출만 수행 (raw_responses + embeddings).
#   결과는 result/<out>_blind/ 밑에 저장. 채점은 별도 스크립트에서.
# OpenFlamingo-3B(vitl-mpt1b) 커리큘럼 11시점 ckpt 로 일반 VLM 벤치마크(line_plot/counting_grid/mmvp)
# 추론 + 임베딩 추출 + 채점. (= ben_vila_8b.py 의 OpenFlamingo 판)
#
# 모델 고유부(loader/run_one)만 OF 로 교체, 파서·채점·저장은 ben_vila_8b 와 동일.
#   · 모델 로드: create_model_and_transforms + .pt state_dict (1회 빌드 후 ckpt마다 weight 교체)
#   · proj_feats = perceiver(resampler) 출력 평균 (1024)
#   · feats = LLM 층 [0,6,12,18,24] prefill 마지막 토큰 (5, 2048)  ← hook 으로 추출
#       layer 0 = cross-attn 전 순수 텍스트(FlamingoLayer[0] 입력)
#       layer 6/12/18/24 = 비전 융합 후(FlamingoLayer[5/11/17/23] 출력)
#   · generate 는 프롬프트 포함 반환 → prompt_len 슬라이싱
#
# 출력: result/{out}/{raw_responses,embeddings,accuracy,parsing}/
#   raw_responses/{ckpt}/{task}.json   embeddings/{ckpt}/{task}.pt
#   accuracy/{ckpt}__{task}.json + _summary_all.json   parsing/{ckpt}__{task}.json
#
# 실행 (extract_benchmark/ 안에서):
#   python ben_openflamingo.py --run openflamingo            # forward
#   python ben_openflamingo.py --run openflamingo_reverse    # reverse
#   python ben_openflamingo.py --run openflamingo --eval_only  # 재채점(GPU 불필요)

import sys
import re
import glob
import json
import argparse

import numpy as np
from sklearn.metrics import f1_score

import torch
from PIL import Image
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))     # .../extract_benchmark
ROOT = os.path.dirname(BASE)                           # .../vlm3
sys.path.insert(0, ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402

# ── 모델 정체성 (OpenFlamingo) ────────────────────────────────
OF_PKG     = os.path.join(ROOT, "Model/open_flamingo")
PRETRAINED = os.path.join(OF_PKG, "ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt")

LAYERS   = [0, 6, 12, 18, 24]              # MPT-1B = 24층 (hidden 2048)
HOOK_IDX = {0: ("pre", 0), 6: ("post", 5), 12: ("post", 11),
            18: ("post", 17), 24: ("post", 23)}   # layer→(hook종류, FlamingoLayer index)
MAX_NEW_TOKENS = 32       # 벤치 답은 숫자/글자 1개라 32면 충분 (base 모델 횡설수설 방지)
BATCH_SIZE     = 16       # 추론 배칭 (좌패딩 → 마지막토큰 -1 정렬)

# forward / reverse = 체크포인트 root 와 출력 폴더만 다름
RUNS = {
    "openflamingo":         dict(root=os.path.join(ROOT, "checkpoint/OpenFlamingo"),         out="openflamingo_blind"),
    "openflamingo_reverse": dict(root=os.path.join(ROOT, "checkpoint/OpenFlamingo_reverse"), out="openflamingo_reverse_blind"),
}
# 무작위 순서 런 자동 등록: checkpoint/OpenFlamingo_order_<tag>_nohex → "openflamingo_order_<tag>" (result/openflamingo_order_<tag>_blind)
RUNS.update(discover_order_runs(os.path.join(ROOT, "checkpoint"), "OpenFlamingo", "openflamingo", out_suffix="_blind"))
STAGES = {
    "S0": "S0_LogMAR", "S1": "S1_Similarity", "S2": "S2_Ishihara",
    "S3": "S3_Naming", "S4": "S4_Depth",
}

BLIND_ROOT = "/home/hail/HDD/blind/unpacked"
DATASETS = {
    "circled_letter":      f"{BLIND_ROOT}/Circled_Letter/dataset.json",
    "counting_grid_blank": f"{BLIND_ROOT}/Counting_Grid_-_Blank_Grids/dataset.json",
    "counting_grid_word":  f"{BLIND_ROOT}/Counting_Grid_-_Word_Grids/dataset.json",
    "nested_squares":      f"{BLIND_ROOT}/Nested_Squares/dataset.json",
    "olympic_circles":     f"{BLIND_ROOT}/Olympic_Counting_-_Circles/dataset.json",
    "olympic_pentagons":   f"{BLIND_ROOT}/Olympic_Counting_-_Pentagons/dataset.json",
    "subway_connections":  f"{BLIND_ROOT}/Subway_Connections/dataset.json",
    "touching_circles":    f"{BLIND_ROOT}/Touching_Circles/dataset.json",
    # 전이 평가(실험 3)의 나머지 둘. line_plot 은 기존 fwd/rev 결과(result_origin)와 같은 샘플셋을 써서
    # 비교 가능하게 한다 — blind/unpacked/Line_Plot_Intersections 는 같은 task 지만 샘플이 다름(이미지 교집합 0).
    "line_plot": "/home/hail/HDD/vlm_inference/benchmark/Task_lowlevel/line_plot/dataset.json",
    "mmvp":      "/home/hail/HDD/vlm_inference/benchmark/MMVP/dataset.json",
}


def _resolve_pt(path):
    """ckpt state(.pt 파일 또는 디렉토리) → 실제 .pt 파일 경로."""
    if os.path.isfile(path):
        return path
    for c in ("flamingo_model.pt", "pytorch_model.bin"):
        p = os.path.join(path, c)
        if os.path.isfile(p):
            return p
    return None


def build_ckpts(root):
    """{ckpt_name: state경로}. S0_start=pretrained, mid=0.5epoch checkpoint, end=stage디렉토리(flamingo_model.pt).
    OF 는 stage당 checkpoint-{0.5ep, 1.0ep} 2개가 저장되는데, 1.0ep(최종 step)은 end(flamingo_model.pt)와
    동일 시점이므로 mid 에서 제외하고 그 앞(=0.5epoch) 체크포인트를 mid 로 쓴다.
    (다른 모델은 checkpoint-* 가 0.5ep 하나뿐이라 자동으로 올바름)"""
    ck = {"S0_start": PRETRAINED}
    for sid, sdir in stage_dirs(root, STAGES).items():
        # step 숫자 기준 정렬 (checkpoint-1562 < checkpoint-3124)
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")),
                      key=lambda p: int(os.path.basename(p).split("-")[-1]))
        # 최종 step 체크포인트는 end 와 겹치므로 제외 → 남은 것 중 마지막(=0.5ep) 을 mid
        mid_cands = mids[:-1] if len(mids) > 1 else mids
        if mid_cands and _resolve_pt(mid_cands[-1]):
            ck[f"{sid}_mid"] = mid_cands[-1]
        if _resolve_pt(sdir):
            ck[f"{sid}_end"] = sdir
    return {k: v for k, v in ck.items() if k == "S0_start" or _resolve_pt(v)}


def load_dataset(path):
    with open(path) as f:
        cases = json.load(f)
    return [{
        "id": str(c.get("id", "")),
        "image_path": c["image"],
        "question": c["conversations"][0]["value"],
        "answer": c["conversations"][1]["value"],
    } for c in cases]


# ── OpenFlamingo loader (inf_openflamingo.py 와 동일) ─────────
sys.path.insert(0, OF_PKG)
from open_flamingo import create_model_and_transforms


class OFInferExtract:
    def __init__(self):
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1,
        )
        self.model = self.model.to("cuda", dtype=torch.bfloat16).eval()
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.flayers = self.model.lang_encoder._get_decoder_layers()   # 24 FlamingoLayer
        self.hidden_size = 2048

    def load_ckpt(self, state_path):
        pt = _resolve_pt(state_path)
        sd = torch.load(pt, map_location="cpu")
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)

    def _expand2square(self, img):
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        r = Image.new(img.mode, (s, s), (128, 128, 128))
        r.paste(img, (0, (s - h) // 2) if w > h else ((s - w) // 2, 0))
        return r

    @torch.inference_mode()
    def run_batch(self, image_paths, questions, temperature):
        """N개 샘플 배칭. 좌패딩이라 모든 샘플 마지막토큰이 -1 위치로 정렬됨.
        반환: preds(list[str]), feats(N,5,2048), proj(N,1024)"""
        N = len(image_paths)
        # ---- 이미지 → vision_x (N, T_img=1, F=1, C, H, W) ----
        vxs = []
        for p in image_paths:
            img = self._expand2square(Image.open(p).convert("RGB"))
            vx = self.image_processor(img)
            if isinstance(vx, dict):
                vx = vx["pixel_values"]
            if isinstance(vx, list):
                vx = vx[0]
            vxs.append(torch.as_tensor(vx))
        vision_x = torch.stack(vxs).unsqueeze(1).unsqueeze(1).to("cuda", dtype=torch.bfloat16)

        # ---- 프롬프트 (<image> 는 question 안에 이미 있음) + 좌패딩 ----
        prompts = [f"USER: {q if '<image>' in q else '<image> ' + q}\nASSISTANT:" for q in questions]
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        ids = enc.input_ids.to("cuda")
        am = enc.attention_mask.to("cuda")

        # ---- hook: perceiver(resampler) + FlamingoLayer ----
        cap = {}
        handles = []
        handles.append(self.model.perceiver.register_forward_hook(
            lambda m, i, o: cap.__setitem__("resampler", o.detach())))
        for L, (kind, idx) in HOOK_IDX.items():
            if kind == "pre":
                handles.append(self.flayers[idx].register_forward_pre_hook(
                    (lambda L: lambda m, args: cap.__setitem__(f"L{L}", args[0].detach()))(L)))
            else:
                handles.append(self.flayers[idx].register_forward_hook(
                    (lambda L: lambda m, i, o: cap.__setitem__(
                        f"L{L}", (o[0] if isinstance(o, tuple) else o).detach()))(L)))

        # ---- 특징용 forward 1회 (prefill) ----
        self.model(vision_x=vision_x, lang_x=ids, attention_mask=am)
        # perceiver (N, T=1, 64, 1024) → latent 평균 → (N, 1024)
        proj = cap["resampler"].float().mean(dim=2).reshape(N, -1).cpu()
        # 각 층 마지막 토큰(좌패딩 → -1) → (N, 5, 2048)
        feat = torch.stack([cap[f"L{L}"][:, -1, :].float() for L in LAYERS], dim=1).cpu()
        for h in handles:
            h.remove()

        # ---- 예측용 generate (프롬프트 포함 반환, 좌패딩이라 ids.shape[1] 이후가 생성분) ----
        out = self.model.generate(
            vision_x=vision_x, lang_x=ids, attention_mask=am,
            max_new_tokens=MAX_NEW_TOKENS, num_beams=1,
            do_sample=temperature > 0, temperature=temperature)
        gen = out[:, ids.shape[1]:]
        preds = [self.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]
        return preds, feat, proj


# ── 벤치마크 파서 / 채점 (ben_vila_8b.py 와 동일) ─────────────
def parse_count(text, valid_range=(0, 2)):
    if not text:
        return None
    m = re.findall(r'\{\s*(-?\d+)\s*\}', text)
    if m:
        v = int(m[-1])
        return v if valid_range[0] <= v <= valid_range[1] else None
    m = re.findall(r'-?\d+', text)
    if m:
        v = int(m[-1])
        return v if valid_range[0] <= v <= valid_range[1] else None
    return None


def parse_grid(text, max_val=10):
    if not text:
        return None
    r = re.search(r'row\w*\s*=?\s*\{?\s*(\d+)', text, re.I)
    c = re.search(r'col\w*\s*=?\s*\{?\s*(\d+)', text, re.I)
    if r and c:
        rv, cv = int(r.group(1)), int(c.group(1))
        return (rv, cv) if (1 <= rv <= max_val and 1 <= cv <= max_val) else None

    nums = re.findall(r'\{\s*(\d+)\s*\}', text)
    if len(nums) >= 2:
        rv, cv = int(nums[0]), int(nums[1])
        return (rv, cv) if (1 <= rv <= max_val and 1 <= cv <= max_val) else None

    nums = re.findall(r'\d+', text)
    if len(nums) >= 2:
        rv, cv = int(nums[0]), int(nums[1])
        return (rv, cv) if (1 <= rv <= max_val and 1 <= cv <= max_val) else None

    return None


def parse_choice(text, valid="ab"):
    if not text:
        return None
    t = text.strip()
    m = re.search(r'[\(\[]?\s*([a-d])\s*[\)\].:]', t, re.I)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    m = re.match(r'\s*([a-d])\b', t, re.I)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    for m in re.finditer(r'\b([a-d])\b', t, re.I):
        if m.group(1).lower() in valid:
            return m.group(1).lower()
    return None


def task_kind(task_name):
    t = task_name.lower()
    if "line" in t:
        return "line_plot"
    if "count" in t or "grid" in t:
        return "counting_grid"
    if "mmvp" in t:
        return "mmvp"
    return "unknown"


def evaluate_one(ckpt_name, ds_name, records, dirs):
    kind = task_kind(ds_name)
    n = len(records)
    details = []
    summary = {"ckpt": ckpt_name, "task": ds_name, "kind": kind, "n": n}

    if kind == "line_plot":
        preds, gts, parsed = [], [], 0
        for r in records:
            p = parse_count(r["output"])
            g = str(r["expected_output"]).strip()
            parsed += p is not None
            preds.append(str(p) if p is not None else "<none>")
            gts.append(g)
            details.append({"id": r["id"], "pred": (str(p) if p is not None else None),
                            "gt": g, "correct": str(p) == g})
        labels = sorted(set(gts), key=lambda x: int(x))
        acc = float(np.mean([p == g for p, g in zip(preds, gts)])) if n else 0
        f1 = float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0))
        summary.update({"parsed": parsed, "parse_ratio": parsed / n if n else 0,
                        "accuracy": acc, "f1": f1})
        print(f"  [{ckpt_name}/{ds_name}] (clf) acc={acc:.4f} F1={f1:.4f} parse={parsed}/{n}")

    elif kind == "counting_grid":
        parsed = exact = 0
        row_err, col_err, all_err = [], [], []
        for r in records:
            pr = parse_grid(r["output"])
            g = tuple(int(x) for x in str(r["expected_output"]).split(","))
            ok = pr is not None
            parsed += ok
            if ok:
                re_, ce_ = abs(pr[0] - g[0]), abs(pr[1] - g[1])
                row_err.append(re_); col_err.append(ce_); all_err += [re_, ce_]
                exact += (pr == g)
            details.append({"id": r["id"], "pred": pr, "gt": g,
                            "row_err": (abs(pr[0] - g[0]) if ok else None),
                            "col_err": (abs(pr[1] - g[1]) if ok else None)})
        summary.update({
            "parsed": parsed, "parse_ratio": parsed / n if n else 0,
            "mae": float(np.mean(all_err)) if all_err else float("nan"),
            "rows_mae": float(np.mean(row_err)) if row_err else float("nan"),
            "cols_mae": float(np.mean(col_err)) if col_err else float("nan"),
            "exact_acc": exact / n if n else 0,
        })
        print(f"  [{ckpt_name}/{ds_name}] (reg) MAE={summary['mae']:.4f} "
              f"exact={summary['exact_acc']:.4f} parse={parsed}/{n}")

    elif kind == "mmvp":
        preds, gts, parsed = [], [], 0
        for r in records:
            p = parse_choice(r["output"])
            g = str(r["expected_output"]).split(":")[0].strip().lower()
            parsed += p is not None
            preds.append(p if p is not None else "<none>")
            gts.append(g)
            details.append({"id": r["id"], "pred": p, "gt": g, "correct": p == g})
        labels = sorted(set(gts))
        acc = float(np.mean([p == g for p, g in zip(preds, gts)])) if n else 0
        f1 = float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0))
        summary.update({"parsed": parsed, "parse_ratio": parsed / n if n else 0,
                        "accuracy": acc, "f1": f1})
        print(f"  [{ckpt_name}/{ds_name}] acc={acc:.4f} F1={f1:.4f} parse={parsed}/{n}")
    else:
        print(f"  [skip] {ckpt_name}/{ds_name}: unknown kind")
        return None

    os.makedirs(dirs["acc"], exist_ok=True)
    os.makedirs(dirs["parse"], exist_ok=True)
    json.dump(summary, open(os.path.join(dirs["acc"], f"{ckpt_name}__{ds_name}.json"), "w"), indent=2)
    json.dump(details, open(os.path.join(dirs["parse"], f"{ckpt_name}__{ds_name}.json"), "w"),
              indent=2, ensure_ascii=False)
    return summary


def process_dataset(loader, ckpt_name, ds_name, ds_path, temperature, dirs):
    # resume: raw_responses json + embedding pt 둘 다 있으면 이미 완료된 것 → 스킵
    _pp = os.path.join(dirs["pred"], ckpt_name, f"{ds_name}.json")
    _ep = os.path.join(dirs["emb"], ckpt_name, f"{ds_name}.pt")
    if os.path.exists(_pp) and os.path.exists(_ep):
        print(f"[skip] {ckpt_name}/{ds_name} (이미 완료)")
        return None
    samples = load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []
    for start in tqdm(range(0, len(samples), BATCH_SIZE),
                      desc=f"{ckpt_name}/{ds_name}", ncols=90):
        chunk = samples[start:start + BATCH_SIZE]
        preds, feat_b, proj_b = loader.run_batch(
            [s["image_path"] for s in chunk], [s["question"] for s in chunk], temperature)
        for s, pred, feat, p_feat in zip(chunk, preds, feat_b, proj_b):
            records.append(dict(id=s["id"], question=s["question"],
                                output=pred, expected_output=s["answer"]))
            feats.append(feat); proj_feats.append(p_feat)
            predictions.append(pred); answers.append(s["answer"]); ids.append(s["id"])

    feats = torch.stack(feats).half()
    proj_feats = torch.stack(proj_feats).half()

    pred_dir = os.path.join(dirs["pred"], ckpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    with open(os.path.join(pred_dir, f"{ds_name}.json"), "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    emb_dir = os.path.join(dirs["emb"], ckpt_name)
    os.makedirs(emb_dir, exist_ok=True)
    torch.save({"feats": feats, "proj_feats": proj_feats, "predictions": predictions,
                "answers": answers, "ids": ids, "layers": LAYERS,
                "hidden_size": loader.hidden_size},
               os.path.join(emb_dir, f"{ds_name}.pt"))

    print(f"[saved] {ckpt_name}/{ds_name}  feats={tuple(feats.shape)} proj={tuple(proj_feats.shape)}")
    return None  # blind: 채점 안 함 (raw_responses/embeddings 만)


def make_dirs(out):
    rdir = os.path.join(BASE, "result", out)
    return {"pred": os.path.join(rdir, "raw_responses"),
            "emb": os.path.join(rdir, "embeddings"),
            "acc": os.path.join(rdir, "accuracy"),
            "parse": os.path.join(rdir, "parsing")}


def save_summary(rows, dirs):
    os.makedirs(dirs["acc"], exist_ok=True)
    out = os.path.join(dirs["acc"], "_summary_all.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n전체 요약 → {out}")


def run(run_key, temperature):
    cfg = RUNS[run_key]
    dirs = make_dirs(cfg["out"])
    ckpts = build_ckpts(cfg["root"])
    print(f"[{run_key}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")
    rows = []
    loader = OFInferExtract()                    # 모델 1회 빌드
    for ckpt_name, state_path in ckpts.items():
        print(f"\n########## {run_key} / {ckpt_name} ##########")
        loader.load_ckpt(state_path)             # weight 교체
        for ds_name in DATASETS:
            s = process_dataset(loader, ckpt_name, ds_name, DATASETS[ds_name], temperature, dirs)
            if s:
                rows.append(s)
        torch.cuda.empty_cache()
    save_summary(rows, dirs)


def run_eval_only(run_key):
    dirs = make_dirs(RUNS[run_key]["out"])
    paths = sorted(glob.glob(os.path.join(dirs["pred"], "*", "*.json")))
    if not paths:
        print(f"raw_responses 없음. (dir={dirs['pred']})")
        return
    rows = []
    for p in paths:
        ckpt = os.path.basename(os.path.dirname(p))
        ds_name = os.path.basename(p).replace(".json", "")
        s = evaluate_one(ckpt, ds_name, json.load(open(p)), dirs)
        if s:
            rows.append(s)
    save_summary(rows, dirs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", choices=list(RUNS) + ["all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--tasks", default=None,
                    help="쉼표구분 task 만 실행 (예: counting_grid_word,circled_letter). 미지정=전체")
    ap.add_argument("--list", action="store_true",
                    help="모델 로드 없이 run 별 체크포인트 11시점만 출력(드라이런)")
    a = ap.parse_args()
    if a.list:
        for rk in (list(RUNS) if a.run == "all" else [a.run]):
            ck = build_ckpts(RUNS[rk]["root"]); print(f"[{rk}] {len(ck)} ckpt  root={RUNS[rk]['root']}")
            print(describe_ckpts(ck, read_order(RUNS[rk]["root"]))); print()
        raise SystemExit(0)
    if a.tasks:
        keep = [t.strip() for t in a.tasks.split(",")]
        for _k in list(DATASETS):
            if _k not in keep:
                DATASETS.pop(_k)
        print(f"[tasks] 실행 대상: {list(DATASETS)}")
    run_keys = list(RUNS) if a.run == "all" else [a.run]
    for rk in run_keys:
        print(f"\n================= RUN: {rk} =================")
        if a.eval_only:
            run_eval_only(rk)
        else:
            run(rk, a.temperature)
