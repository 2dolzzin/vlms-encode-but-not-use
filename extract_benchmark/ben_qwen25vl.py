import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ben_qwen25vl.py
# Qwen2.5-VL-7B 커리큘럼 11시점 ckpt 로 일반 VLM 벤치마크(line_plot/counting_grid/mmvp)
# 추론 + 임베딩 추출 + 채점. (= ben_llava.py 벤치마크 로직 + inf_qwen25vl.py 의 Qwen 로더)
#
# 실행 (extract_benchmark/ 안에서, conda activate qwenvl):
#   python ben_qwen25vl.py --run qwen25vl_7b            # forward
#   python ben_qwen25vl.py --run qwen25vl_7b_reverse    # reverse
#   python ben_qwen25vl.py --run qwen25vl_7b --eval_only

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

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "train", "qwen25vl_curriculum"))
from qwen_data import (DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, build_prompt,  # noqa: E402
                       get_visual, load_model, load_processor, text_config)

# ── 모델 정체성 (Qwen2.5-VL-7B: 28 layer, hidden 3584) ───────────
LAYERS = [0, 7, 14, 21, 28]
TOP_P          = 0.7
MAX_NEW_TOKENS = 512

BASE_MODEL = os.path.join(ROOT, "Model/Qwen2.5-VL-7B-Instruct")
RUNS = {
    "qwen25vl_7b":         dict(root=os.path.join(ROOT, "checkpoint/Qwen25VL_7B"),         out="qwen25vl_7b"),
    "qwen25vl_7b_reverse": dict(root=os.path.join(ROOT, "checkpoint/Qwen25VL_7B_reverse"), out="qwen25vl_7b_reverse"),
}
# 무작위 순서 런 자동 등록: checkpoint/Qwen25VL_7B_order_<tag>_nohex → "qwen25vl_7b_order_<tag>" (result/qwen25vl_7b_order_<tag>)
RUNS.update(discover_order_runs(os.path.join(ROOT, "checkpoint"), "Qwen25VL_7B", "qwen25vl_7b"))
STAGES = {
    "S0": "S0_LogMAR", "S1": "S1_Similarity", "S2": "S2_Ishihara",
    "S3": "S3_Naming", "S4": "S4_Depth",
}

DATASETS = {
    "line_plot":     "/home/hail/HDD/vlm_inference/benchmark/Task_lowlevel/line_plot/dataset.json",
    "counting_grid": "/home/hail/HDD/vlm_inference/benchmark/Task_lowlevel/counting_grid/dataset.json",
    "mmvp":          "/home/hail/HDD/vlm_inference/benchmark/MMVP/dataset.json",
}


def _is_ready(d):
    """학습이 끝나 가중치까지 저장된 폴더만 체크포인트로 인정.
    _stage.sh 가 시작할 때 빈 end/ 를 먼저 만들기 때문에, 학습과 추론을 동시에 돌리면
    빈 폴더를 완성된 ckpt 로 오인해 로딩이 터진다. config.json + safetensors 존재로 판정."""
    return (os.path.isdir(d)
            and os.path.exists(os.path.join(d, "config.json"))
            and bool(glob.glob(os.path.join(d, "*.safetensors"))))


def build_ckpts(root):
    ck = {"S0_start": BASE_MODEL}
    for sid, sdir in stage_dirs(root, STAGES).items():
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")))
        if len(mids) > 1:
            print(f"[warn] {sdir} 에 checkpoint-* 여러 개 → {mids[-1]} 사용")
        mids = [m for m in mids if _is_ready(m)]
        if mids:
            ck[f"{sid}_mid"] = mids[-1]
        end = os.path.join(sdir, "end")
        if _is_ready(end):
            ck[f"{sid}_end"] = end
    return {k: v for k, v in ck.items() if k == "S0_start" or os.path.exists(v)}


def load_dataset(path):
    with open(path) as f:
        cases = json.load(f)
    return [{
        "id": str(c.get("id", "")),
        "image_path": c["image"],
        "question": c["conversations"][0]["value"],
        "answer": c["conversations"][1]["value"],
    } for c in cases]


# ── Qwen2.5-VL loader (inf_qwen25vl.py 와 동일) ───────────────────
class QwenInferExtract:
    def __init__(self, ckpt_path, device="cuda"):
        self.device = device
        self.processor = load_processor(ckpt_path, DEFAULT_MIN_PIXELS, DEFAULT_MAX_PIXELS)
        self.model = load_model(ckpt_path, dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.visual = get_visual(self.model)
        self.hidden_size = text_config(self.model).hidden_size
        n_layers = text_config(self.model).num_hidden_layers
        assert max(LAYERS) <= n_layers, f"LAYERS {LAYERS} > num_hidden_layers {n_layers}"
        self.eos_ids = [self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        self.processor.tokenizer.eos_token_id]

    @torch.inference_mode()
    def run_one(self, image_path, question, layers, temperature):
        image = Image.open(image_path).convert("RGB")
        prompt = build_prompt(self.processor, question)
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt").to(self.device)

        pv = inputs["pixel_values"].to(self.visual.dtype)
        image_embeds = self.visual(pv, grid_thw=inputs["image_grid_thw"])     # (n_img_tokens, hidden)
        proj_feat = image_embeds.float().mean(dim=0).cpu()

        gen_kw = dict(max_new_tokens=MAX_NEW_TOKENS, use_cache=True, eos_token_id=self.eos_ids,
                      return_dict_in_generate=True, output_hidden_states=True)
        if temperature > 0:
            gen_kw.update(do_sample=True, temperature=temperature, top_p=TOP_P)
        else:
            gen_kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**inputs, **gen_kw)

        n_in = inputs["input_ids"].shape[1]
        prediction = self.processor.tokenizer.decode(
            out.sequences[0, n_in:], skip_special_tokens=True).strip()

        prefill = out.hidden_states[0]
        feat = torch.stack([prefill[L][0, -1, :].float().cpu() for L in layers])
        return prediction, feat, proj_feat


# ── 벤치마크 파서 / 채점 (ben_llava.py 와 동일) ───────────────────
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
    # resume: raw_responses json + embedding pt 둘 다 있으면 추론은 건너뛰고 채점만
    _pp = os.path.join(dirs["pred"], ckpt_name, f"{ds_name}.json")
    _ep = os.path.join(dirs["emb"], ckpt_name, f"{ds_name}.pt")
    if os.path.exists(_pp) and os.path.exists(_ep):
        print(f"[skip] {ckpt_name}/{ds_name} (이미 완료 → 채점만)")
        return evaluate_one(ckpt_name, ds_name, json.load(open(_pp)), dirs)

    samples = load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []
    for s in tqdm(samples, desc=f"{ckpt_name}/{ds_name}", ncols=90):
        pred, feat, p_feat = loader.run_one(s["image_path"], s["question"], LAYERS, temperature)
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
    return evaluate_one(ckpt_name, ds_name, records, dirs)


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
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {run_key} / {ckpt_name} ##########")
        loader = QwenInferExtract(ckpt_path)
        for ds_name in DATASETS:
            s = process_dataset(loader, ckpt_name, ds_name, DATASETS[ds_name], temperature, dirs)
            if s:
                rows.append(s)
        del loader
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
                    help="쉼표구분 task 만 실행 (예: mmvp,line_plot). 미지정=전체")
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
