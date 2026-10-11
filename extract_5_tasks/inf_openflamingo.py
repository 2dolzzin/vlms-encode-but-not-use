import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# inf_openflamingo.py
# OpenFlamingo-3B(vitl-mpt1b) 커리큘럼 11시점 ckpt에서 inf_vila/inf_llava/inf_mipha 와
# 동일하게 (1) 추론 → 예측 raw_responses json, (2) 임베딩 추출 → .pt 저장.
#
# ── OpenFlamingo 특성(다른 모델과 다른 점) ──────────────────────────────
#  · 모델 로드: HF dir 가 아니라 create_model_and_transforms + .pt state_dict 로드
#               (ViT/MPT 빌드는 비싸므로 '한 번' 만들고 ckpt마다 weight만 교체)
#  · 비전 합치는 법: projector(이어붙이기)가 아니라 perceiver + gated cross-attn.
#       - proj_feat 대응 = perceiver(resampler) 출력 평균  → dim 1024
#       - cross-attn 은 텍스트 스트림(MPT, dim 2048)에 곁들여짐
#  · 추출 지점:
#       proj_feats = resampler(perceiver) 출력 평균                (N, 1024)
#       feats      = LLM 층 [0,6,12,18,24] prefill 마지막 토큰      (N, 5, 2048)
#                    layer 0 = cross-attn 전 '순수 텍스트 임베딩'(FlamingoLayer[0] 입력)
#                    layer 6/12/18/24 = 비전 융합 후 (FlamingoLayer[5/11/17/23] 출력)
#  · hidden_states 가 24층 출력을 누락하므로 → forward hook 으로 직접 추출
#  · generate 는 '프롬프트 포함' 시퀀스를 반환 → prompt_len 으로 슬라이싱
#  · 데이터셋 답 포맷: flamingo(raw, "Output:" 접두사 없음). Task_5_stage 가 이미 raw.
#
# 저장 키는 inf_vila 와 동일(feats/proj_feats) → probe_regressor 그대로 사용 가능.
# 출력:
#   result/openflamingo/inference/raw_responses/{ckpt}/{task}.json
#   result/openflamingo/embedding/{ckpt}/{task}.pt
#
# 실행:
#   python inf_openflamingo.py                 # forward 11시점 × 5task
#   python inf_openflamingo.py --reverse       # reverse 커리큘럼

import sys
import json
import glob
import argparse

import torch
from PIL import Image


# ============================================================
# 설정
# ============================================================
OF_PKG = "../Model/open_flamingo"          # `from open_flamingo import ...`
PRETRAINED = "/home/hail/vlm3/Model/open_flamingo/ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt"

LAYERS    = [0, 6, 12, 18, 24]             # MPT-1B = 24층(hidden 2048)
HOOK_IDX  = {0: ("pre", 0), 6: ("post", 5), 12: ("post", 11),
             18: ("post", 17), 24: ("post", 23)}   # layer→(hook종류, FlamingoLayer index)

MAX_NEW_TOKENS = 64                        # OF 답은 짧음(+<|endofchunk|>에서 멈춤). forward 완료본과 동일 유지

MODELS = {
    "openflamingo": dict(
        root="../checkpoint/OpenFlamingo",
        out="openflamingo",
        emb="embedding",
    ),
}
# 무작위 순서 커리큘럼 런(checkpoint/<Model>_order_<tag>_nohex) 공용 유틸 — 저장소 루트의 order_runs.py
_ORDER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ORDER_ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402
CKPT_ROOT = os.path.join(_ORDER_ROOT, "checkpoint")
# 무작위 순서 런 자동 등록: OpenFlamingo_order_<tag>_nohex → "openflamingo_order_<tag>"
MODELS.update(discover_order_runs(CKPT_ROOT, "OpenFlamingo", "openflamingo", emb="embedding"))

STAGES = {
    "S0": "S0_LogMAR",
    "S1": "S1_Similarity",
    "S2": "S2_Ishihara",
    "S3": "S3_Naming",
    "S4": "S4_Depth",
}

# 추론 데이터셋 (다른 모델과 동일 파일; 답이 raw = flamingo 포맷)
DATASETS = {
    "S0_logamr":     "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
    # ⚠ S1 은 nohex 가 표준(hex 프롬프트는 정답 누출 → 학습/검증/추론 전부 nohex). 기존 hex 결과와 섞지 말 것.
    "S1_similarity": "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json",
    "S2_ishihara":   "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
    "S3_naming":     "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "S4_depth":      "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
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


def build_ckpts(cfg):
    """{ckpt_name: state경로} 자동 구성. start=pretrained, mid=0.5epoch checkpoint, end=stage디렉토리(flamingo_model.pt).
    OF 는 stage당 checkpoint-{0.5ep, 1.0ep} 2개가 저장되는데, 1.0ep(최종 step)은 end 와 동일 시점이므로
    mid 에서 제외하고 그 앞(=0.5epoch) 체크포인트를 mid 로 쓴다.
    실제 .pt 가 있는 것만 담는다."""
    ck = {"S0_start": PRETRAINED}
    for sid, sdir in stage_dirs(cfg["root"], STAGES).items():
        # step 숫자 기준 정렬 (checkpoint-1562 < checkpoint-3124)
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")),
                      key=lambda p: int(os.path.basename(p).split("-")[-1]))
        # 최종 step 체크포인트는 end 와 겹치므로 제외 → 남은 것 중 마지막(=0.5ep) 을 mid
        mid_cands = mids[:-1] if len(mids) > 1 else mids
        if mid_cands and _resolve_pt(mid_cands[-1]):
            ck[f"{sid}_mid"] = mid_cands[-1]
        if _resolve_pt(sdir):                # end = stage 디렉토리의 flamingo_model.pt
            ck[f"{sid}_end"] = sdir
    return {k: v for k, v in ck.items() if k == "S0_start" or _resolve_pt(v)}


# ============================================================
# 데이터 로드 — conversations 포맷 (다른 모델과 동일)
# ============================================================
def load_dataset(path):
    with open(path) as f:
        cases = json.load(f)
    out = []
    for c in cases:
        out.append({
            "id":         str(c.get("id", "")),
            "image_path": c["image"],
            "question":   c["conversations"][0]["value"],   # [0]=질문(<image> 포함)
            "answer":     c["conversations"][1]["value"],   # [1]=정답(raw)
        })
    return out


# ============================================================
# OpenFlamingo loader (모델 1회 빌드 + ckpt마다 weight 교체)
# ============================================================
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
    def run_one(self, image_path, question, temperature):
        # ---- 이미지 → vision_x (B,T_img,F,C,H,W) ----
        img = self._expand2square(Image.open(image_path).convert("RGB"))
        vx = self.image_processor(img)
        if isinstance(vx, dict):
            vx = vx["pixel_values"]
        if isinstance(vx, list):
            vx = vx[0]
        vx = torch.as_tensor(vx).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        vx = vx.to("cuda", dtype=torch.bfloat16)

        # ---- 프롬프트 (학습 포맷과 동일: <image> 는 question 안에 이미 있음) ----
        prompt = f"USER: {question}\nASSISTANT:"
        enc = self.tokenizer(prompt, return_tensors="pt")
        ids = enc.input_ids.to("cuda")
        am = enc.attention_mask.to("cuda")

        # ---- hook 등록: perceiver(resampler) + FlamingoLayer ----
        cap = {}
        handles = []
        handles.append(self.model.perceiver.register_forward_hook(
            lambda m, i, o: cap.__setitem__("resampler", o.detach())))
        for L, (kind, idx) in HOOK_IDX.items():
            if kind == "pre":   # layer 0 = FlamingoLayer 입력(=순수 텍스트 임베딩)
                handles.append(self.flayers[idx].register_forward_pre_hook(
                    (lambda L: lambda m, args: cap.__setitem__(f"L{L}", args[0].detach()))(L)))
            else:               # layer 6/12/18/24 = FlamingoLayer 출력 (튜플 → [0])
                handles.append(self.flayers[idx].register_forward_hook(
                    (lambda L: lambda m, i, o: cap.__setitem__(
                        f"L{L}", (o[0] if isinstance(o, tuple) else o).detach()))(L)))

        # ---- 특징용 forward 1회 (prefill) ----
        self.model(vision_x=vx, lang_x=ids, attention_mask=am)

        # perceiver 출력 (b,T,n,d) → latent 평균 → (1024)
        proj_feat = cap["resampler"].float().mean(dim=(0, 1, 2)).cpu()
        # 각 층 마지막 토큰 → (5, 2048)
        feat = torch.stack([cap[f"L{L}"][0, -1, :].float().cpu() for L in LAYERS])

        for h in handles:
            h.remove()

        # ---- 예측용 generate (프롬프트 포함 반환 → 슬라이싱) ----
        out = self.model.generate(
            vision_x=vx, lang_x=ids, attention_mask=am,
            max_new_tokens=MAX_NEW_TOKENS, num_beams=1,
            do_sample=temperature > 0, temperature=temperature)
        prediction = self.tokenizer.decode(
            out[0][ids.shape[1]:], skip_special_tokens=True).strip()

        return prediction, feat, proj_feat


# ============================================================
# 데이터셋 1개 처리 → 예측 json + 임베딩 pt
# ============================================================
def process_dataset(loader, cfg, ckpt_name, ds_name, ds_path, temperature):
    # 재개: 예측 json + 임베딩 pt 가 둘 다 있으면 건너뜀 (중간에 죽어도 이어서 돌릴 수 있게; qwen/blind 와 동일)
    if (os.path.exists(os.path.join(os.path.join("result", cfg["out"], "inference", "raw_responses", ckpt_name), f"{ds_name}.json"))
            and os.path.exists(os.path.join(os.path.join("result", cfg["out"], cfg["emb"], ckpt_name), f"{ds_name}.pt"))):
        print(f"[skip] {ckpt_name}/{ds_name} (이미 완료)")
        return
    samples = load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []

    for j, s in enumerate(samples):
        pred, feat, p_feat = loader.run_one(s["image_path"], s["question"], temperature)
        records.append(dict(id=s["id"], question=s["question"],
                            output=pred, expected_output=s["answer"]))
        feats.append(feat)
        proj_feats.append(p_feat)
        predictions.append(pred)
        answers.append(s["answer"])
        ids.append(s["id"])
        if (j + 1) % 50 == 0:
            print(f"    ... {j+1}/{len(samples)}")

    feats = torch.stack(feats).half()              # (N, 5, 2048)
    proj_feats = torch.stack(proj_feats).half()    # (N, 1024)

    pred_dir = os.path.join("result", cfg["out"], "inference", "raw_responses", ckpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{ds_name}.json")
    with open(pred_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    emb_dir = os.path.join("result", cfg["out"], cfg["emb"], ckpt_name)
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, f"{ds_name}.pt")
    torch.save({
        "feats": feats,              # LLM 층 [0,6,12,18,24]
        "proj_feats": proj_feats,    # resampler 출력
        "predictions": predictions,
        "answers": answers,
        "ids": ids,
        "layers": LAYERS,
        "hidden_size": loader.hidden_size,
    }, emb_path)

    print(f"[saved] {ckpt_name}/{ds_name} → {pred_path}")
    print(f"        {emb_path}  feats={tuple(feats.shape)} proj={tuple(proj_feats.shape)}")


# ============================================================
# 실행 — ckpt × dataset 이중 루프 (모델은 1회 빌드, weight만 교체)
# ============================================================
def run(model, temperature, reverse=False):
    cfg = dict(MODELS[model])
    tag = model
    if reverse and "_order_" in model:
        raise SystemExit(f"--reverse 는 forward 런에만 쓴다. {model} 은 무작위 순서 런이라 폴더 자체가 순서를 담고 있음.")
    if reverse:
        cfg["root"] = cfg["root"] + "_reverse"   # ../checkpoint/OpenFlamingo_reverse
        cfg["out"]  = cfg["out"]  + "_reverse"   # result/openflamingo_reverse
        tag = model + "_reverse"

    ckpts = build_ckpts(cfg)
    print(f"[{tag}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")

    loader = OFInferExtract()                    # 모델 1회 빌드
    for ckpt_name, state_path in ckpts.items():
        print(f"\n########## {tag} / {ckpt_name} ##########")
        loader.load_ckpt(state_path)             # weight 교체
        for ds_name in DATASETS:
            print(f"  -- {ds_name} --")
            process_dataset(loader, cfg, ckpt_name, ds_name,
                            DATASETS[ds_name], temperature)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="openflamingo", choices=list(MODELS) + ["all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reverse", action="store_true",
                    help="리버스 커리큘럼(OpenFlamingo_reverse) → result/openflamingo_reverse")
    ap.add_argument("--list", action="store_true",
                    help="모델 로드 없이 run 별 체크포인트 11시점만 출력(드라이런)")
    a = ap.parse_args()
    if a.list:
        for mk in (list(MODELS) if a.model == "all" else [a.model]):
            cfg = dict(MODELS[mk])
            if a.reverse and "_order_" not in mk:
                cfg["root"] += "_reverse"; mk += "_reverse"
            ck = build_ckpts(cfg); print(f"[{mk}] {len(ck)} ckpt  root={cfg['root']}")
            print(describe_ckpts(ck, read_order(cfg["root"]))); print()
        raise SystemExit(0)
    run(a.model, a.temperature, a.reverse)
