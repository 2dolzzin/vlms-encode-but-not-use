import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# inf_qwen25vl.py
# Qwen2.5-VL-7B-Instruct 커리큘럼 11시점 ckpt 에서 inf_llava.py / inf_vila.py 와 동일하게
#   (1) 추론 → 예측 raw_responses json 저장
#   (2) 임베딩 추출 → proj(비전 인코더+merger 직후, 이미지 토큰 평균) + layer 0/7/14/21/28 의
#       prefill 마지막 토큰 hidden state 를 .pt 로 저장
#
# 출력 (llava 와 동일 레이아웃 → probing --model qwen25vl_7b 로 바로 사용 가능):
#   result/qwen25vl_7b/inference/raw_responses/{ckpt}/{task}.json
#   result/qwen25vl_7b/embedding/{ckpt}/{task}.pt
#       { feats:(N,len(LAYERS),hidden) fp16, proj_feats:(N,hidden) fp16,
#         predictions, answers, ids, layers, hidden_size }
#
# 실행 (extract_5_tasks/ 안에서, conda activate qwenvl):
#   python inf_qwen25vl.py --model qwen25vl_7b
#   python inf_qwen25vl.py --model qwen25vl_7b_reverse
#   python inf_qwen25vl.py --model qwen25vl_7b --ckpt S0_start,S4_end   # 일부 ckpt 만
#
# 채점: python acc_parsing.py --model qwen25vl_7b

import sys
import json
import glob
import argparse

import torch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT, "train", "qwen25vl_curriculum"))
from qwen_data import (DEFAULT_MAX_PIXELS, DEFAULT_MIN_PIXELS, build_prompt,  # noqa: E402
                       get_visual, load_model, load_processor, text_config)


# ============================================================
# 설정 — 모델 고유값만 llava/vila 와 다름
# ============================================================
# Qwen2.5-VL-7B: LLM 28 layer, hidden 3584 → 인덱스 28 유효. 5 지점 등간격 (llava 0/8/16/24/32 대응)
LAYERS = [0, 7, 14, 21, 28]
EMB    = "embedding"           # result/{out}/embedding

TOP_P          = 0.7
MAX_NEW_TOKENS = 512

BASE_MODEL = os.path.join(ROOT, "Model/Qwen2.5-VL-7B-Instruct")

MODELS = {
    "qwen25vl_7b": dict(
        base=BASE_MODEL,
        root=os.path.join(ROOT, "checkpoint/Qwen25VL_7B"),
        out="qwen25vl_7b",              # result/qwen25vl_7b
    ),
    "qwen25vl_7b_reverse": dict(
        base=BASE_MODEL,
        root=os.path.join(ROOT, "checkpoint/Qwen25VL_7B_reverse"),
        out="qwen25vl_7b_reverse",      # result/qwen25vl_7b_reverse
    ),
}

# 무작위 순서 커리큘럼 런(checkpoint/<Model>_order_<tag>_nohex) 공용 유틸 — 저장소 루트의 order_runs.py
_ORDER_ROOT = ROOT
sys.path.insert(0, _ORDER_ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402
CKPT_ROOT = os.path.join(_ORDER_ROOT, "checkpoint")
# 무작위 순서 런 자동 등록: Qwen25VL_7B_order_<tag>_nohex → "qwen25vl_7b_order_<tag>"
MODELS.update(discover_order_runs(CKPT_ROOT, "Qwen25VL_7B", "qwen25vl_7b", base=BASE_MODEL))

STAGES = {
    "S0": "S0_LogMAR",
    "S1": "S1_Similarity",
    "S2": "S2_Ishihara",
    "S3": "S3_Naming",
    "S4": "S4_Depth",
}

DATASETS = {
    "S0_logamr":     "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
    # ⚠ S1 은 nohex(프롬프트에서 hex 코드 제거) 버전. 학습/검증/추론 전부 nohex 로 통일.
    #    hex 프롬프트는 (Left:#2800ff, Right:#0028ff) 처럼 정답을 흘려서 이미지를 안 봐도 풀린다.
    "S1_similarity": "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json",
    "S2_ishihara":   "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
    "S3_naming":     "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "S4_depth":      "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
}


def _is_ready(d):
    """학습이 끝나 가중치까지 저장된 폴더만 체크포인트로 인정.
    _stage.sh 가 시작할 때 빈 end/ 를 먼저 만들기 때문에, 학습과 추론을 동시에 돌리면
    빈 폴더를 완성된 ckpt 로 오인해 로딩이 터진다. config.json + safetensors 존재로 판정."""
    return (os.path.isdir(d)
            and os.path.exists(os.path.join(d, "config.json"))
            and bool(glob.glob(os.path.join(d, "*.safetensors"))))


def build_ckpts(cfg):
    """{ckpt_name: path}. S0_start=base, mid=checkpoint-*, end=end/. 존재하는 것만."""
    ck = {"S0_start": cfg["base"]}
    for sid, sdir in stage_dirs(cfg["root"], STAGES).items():
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")))
        if len(mids) > 1:
            print(f"[warn] {sdir} 에 checkpoint-* 여러 개 → {mids[-1]} 를 mid 로 사용")
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
        "id":         str(c.get("id", "")),
        "image_path": c["image"],
        "question":   c["conversations"][0]["value"],
        "answer":     c["conversations"][1]["value"],
    } for c in cases]


# ============================================================
# Qwen2.5-VL loader
# ============================================================
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

        # ⭐ projector(merger) 출력 = LLM 입력 직전 비전 표현 → 이미지 토큰 평균 (llava mm_projector 평균과 대응)
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

        # HF generate: sequences 에 입력 프롬프트 포함 → 잘라냄
        n_in = inputs["input_ids"].shape[1]
        prediction = self.processor.tokenizer.decode(
            out.sequences[0, n_in:], skip_special_tokens=True).strip()

        # 임베딩: 첫 스텝(prefill) 마지막 토큰 hidden state. hidden_states[0][L] : (1, seq, hidden), L=0..28
        prefill = out.hidden_states[0]
        feat = torch.stack([prefill[L][0, -1, :].float().cpu() for L in layers])
        return prediction, feat, proj_feat


# ============================================================
# 데이터셋 1개 처리 → 예측 json + 임베딩 pt
# ============================================================
def process_dataset(loader, cfg, ckpt_name, ds_name, ds_path, temperature):
    pred_dir = os.path.join(BASE, "result", cfg["out"], "inference", "raw_responses", ckpt_name)
    emb_dir = os.path.join(BASE, "result", cfg["out"], EMB, ckpt_name)
    pred_path = os.path.join(pred_dir, f"{ds_name}.json")
    emb_path = os.path.join(emb_dir, f"{ds_name}.pt")
    if os.path.exists(pred_path) and os.path.exists(emb_path):
        print(f"[skip] {ckpt_name}/{ds_name} (이미 완료)")
        return

    samples = load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []

    for j, s in enumerate(samples):
        pred, feat, p_feat = loader.run_one(s["image_path"], s["question"], LAYERS, temperature)
        records.append(dict(id=s["id"], question=s["question"],
                            output=pred, expected_output=s["answer"]))
        feats.append(feat)
        proj_feats.append(p_feat)
        predictions.append(pred)
        answers.append(s["answer"])
        ids.append(s["id"])
        if (j + 1) % 50 == 0:
            print(f"    ... {j+1}/{len(samples)}", flush=True)

    feats = torch.stack(feats).half()              # (N, len(LAYERS), hidden) fp16
    proj_feats = torch.stack(proj_feats).half()    # (N, hidden) fp16

    os.makedirs(pred_dir, exist_ok=True)
    with open(pred_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    os.makedirs(emb_dir, exist_ok=True)
    torch.save({
        "feats": feats,
        "proj_feats": proj_feats,
        "predictions": predictions,
        "answers": answers,
        "ids": ids,
        "layers": LAYERS,
        "hidden_size": loader.hidden_size,
    }, emb_path)

    print(f"[saved] {ckpt_name}/{ds_name} → {pred_path}")
    print(f"        {emb_path}  feats={tuple(feats.shape)} proj={tuple(proj_feats.shape)}")


def run(model, temperature, only_ckpts=None, only_tasks=None):
    cfg = MODELS[model]
    ckpts = build_ckpts(cfg)
    if only_ckpts:
        ckpts = {k: v for k, v in ckpts.items() if k in only_ckpts}
    tasks = {k: v for k, v in DATASETS.items() if not only_tasks or k in only_tasks}
    print(f"[{model}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}  tasks={list(tasks)}")
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {model} / {ckpt_name} ##########")
        loader = QwenInferExtract(ckpt_path)
        for ds_name in tasks:
            print(f"  -- {ds_name} --")
            process_dataset(loader, cfg, ckpt_name, ds_name, tasks[ds_name], temperature)
        del loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="qwen25vl_7b", choices=list(MODELS) + ["all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--ckpt", type=str, default=None, help="쉼표구분 (예: S0_start,S4_end). 미지정=전체")
    ap.add_argument("--task", type=str, default=None, help="쉼표구분 (예: S1_similarity). 미지정=전체")
    ap.add_argument("--list", action="store_true",
                    help="모델 로드 없이 run 별 체크포인트 11시점만 출력(드라이런)")
    a = ap.parse_args()
    if a.list:
        for mk in (list(MODELS) if a.model == "all" else [a.model]):
            ck = build_ckpts(MODELS[mk]); print(f"[{mk}] {len(ck)} ckpt  root={MODELS[mk]['root']}")
            print(describe_ckpts(ck, read_order(MODELS[mk]["root"]))); print()
        raise SystemExit(0)
    run(a.model, a.temperature,
        only_ckpts=a.ckpt.split(",") if a.ckpt else None,
        only_tasks=a.task.split(",") if a.task else None)
