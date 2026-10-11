import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# inf_llava.py
# LLaVA-1.5-7b 커리큘럼 11시점 ckpt에서 inf_vila.py / inf_mipha.py 와 동일하게
#   (1) 추론 → 예측 raw_responses json 저장
#   (2) 임베딩 추출 → proj(비전 인코더+mm_projector 직후) + layer 0/8/16/24/32 의
#       prefill 마지막 토큰 hidden state 를 .pt 로 저장
#
# 출력 (vila_3b 와 동일 레이아웃 → probing 이 result/llava/embedding 로 바로 사용 가능):
#   result/llava/inference/raw_responses/{ckpt}/{task}.json
#   result/llava/embedding/{ckpt}/{task}.pt
#       { feats:(N,len(LAYERS),hidden) fp16, proj_feats:(N,hidden) fp16,
#         predictions, answers, ids, layers, hidden_size }
#
# 실행:
#   python inf_llava.py                       # 11시점 × 5task
#   python inf_llava.py --temperature 0.0

import sys
import json
import glob
import re
import argparse

import torch
from PIL import Image


# ============================================================
# 설정 — 모델 고유값만 vila/mipha 와 다름
# ============================================================
LLAVA_REPO = "../Model/LLaVA"
CONV_MODE  = "llava_v1"
MODEL_NAME = "llava-v1.5-7b"   # builder 가 'llava' 포함여부로 멀티모달 로딩 분기
MODEL_BASE = None              # full checkpoint → base 없음

LAYERS = [0, 8, 16, 24, 32]    # vicuna-7b: 32 layer, hidden=4096 → index 32 유효
EMB    = "embedding"           # result/{out}/embedding

TOP_P          = 0.7
MAX_NEW_TOKENS = 512

# vila MODELS 패턴 모방 — forward / reverse 는 (체크포인트 root, 출력 폴더)만 다름.
# 리버스 학습은 체크포인트 가중치에 들어있으므로 stage 폴더명·base·로직은 동일.
MODELS = {
    "llava": dict(
        base="../Model/LLaVA/llava-v1.5-7b",
        root="../checkpoint/llava",
        out="llava",              # result/llava
    ),
    "llava_reverse": dict(
        base="../Model/LLaVA/llava-v1.5-7b",
        root="../checkpoint/llava_reverse",      # 디스크 실제 폴더명 (구 llava_2_reverse 는 없음)
        out="llava_reverse",      # result/llava_reverse
    ),
}

# stage 폴더명 (llava 체크포인트 디렉토리, 대소문자 주의: S0 는 LogMAR)
# 무작위 순서 커리큘럼 런(checkpoint/<Model>_order_<tag>_nohex) 공용 유틸 — 저장소 루트의 order_runs.py
_ORDER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ORDER_ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402
CKPT_ROOT = os.path.join(_ORDER_ROOT, "checkpoint")
# 무작위 순서 런 자동 등록: checkpoint/llava_order_<tag>_nohex → "llava_order_<tag>" (result/llava_order_<tag>)
MODELS.update(discover_order_runs(CKPT_ROOT, "llava", "llava", base="../Model/LLaVA/llava-v1.5-7b"))

STAGES = {
    "S0": "S0_LogMAR",
    "S1": "S1_Similarity",
    "S2": "S2_Ishihara",
    "S3": "S3_Naming",
    "S4": "S4_Depth",
}

# 데이터셋 (시점 무관 동일, vila/mipha 와 같은 파일)
DATASETS = {
    "S0_logamr":         "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
    # ⚠ S1 은 nohex 가 표준(hex 프롬프트는 정답 누출 → 학습/검증/추론 전부 nohex). 기존 hex 결과와 섞지 말 것.
    "S1_similarity":     "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json",
    "S2_ishihara":  "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
    "S3_naming":         "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "S4_depth":      "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
}



def build_ckpts(cfg):
    """{ckpt_name: path} 자동 구성. S0_start=base, mid=checkpoint-*, end=end/.
    실제 존재하는 경로만 담는다."""
    ck = {"S0_start": cfg["base"]}
    for sid, sdir in stage_dirs(cfg["root"], STAGES).items():
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")))
        if len(mids) > 1:
            print(f"[warn] {sdir} 에 checkpoint-* 여러 개 → {mids[-1]} 를 mid 로 사용")
        if mids:
            ck[f"{sid}_mid"] = mids[-1]
        end = os.path.join(sdir, "end")
        if os.path.isdir(end):
            ck[f"{sid}_end"] = end
    return {k: v for k, v in ck.items() if k == "S0_start" or os.path.exists(v)}


# ============================================================
# 데이터 로드 — conversations 포맷 (vila/mipha 와 동일)
# ============================================================
def load_dataset(path):
    with open(path) as f:
        cases = json.load(f)
    out = []
    for c in cases:
        out.append({
            "id":         str(c.get("id", "")),
            "image_path": c["image"],
            "question":   c["conversations"][0]["value"],   # [0]=질문
            "answer":     c["conversations"][1]["value"],   # [1]=정답
        })
    return out


# ============================================================
# LLaVA loader
# ============================================================
sys.path.insert(0, LLAVA_REPO)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (KeywordsStoppingCriteria, process_images,
                            tokenizer_image_token)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates


class LLaVAInferExtract:
    def __init__(self, ckpt_path):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            ckpt_path, MODEL_BASE, MODEL_NAME)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.inference_mode()
    def run_one(self, image_path, question, layers, temperature):
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # ⭐ MM projector output (텍스트 합쳐지기 전, LLM 입력 직전 비전 표현)
        vision_tower = self.model.get_vision_tower()
        mm_projector = self.model.get_model().mm_projector
        vision_raw = vision_tower(image_tensor)              # (1, P, vis_dim)
        vision_projected = mm_projector(vision_raw)          # (1, P, hidden)
        proj_feat = vision_projected.mean(dim=1).squeeze(0).float().cpu()

        q = question if "<image>" in question else "<image>\n" + question
        conv = conv_templates[CONV_MODE].copy()        # llava_v1 템플릿
        conv.append_message(conv.roles[0], q)          # 질문
        conv.append_message(conv.roles[1], None)       # 답변 자리 비움
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # 한 번의 generate 로 {예측 + hidden state} 획득
        out = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            return_dict_in_generate=True,   # sequences + hidden_states
            output_hidden_states=True,
        )

        # ---- 예측: LLaVA generate 는 inputs_embeds 로 호출 → sequences 에 생성 토큰만
        #      담김 (VILA 와 동일, 입력부 슬라이싱 불필요) ----
        prediction = self.tokenizer.batch_decode(
            out.sequences, skip_special_tokens=True)[0].strip()
        if prediction.endswith(stop_str):
            prediction = prediction[:-len(stop_str)].strip()

        # ---- 임베딩: 첫 스텝(prefill)의 마지막 토큰 위치 hidden state ----
        prefill = out.hidden_states[0]
        feat = torch.stack([prefill[L][0, -1, :].float().cpu() for L in layers])

        return prediction, feat, proj_feat


# ============================================================
# 데이터셋 1개 처리 → 예측 json + 임베딩 pt
# ============================================================
def process_dataset(loader, cfg, ckpt_name, ds_name, ds_path, temperature):
    # 재개: 예측 json + 임베딩 pt 가 둘 다 있으면 건너뜀 (중간에 죽어도 이어서 돌릴 수 있게; qwen/blind 와 동일)
    if (os.path.exists(os.path.join(os.path.join("result", cfg["out"], "inference", "raw_responses", ckpt_name), f"{ds_name}.json"))
            and os.path.exists(os.path.join(os.path.join("result", cfg["out"], EMB, ckpt_name), f"{ds_name}.pt"))):
        print(f"[skip] {ckpt_name}/{ds_name} (이미 완료)")
        return
    samples = load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []

    for j, s in enumerate(samples):
        pred, feat, p_feat = loader.run_one(
            s["image_path"], s["question"], LAYERS, temperature)

        records.append(dict(
            id=s["id"], question=s["question"],
            output=pred, expected_output=s["answer"],
        ))
        feats.append(feat)
        proj_feats.append(p_feat)
        predictions.append(pred)
        answers.append(s["answer"])
        ids.append(s["id"])
        if (j + 1) % 50 == 0:
            print(f"    ... {j+1}/{len(samples)}")

    feats = torch.stack(feats).half()              # (N, len(LAYERS), hidden) fp16
    proj_feats = torch.stack(proj_feats).half()    # (N, hidden) fp16

    # ---- 예측 json → result/{out}/inference/raw_responses/{ckpt}/{ds}.json ----
    pred_dir = os.path.join("result", cfg["out"], "inference", "raw_responses", ckpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{ds_name}.json")
    with open(pred_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # ---- 임베딩 pt → result/{out}/embedding/{ckpt}/{ds}.pt ----
    emb_dir = os.path.join("result", cfg["out"], EMB, ckpt_name)
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, f"{ds_name}.pt")
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


# ============================================================
# 실행 — ckpt × dataset 이중 루프
# ============================================================
def run(model, temperature):
    cfg = MODELS[model]
    ckpts = build_ckpts(cfg)
    print(f"[{model}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {model} / {ckpt_name} ##########")
        loader = LLaVAInferExtract(ckpt_path)
        for ds_name in DATASETS:
            print(f"  -- {ds_name} --")
            process_dataset(loader, cfg, ckpt_name, ds_name,
                            DATASETS[ds_name], temperature)
        del loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="llava", choices=list(MODELS) + ["all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--list", action="store_true",
                    help="모델 로드 없이 run 별 체크포인트 11시점만 출력(드라이런)")
    a = ap.parse_args()
    if a.list:
        for mk in (list(MODELS) if a.model == "all" else [a.model]):
            ck = build_ckpts(MODELS[mk]); print(f"[{mk}] {len(ck)} ckpt  root={MODELS[mk]['root']}")
            print(describe_ckpts(ck, read_order(MODELS[mk]["root"]))); print()
        raise SystemExit(0)
    run(a.model, a.temperature)
