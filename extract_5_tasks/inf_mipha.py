import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
# inf_mipha.py
# Mipha-3B(phi2) 커리큘럼 11시점 ckpt 에서 inf_llava.py 와 동일하게 (1) 추론 → raw_responses json,
# (2) 임베딩 추출 → .pt. 실행 (extract_5_tasks/ 안에서, conda activate mipha):
#   python inf_mipha.py --model mipha                 # forward
#   python inf_mipha.py --model mipha_reverse         # reverse
#   python inf_mipha.py --model mipha_order_14203     # 무작위 순서 (checkpoint/Mipha_order_*_nohex 자동 등록)
#   python inf_mipha.py --model all --list            # 체크포인트 인식만 확인(드라이런)

import sys
import json
import glob
import argparse

import torch
from PIL import Image


MIPHA_REPO = "../Model/llava-phi"
CONV_MODE  = "phi"            
MODEL_BASE = None             
MODEL_NAME = "mipha-phi2"

LAYERS = [0, 8, 16, 24, 32]   # phi-2: num_hidden_layers=32, hidden=2560 → 인덱스 32까지 유효

TOP_P          = 0.7
MAX_NEW_TOKENS = 512
EMB            = "embedding"           # result/{out}/embedding  (probing EMB_ROOT 와 일치)

# 무작위 순서 커리큘럼 런 공용 유틸 — 저장소 루트의 order_runs.py
_ORDER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ORDER_ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402
CKPT_ROOT = os.path.join(_ORDER_ROOT, "checkpoint")

# inf_llava/inf_vila 와 같은 패턴. forward / reverse 는 (체크포인트 root, 출력 폴더)만 다름.
# 출력: result/{out}/inference/raw_responses/{ckpt}/{task}.json , result/{out}/embedding/{ckpt}/{task}.pt
_BASE = "../Model/llava-phi/Mipha-3B-phi2"
MODELS = {
    "mipha":         dict(base=_BASE, root="../checkpoint/Mipha",         out="mipha"),
    "mipha_reverse": dict(base=_BASE, root="../checkpoint/Mipha_reverse", out="mipha_reverse"),
}
# 무작위 순서 런 자동 등록: Mipha_order_<tag>_nohex → "mipha_order_<tag>"
MODELS.update(discover_order_runs(CKPT_ROOT, "Mipha", "mipha", base=_BASE))

# stage 폴더명 (Mipha 는 "-phi2" 접미사)
STAGES = {
    "S0": "S0_LogMAR-phi2",
    "S1": "S1_Similarity-phi2",
    "S2": "S2_Ishihara-phi2",
    "S3": "S3_Naming-phi2",
    "S4": "S4_Depth-phi2",
}


def build_ckpts(cfg):
    """{ckpt_name: path}. S0_start=base, mid=checkpoint-*(glob), end=end/. 존재하는 것만."""
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


DATASETS = {
    "S0_logamr":         "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
    # ⚠ S1 은 nohex 가 표준(hex 프롬프트는 정답 누출 → 학습/검증/추론 전부 nohex). 기존 hex 결과와 섞지 말 것.
    "S1_similarity":     "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json",
    "S2_ishihara":  "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
    "S3_naming":         "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "S4_depth":      "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
}



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


sys.path.insert(0, MIPHA_REPO)
from mipha.model.builder import load_pretrained_model
from mipha.mm_utils import (KeywordsStoppingCriteria, process_images,
                            tokenizer_image_token)
from mipha.constants import IMAGE_TOKEN_INDEX
from mipha.conversation import SeparatorStyle, conv_templates


class MiphaInferExtract:
    def __init__(self, ckpt_path):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            ckpt_path, MODEL_BASE, MODEL_NAME)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size

    @torch.inference_mode()
    def run_one(self, image_path, question, layers, temperature):
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        
        # ⭐ MM projector output (텍스트 합쳐지기 전, LLM 입력 직전)
        vision_tower = self.model.get_vision_tower()
        mm_projector = self.model.get_model().mm_projector
        vision_raw = vision_tower(image_tensor)              # (1, P, 1152)
        vision_projected = mm_projector(vision_raw)          # (1, P, 2560)
        proj_feat = vision_projected.mean(dim=1).squeeze(0).float().cpu()

        q = question if "<image>" in question else "<image>\n" + question
        conv = conv_templates[CONV_MODE].copy()   # phi 템플릿
        conv.append_message(conv.roles[0], q)      # 질문
        conv.append_message(conv.roles[1], None)   # 답변 자리 비움
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)
        input_token_len = input_ids.shape[1]       # ★ 디코딩 슬라이싱용

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # 한 번의 generate로 {예측 + hidden state} 획득
        out = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            eos_token_id=self.tokenizer.eos_token_id,   # eval 방식(경고 억제/정상 종료)
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        # ---- (1) 예측: Mipha는 입력+생성을 함께 반환 → 입력부 잘라냄 ----
        gen_ids = out.sequences[:, input_token_len:]
        prediction = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True)[0].strip()
        if prediction.endswith(stop_str):
            prediction = prediction[:-len(stop_str)].strip()

        # ---- (2) 임베딩: 첫 스텝(prefill)의 마지막 토큰 (VILA와 동일) ----
        prefill = out.hidden_states[0]
        feat = torch.stack([prefill[L][0, -1, :].float().cpu() for L in layers])

        return prediction, feat, proj_feat


# ============================================================
# 데이터셋 1개 처리 → 예측 json + 임베딩 pt
# ============================================================
def process_dataset(loader, cfg, ckpt_name, ds_name, ds_path, temperature):
    # 재개: 예측 json + 임베딩 pt 가 둘 다 있으면 건너뜀 (중간에 죽어도 이어서 돌릴 수 있게; qwen/blind 와 동일)
    if (os.path.exists(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", cfg["out"], "inference", "raw_responses", ckpt_name), f"{ds_name}.json"))
            and os.path.exists(os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", cfg["out"], EMB, ckpt_name), f"{ds_name}.pt"))):
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

    feats = torch.stack(feats).half()   # (N, len(LAYERS), hidden) fp16
    proj_feats = torch.stack(proj_feats).half()  # (N, 2560) fp16

    # ---- 예측 json → predictions/{ckpt}/{dataset}.json ----
    out_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result", cfg["out"])
    pred_dir = os.path.join(out_root, "inference", "raw_responses", ckpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{ds_name}.json")
    with open(pred_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # ---- 임베딩 pt → embeddings/{ckpt}/{dataset}.pt ----
    emb_dir = os.path.join(out_root, EMB, ckpt_name)
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
# 실행 — 모델 × 데이터 이중 루프
# ============================================================
def run(model, temperature):
    cfg = MODELS[model]
    ckpts = build_ckpts(cfg)
    print(f"[{model}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {model} / {ckpt_name} ##########")
        loader = MiphaInferExtract(ckpt_path)
        for ds_name in DATASETS:
            print(f"  -- {ds_name} --")
            process_dataset(loader, cfg, ckpt_name, ds_name,
                            DATASETS[ds_name], temperature)
        del loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="mipha", choices=list(MODELS) + ["all"])
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
