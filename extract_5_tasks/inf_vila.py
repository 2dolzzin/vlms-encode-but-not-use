import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import json
import glob
import argparse

import torch
from PIL import Image


# ============================================================
# 공통 설정
# ============================================================
VILA_REPO  = "../Model/VILA"
MODEL_BASE = "llava_llama"        # load_pretrained_model 의 model_base (3b/8b 동일)

LAYERS = [0, 8, 16, 24, 32]
TOP_P          = 0.7
MAX_NEW_TOKENS = 512

# 모델별로 다른 건 (conv 모드 / 베이스 모델 / 체크포인트 루트 / 출력 폴더)뿐.
# 체크포인트 step 번호는 3b=1563, 8b=미정(학습 중단) → build_ckpts 가 glob 으로 자동 탐색.
MODELS = {
    "vila_3b": dict(
        conv="vicuna_v1",
        base="../Model/VILA/VILA1.5-3b",
        root="../checkpoint/VILA_3B",
        out="vila_3b",
        emb="embedding",          # result/vila_3b/embedding  (probing MODEL_EMB 와 일치)
    ),
    "vila_8b": dict(
        conv="llama_3",
        base="../Model/VILA/Llama-3-VILA1.5-8B",
        root="../checkpoint/VILA_8B",
        out="vila_8b",
        emb="embedding",
    ),
}

# 무작위 순서 커리큘럼 런(checkpoint/<Model>_order_<tag>_nohex) 공용 유틸 — 저장소 루트의 order_runs.py
_ORDER_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ORDER_ROOT)
from order_runs import stage_dirs, discover_order_runs, read_order, describe_ckpts  # noqa: E402
CKPT_ROOT = os.path.join(_ORDER_ROOT, "checkpoint")
# 무작위 순서 런 자동 등록: VILA_3B_order_<tag>_nohex → "vila_3b_order_<tag>", VILA_8B_... → "vila_8b_order_<tag>"
MODELS.update(discover_order_runs(CKPT_ROOT, "VILA_3B", "vila_3b", conv="vicuna_v1",
                                  base="../Model/VILA/VILA1.5-3b", emb="embedding"))
MODELS.update(discover_order_runs(CKPT_ROOT, "VILA_8B", "vila_8b", conv="llama_3",
                                  base="../Model/VILA/Llama-3-VILA1.5-8B", emb="embedding"))

# stage 폴더명 (3b/8b 공통). {stage_id: 체크포인트 하위 디렉토리}
STAGES = {
    "S0": "S0_LogMAR",
    "S1": "S1_Similarity",
    "S2": "S2_Ishihara",
    "S3": "S3_Naming",
    "S4": "S4_Depth",
}

DATASETS = {
    "S0_logamr":         "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
    # ⚠ S1 은 nohex 가 표준(hex 프롬프트는 정답 누출 → 학습/검증/추론 전부 nohex). 기존 hex 결과와 섞지 말 것.
    "S1_similarity":     "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json",
    "S2_ishihara":  "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
    "S3_naming":         "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "S4_depth":      "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
}


def build_ckpts(cfg):
    """{ckpt_name: path} 자동 구성. mid = checkpoint-*(glob), end = end/.
    실제 존재하는 경로만 담는다 (8b 학습 중단 → 일부 stage 가 없을 수 있음)."""
    root, base = cfg["root"], cfg["base"]
    ck = {"S0_start": base}
    for sid, sdir in stage_dirs(root, STAGES).items():
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")))
        if len(mids) > 1:
            print(f"[warn] {sdir} 에 checkpoint-* 가 여러 개 → {mids[-1]} 를 mid 로 사용")
        if mids:
            ck[f"{sid}_mid"] = mids[-1]
        end = os.path.join(sdir, "end")
        if os.path.isdir(end):
            ck[f"{sid}_end"] = end
    # 존재하는 것만 남김 (S0_start=base 는 항상 포함)
    return {k: v for k, v in ck.items() if k == "S0_start" or os.path.exists(v)}


# ============================================================
# 데이터 로드 — conversations 포맷
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
# VILA loader
# ============================================================
sys.path.insert(0, VILA_REPO)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (KeywordsStoppingCriteria, process_images,
                            tokenizer_image_token)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates


class VILAInferExtract:
    def __init__(self, ckpt_path, conv_mode):
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            ckpt_path, MODEL_BASE, None)
        self.model.eval()
        self.hidden_size = self.model.config.hidden_size
        self.conv_mode = conv_mode

    @torch.inference_mode()
    def run_one(self, image_path, question, layers, temperature):
        image = Image.open(image_path).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        # ⭐ MM projector output (텍스트 합쳐지기 전, LLM 입력 직전 비전 표현)
        vision_tower = self.model.get_vision_tower()
        mm_projector = self.model.get_mm_projector()
        vision_raw = vision_tower(image_tensor)              # (1, P, vis_dim)
        vision_projected = mm_projector(vision_raw)          # (1, P, hidden)
        proj_feat = vision_projected.mean(dim=1).squeeze(0).float().cpu()

        q = question if "<image>" in question else "<image>\n" + question
        conv = conv_templates[self.conv_mode].copy()   # 3b=vicuna_v1 / 8b=llama_3
        conv.append_message(conv.roles[0], q)          # 질문 추가
        conv.append_message(conv.roles[1], None)       # 답변 자리 추가
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        input_ids = torch.as_tensor(input_ids).cuda().unsqueeze(0)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

        # generate 로 {예측 + hidden state} 획득
        out = self.model.generate(
            input_ids,
            images = image_tensor,
            do_sample = temperature > 0,
            temperature = temperature,
            top_p = TOP_P,
            max_new_tokens = MAX_NEW_TOKENS,
            use_cache = True,
            stopping_criteria = [stopping_criteria],
            return_dict_in_generate = True,   # sequences + hidden_states
            output_hidden_states = True,
        )

        # ---- 예측: VILA generate 는 inputs_embeds 로 호출 → sequences 에 생성 토큰만 담김
        #      (입력 프롬프트 토큰 미포함) 이므로 슬라이싱하지 말고 그대로 디코딩 ----
        gen_ids = out.sequences
        prediction = self.tokenizer.batch_decode(
            gen_ids, skip_special_tokens=True)[0].strip()
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
            and os.path.exists(os.path.join(os.path.join("result", cfg["out"], cfg["emb"], ckpt_name), f"{ds_name}.pt"))):
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

    # ---- 예측 json → raw_responses/{ckpt}/{dataset}.json ----
    pred_dir = os.path.join("result", cfg["out"], "inference", "raw_responses", ckpt_name)
    os.makedirs(pred_dir, exist_ok=True)
    pred_path = os.path.join(pred_dir, f"{ds_name}.json")
    with open(pred_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # ---- 임베딩 pt → {emb}/{ckpt}/{dataset}.pt ----
    emb_dir = os.path.join("result", cfg["out"], cfg["emb"], ckpt_name)
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
def run(model, temperature, reverse=False):
    cfg = dict(MODELS[model])
    tag = model
    if reverse and "_order_" in model:
        raise SystemExit(f"--reverse 는 forward 런에만 쓴다. {model} 은 무작위 순서 런이라 폴더 자체가 순서를 담고 있음.")
    if reverse:
        # 리버스 커리큘럼: 체크포인트 루트와 출력 폴더만 _reverse 로 스위치.
        # stage 하위 폴더명(S0_LogMAR..)은 fwd/rev 동일.
        cfg["root"] = cfg["root"] + "_reverse"   # ../checkpoint/VILA_3B → _reverse
        cfg["out"]  = cfg["out"]  + "_reverse"   # result/vila_3b → vila_3b_reverse
        tag = model + "_reverse"
    ckpts = build_ckpts(cfg)
    print(f"[{tag}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {tag} / {ckpt_name} ##########")
        loader = VILAInferExtract(ckpt_path, cfg["conv"])
        for ds_name in DATASETS:
            print(f"  -- {ds_name} --")
            process_dataset(loader, cfg, ckpt_name, ds_name,
                            DATASETS[ds_name], temperature)
        del loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="vila_3b", choices=list(MODELS) + ["all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reverse", action="store_true",
                    help="리버스 커리큘럼 체크포인트(VILA_*_reverse) → result/{model}_reverse")
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
