# extract_s4_tokenlevel.py
# Experiment B 특징 추출: S4_Depth 449개에 대해 projector 출력(mean pooling 전, LM 입력
# 직전)을 토큰 단위로 뽑아, 추출 시점에 AdaptiveAvgPool2d((2,2))/((4,4))로 풀링해
# [N,4,d], [N,16,d]만 저장한다 (원본 [N,T,d] 저장은 용량 문제로 생략 — 스펙에서 허용).
#
# "2x2/4x4 그리드"는 커널풀링이 아니라 패치 그리드 전체를 2x2=4개 / 4x4=16개 영역으로
# 나누는 것 — AdaptiveAvgPool2d로 구현, 나눠떨어지지 않아도(Mipha 27x27, LLaVA 24x24,
# VILA 14x14 전부) 자동으로 경계를 배분한다.
#
# generate() 없이 vision_tower+projector forward만 수행 (probing에 LM 출력 불필요) → 빠름.
# OpenFlamingo는 제외 (Perceiver Resampler, 공간 배치 없음).
#
# 실행 (extract_5_tasks/ 안에서, 모델별 conda env):
#   python s1_rebuttal/extract_s4_tokenlevel.py --model llava

import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import sys
import json
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXTRACT5)
sys.path.insert(0, EXTRACT5)

S4_DATASET = "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json"
OUT_ROOT = os.path.join(EXTRACT5, "result_expB_s4_tokenlevel")

GRID = {"llava": (24, 24), "mipha": (27, 27), "vila_3b": (14, 14), "vila_8b": (14, 14)}

MODEL_CFG = {
    "llava": dict(base="../Model/LLaVA/llava-v1.5-7b", root_fwd="../checkpoint/llava",
                 root_rev="../checkpoint/llava_reverse"),
    "mipha": dict(base="../Model/llava-phi/Mipha-3B-phi2", root_fwd="../checkpoint/Mipha",
                 root_rev="../checkpoint/Mipha_reverse", stage_suffix="-phi2"),
    "vila_3b": dict(base="../Model/VILA/VILA1.5-3b", root_fwd="../checkpoint/VILA_3B",
                    root_rev="../checkpoint/VILA_3B_reverse"),
    "vila_8b": dict(base="../Model/VILA/Llama-3-VILA1.5-8B", root_fwd="../checkpoint/VILA_8B",
                    root_rev="../checkpoint/VILA_8B_reverse"),
}

STAGES = {"S0": "S0_LogMAR", "S1": "S1_Similarity", "S2": "S2_Ishihara",
         "S3": "S3_Naming", "S4": "S4_Depth"}


def build_ckpts(cfg, direction):
    root = cfg["root_fwd"] if direction == "fwd" else cfg["root_rev"]
    suffix = cfg.get("stage_suffix", "")
    import glob
    ck = {"S0_start": cfg["base"]}
    for sid, dname in STAGES.items():
        sdir = os.path.join(root, dname + suffix)
        mids = sorted(glob.glob(os.path.join(sdir, "checkpoint-*")))
        if mids:
            ck[f"{sid}_mid"] = mids[-1]
        end = os.path.join(sdir, "end")
        if os.path.isdir(end):
            ck[f"{sid}_end"] = end
    return {k: v for k, v in ck.items() if k == "S0_start" or os.path.exists(v)}


def load_llava(ckpt):
    import inf_llava
    from llava.model.builder import load_pretrained_model
    tok, model, imgproc, _ = load_pretrained_model(ckpt, None, "llava-v1.5-7b")
    model.eval()
    return model, imgproc


def load_mipha(ckpt):
    sys.path.insert(0, "../Model/llava-phi")
    from mipha.model.builder import load_pretrained_model
    tok, model, imgproc, _ = load_pretrained_model(ckpt, None, "mipha-phi2")
    model.eval()
    return model, imgproc


def load_vila(ckpt):
    from llava.model.builder import load_pretrained_model
    tok, model, imgproc, _ = load_pretrained_model(ckpt, "llava_llama", None)
    model.eval()
    return model, imgproc


LOADERS = {"llava": load_llava, "mipha": load_mipha, "vila_3b": load_vila, "vila_8b": load_vila}

# repo path 삽입 (모델별 mm_utils.process_images 사용 위함)
REPO = {"llava": "../Model/LLaVA", "mipha": "../Model/llava-phi", "vila_3b": "../Model/VILA", "vila_8b": "../Model/VILA"}


def get_process_images(model_key):
    sys.path.insert(0, REPO[model_key])
    if model_key == "mipha":
        from mipha.mm_utils import process_images
    else:
        from llava.mm_utils import process_images
    return process_images


def pool_grid(vision_projected, H, W):
    """vision_projected: (1, T, d) -> (1,d,H,W) -> AdaptiveAvgPool2d -> (4,d),(16,d)"""
    d = vision_projected.shape[-1]
    grid = vision_projected.reshape(1, H, W, d).permute(0, 3, 1, 2)  # (1,d,H,W)
    p2 = F.adaptive_avg_pool2d(grid, (2, 2)).permute(0, 2, 3, 1).reshape(4, d)
    p4 = F.adaptive_avg_pool2d(grid, (4, 4)).permute(0, 2, 3, 1).reshape(16, d)
    return p2.float().cpu(), p4.float().cpu()


@torch.no_grad()
def extract_ckpt(model_key, ckpt_path, samples):
    load_fn = LOADERS[model_key]
    model, imgproc = load_fn(ckpt_path)
    process_images = get_process_images(model_key)
    H, W = GRID[model_key]

    feats_2x2, feats_4x4, answers, ids = [], [], [], []
    for s in samples:
        image = Image.open(s["image"]).convert("RGB")
        image_tensor = process_images([image], imgproc, model.config)
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        vision_tower = model.get_vision_tower()
        if model_key in ("vila_3b", "vila_8b"):
            mm_projector = model.get_mm_projector()
        else:
            mm_projector = model.get_model().mm_projector
        vision_raw = vision_tower(image_tensor)
        vision_projected = mm_projector(vision_raw)  # (1, T, d)

        p2, p4 = pool_grid(vision_projected, H, W)
        feats_2x2.append(p2)
        feats_4x4.append(p4)
        answers.append(s["answer"])
        ids.append(s["id"])

    del model
    torch.cuda.empty_cache()
    return {
        "feats_2x2": torch.stack(feats_2x2).half(),   # (N,4,d)
        "feats_4x4": torch.stack(feats_4x4).half(),   # (N,16,d)
        "answers": answers, "ids": ids, "grid": [H, W],
    }


def load_s4_samples():
    cases = json.load(open(S4_DATASET))
    return [{"id": str(c["id"]), "image": c["image"], "answer": c["conversations"][1]["value"]}
            for c in cases]


def run(model_key):
    cfg = MODEL_CFG[model_key]
    samples = load_s4_samples()
    for direction, tag in [("fwd", model_key), ("rev", f"{model_key}_reverse")]:
        ckpts = build_ckpts(cfg, direction)
        print(f"[{tag}] 체크포인트 {len(ckpts)}개: {list(ckpts)}")
        out_dir = os.path.join(OUT_ROOT, tag)
        os.makedirs(out_dir, exist_ok=True)
        for ckpt_name, ckpt_path in ckpts.items():
            out_path = os.path.join(out_dir, f"{ckpt_name}.pt")
            if os.path.exists(out_path):
                print(f"  [cached] {ckpt_name}")
                continue
            print(f"  extracting {ckpt_name} ...")
            data = extract_ckpt(model_key, ckpt_path, samples)
            torch.save(data, out_path)
            print(f"  [saved] {ckpt_name} -> {out_path}  2x2={tuple(data['feats_2x2'].shape)} 4x4={tuple(data['feats_4x4'].shape)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG))
    a = ap.parse_args()
    run(a.model)
