# hex_contam_check.py
# 기존 hex-포함 프롬프트로 학습된 체크포인트(Base/S1_fwd/S1_rev)에 대해, hex 제거 프롬프트
# (dataset_c.json = 원본 이미지 + hex 삭제)로 hidden state를 다시 뽑아 LM-depth probe를
# 재학습·평가한다 (probe_regressor.py의 regress_one_layer/make_targets 그대로 재사용,
# RidgeCV 5-fold OOF, probe_regressor.py와 동일한 3 seed).
#
# projector probe는 텍스트와 무관(이미지만 거침)하므로 다시 뽑지 않음 — 기존 값 그대로 인용.
#
# 실행 (extract_5_tasks/ 안에서, llava conda env):
#   python s1_rebuttal/hex_contam_check.py --model llava

import os
import sys
import json
import argparse

import numpy as np
import torch

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXTRACT5)
sys.path.insert(0, EXTRACT5)
sys.path.insert(0, os.path.join(os.path.dirname(EXTRACT5), "probing"))

import probe_regressor as pr  # noqa: E402
pr.USE_WANDB = False

VARIANTS_DIR = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants"
C_NOHEX_FILE = os.path.join(VARIANTS_DIR, "dataset_c.json")   # 원본 이미지 + hex 제거
LAYERS = [0, 8, 16, 24, 32]
TEMPERATURE = 0.0
SEEDS = pr.SEEDS  # [53, 9, 144], probe_regressor.py 와 동일

CKPTS = {
    "llava": {
        "Base":   "/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b",
        "S1_fwd": "/home/hail/vlm3/checkpoint/llava/S1_Similarity/end",
        "S1_rev": "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity/end",
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/llava/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity_nohex/end",
    },
}

OUT_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal", "hex_contam_check")


def get_loader_cls(model):
    if model == "llava":
        import inf_llava
        return inf_llava.LLaVAInferExtract
    raise ValueError(model)


def extract(model, ckpt_name, ckpt_path):
    loader_cls = get_loader_cls(model)
    loader = loader_cls(ckpt_path)
    samples = json.load(open(C_NOHEX_FILE))

    feats, proj_feats, answers, ids = [], [], [], []
    for s in samples:
        pred, feat, proj = loader.run_one(s["image"], s["question"], LAYERS, TEMPERATURE)
        feats.append(feat)
        proj_feats.append(proj)
        answers.append(s["sim_img"])
        ids.append(s["id"])

    feats = torch.stack(feats).float()          # (N, len(LAYERS), hidden)
    proj_feats = torch.stack(proj_feats).float()  # (N, hidden) — 참고용, probe는 layer만 사용

    del loader
    torch.cuda.empty_cache()
    return {"feats": feats, "proj_feats": proj_feats, "answers": answers, "ids": ids, "layers": LAYERS}


def probe_layers(data):
    y = pr.make_targets(data["answers"], task_name="S1_similarity")
    out = {}
    for i, L in enumerate(data["layers"]):
        X = data["feats"][:, i, :].numpy()
        per_seed = []
        for seed in SEEDS:
            pr.SEED = seed
            per_seed.append(pr.regress_one_layer(X, y, f"layer_{L}", run=None, layer=L, probe_type="ridge"))
        out[f"layer_{L}"] = {
            "mae": float(np.mean([r["mae"] for r in per_seed])),
            "pearson": float(np.mean([r["pearson"] for r in per_seed])),
            "spearman": float(np.mean([r["spearman"] for r in per_seed])),
            "r2": float(np.mean([r["r2"] for r in per_seed])),
        }
    baseline_mae = float(np.mean(np.abs(y - y.mean())))
    for L in data["layers"]:
        out[f"layer_{L}"]["skill_vs_const"] = 1 - out[f"layer_{L}"]["mae"] / baseline_mae
    out["_baseline_mae"] = baseline_mae
    return out


def run(model, ckpt_names=None):
    ckpts = ({k: CKPTS[model][k] for k in ckpt_names} if ckpt_names else CKPTS[model])
    out_path = os.path.join(OUT_ROOT, f"{model}_nohex_probe.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## hex_contam_check / {model} / {ckpt_name} ##########")
        data = extract(model, ckpt_name, ckpt_path)
        results[ckpt_name] = probe_layers(data)
        print(json.dumps(results[ckpt_name], indent=2))

    os.makedirs(OUT_ROOT, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava")
    ap.add_argument("--ckpts", default=None, help="쉼표구분, 미지정시 전체")
    a = ap.parse_args()
    run(a.model, a.ckpts.split(",") if a.ckpts else None)
