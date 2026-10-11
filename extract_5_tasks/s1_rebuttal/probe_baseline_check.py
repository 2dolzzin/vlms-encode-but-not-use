# probe_baseline_check.py
# 2순위: 기존 S1 projector/LM-depth probe(ridge, 5-fold OOF, probing/probe_regressor.py 결과물)를
# 상수 baseline과 비교. 재추론/재학습 없음 — probing/<model>/inf_regression_{1,2,3}/ridge/*.json
# 을 그대로 읽어서 seed 3개 평균 + skill 계산만 한다.
#
# 실행: python s1_rebuttal/probe_baseline_check.py --model llava

import os
import json
import argparse
import numpy as np

PROBING_ROOT = "/home/hail/vlm3/probing"
SEEDS = [1, 2, 3]
PROBE_KEYS = ["proj", "layer_0", "layer_8", "layer_16", "layer_24", "layer_32"]


def load(model, seed, ckpt):
    path = os.path.join(PROBING_ROOT, model, f"inf_regression_{seed}", "ridge",
                        f"{ckpt}__S1_similarity.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d[ckpt]["S1_similarity"]


def summarize(model, ckpt):
    runs = [load(model, s, ckpt) for s in SEEDS]
    runs = [r for r in runs if r is not None]
    if not runs:
        return None
    baseline_maes = [r["_baseline_mae"] for r in runs]
    out = {"n_seeds": len(runs), "baseline_mae_mean": float(np.mean(baseline_maes))}
    for key in PROBE_KEYS:
        maes = [r[key]["mae"] for r in runs if key in r]
        pears = [r[key]["pearson"] for r in runs if key in r]
        spears = [r[key]["spearman"] for r in runs if key in r]
        r2s = [r[key]["r2"] for r in runs if key in r]
        if not maes:
            continue
        mae_mean = float(np.mean(maes))
        skill = 1 - mae_mean / out["baseline_mae_mean"]
        out[key] = {
            "mae": mae_mean, "pearson": float(np.mean(pears)), "spearman": float(np.mean(spears)),
            "r2": float(np.mean(r2s)), "skill_vs_const": skill,
        }
    return out


def fmt(v, nd=3):
    return f"{v:.{nd}f}"


def print_table(model, ckpts):
    print(f"\n### {model} — S1 probe (ridge, 5-fold OOF, {len(SEEDS)}seed 평균) vs 상수(평균예측) baseline\n")
    header = "| ckpt | probe | 1-MAE | pearson | spearman | r2 | skill(vs const-mean) |"
    sep = "|---|---|---|---|---|---|---|"
    print(header); print(sep)
    lines = [header, sep]
    for ckpt in ckpts:
        s = summarize(model, ckpt)
        if s is None:
            print(f"| {ckpt} | - | (없음) | | | | |")
            continue
        for key in PROBE_KEYS:
            if key not in s:
                continue
            v = s[key]
            row = (f"| {ckpt} | {key} | {fmt(1 - v['mae'])} | {fmt(v['pearson'])} "
                   f"| {fmt(v['spearman'])} | {fmt(v['r2'])} | {fmt(v['skill_vs_const'])} |")
            print(row); lines.append(row)
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava")
    ap.add_argument("--ckpts", default="S0_start,S1_end")
    a = ap.parse_args()
    lines_all = []
    for m in a.model.split(","):
        lines_all += print_table(m, a.ckpts.split(","))
    out_path = f"/home/hail/vlm3/extract_5_tasks/result_s1_rebuttal/_probe_baseline_check.md"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    open(out_path, "w").write("\n".join(lines_all) + "\n")
    print(f"\n[saved] {out_path}")
