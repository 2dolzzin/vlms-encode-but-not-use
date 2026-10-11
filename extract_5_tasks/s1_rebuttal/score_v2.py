# score_v2.py
# 상수예측에 취약한 1-MAE/follow-rate 대신, 상수예측을 걸러내는 보조 지표를 계산한다.
# 재추론 없음 — run_s1_variant.py 가 이미 만든 raw_responses(result_s1_rebuttal/<model>/<ckpt>/<variant>.json)
# 를 그대로 재사용.
#
# 실행: python s1_rebuttal/score_v2.py --model llava

import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
from scipy.stats import pearsonr, spearmanr

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_similarity  # noqa: E402

RESULT_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal")
CKPT_ORDER = ["Base", "S1_fwd", "S1_rev", "Final_fwd_S4", "Final_rev_S0", "S1_fwd_nohex", "S1_rev_nohex"]
PLAIN_VARIANTS = ["orig", "a_text_only", "c_nohex", "d_gray_nohex"]
CONST_STD_EPS = 1e-6


def score_one(records):
    pv, gv = [], []
    for r in records:
        p = parse_similarity(r["output"])
        if p is not None:
            pv.append(p)
            gv.append(float(r["sim_img"]))
    n = len(records)
    if len(pv) < 2:
        return {"n": n, "parsed": len(pv), "one_minus_mae": float("nan"),
                "std": float("nan"), "pearson": "const_collapse" if len(pv) else "no_data",
                "spearman": "const_collapse" if len(pv) else "no_data", "skill": float("nan")}

    pv, gv = np.array(pv), np.array(gv)
    mae = float(np.mean(np.abs(pv - gv)))
    std = float(np.std(pv))

    # 상수 baseline: 해당 78개 전체 sim_img의 median 고정 예측
    gt_all = np.array([float(r["sim_img"]) for r in records])
    const_pred = np.median(gt_all)
    mae_const = float(np.mean(np.abs(const_pred - gt_all)))
    skill = 1 - mae / mae_const if mae_const > 0 else float("nan")

    if std < CONST_STD_EPS:
        pearson, spearman = "const_collapse", "const_collapse"
    else:
        try:
            pearson = float(pearsonr(gv, pv)[0])
        except Exception:
            pearson = "const_collapse"
        try:
            spearman = float(spearmanr(gv, pv)[0])
        except Exception:
            spearman = "const_collapse"

    return {"n": n, "parsed": len(pv), "one_minus_mae": 1 - mae, "std": std,
            "pearson": pearson, "spearman": spearman, "skill": skill}


def output_distribution(records, top=5):
    vals = []
    for r in records:
        p = parse_similarity(r["output"])
        if p is not None:
            vals.append(p)
    n_unique = len(set(vals))
    top_vals = Counter(vals).most_common(top)
    return {"n": len(records), "parsed": len(vals), "n_unique": n_unique,
            "top": [{"value": v, "count": c, "ratio": c / len(vals)} for v, c in top_vals]}


def fmt(v, nd=3):
    if isinstance(v, str):
        return v
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    return f"{v:.{nd}f}"


def run(model):
    ckpt_dirs = [c for c in CKPT_ORDER if os.path.isdir(os.path.join(RESULT_ROOT, model, c))]
    all_scores = {}
    for ckpt in ckpt_dirs:
        all_scores[ckpt] = {}
        for variant in PLAIN_VARIANTS:
            path = os.path.join(RESULT_ROOT, model, ckpt, f"{variant}.json")
            if not os.path.exists(path):
                continue
            records = json.load(open(path))
            all_scores[ckpt][variant] = score_one(records)

    json.dump(all_scores, open(os.path.join(RESULT_ROOT, model, "_summary_v2.json"), "w"), indent=2)

    lines = []
    for variant in PLAIN_VARIANTS:
        lines.append(f"\n### {model} — {variant}\n")
        header = "| ckpt | 1-MAE | std | pearson | spearman | skill(vs const-median) |"
        sep = "|---|---|---|---|---|---|"
        lines += [header, sep]
        for ckpt in ckpt_dirs:
            s = all_scores[ckpt].get(variant)
            if s is None:
                continue
            row = f"| {ckpt} | {fmt(s['one_minus_mae'])} | {fmt(s['std'])} | {fmt(s['pearson'])} | {fmt(s['spearman'])} | {fmt(s['skill'])} |"
            lines.append(row)
    text = "\n".join(lines) + "\n"
    print(text)
    md_path = os.path.join(RESULT_ROOT, model, "_table_v2.md")
    open(md_path, "w").write(text)
    print(f"[saved] {md_path}")


def run_dist(model, ckpts, variant="orig"):
    print(f"\n### {model} — {variant} 출력 분포 (unique 값 개수 / top5)\n")
    out = {}
    for ckpt in ckpts:
        path = os.path.join(RESULT_ROOT, model, ckpt, f"{variant}.json")
        records = json.load(open(path))
        d = output_distribution(records)
        out[ckpt] = d
        print(f"[{ckpt}] n={d['n']} parsed={d['parsed']} unique={d['n_unique']}")
        for t in d["top"]:
            print(f"    value={t['value']:<8} count={t['count']:<3} ratio={t['ratio']:.3f}")
    json_path = os.path.join(RESULT_ROOT, model, f"_dist_{variant}.json")
    json.dump(out, open(json_path, "w"), indent=2)
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava")
    ap.add_argument("--dist", action="store_true", help="orig 출력 분포만 확인 (Base, S1_fwd)")
    a = ap.parse_args()
    if a.dist:
        run_dist(a.model, ["Base", "S1_fwd"])
    else:
        run(a.model)
