# score_v3_ols.py
# (b) mismatch_v2 채점: output ~ b0 + b_img*sim_img + b_text*sim_text (OLS).
# 이미지를 쓰면 b_img 유의/크고 b_text~0, 텍스트를 쓰면 반대, 상수 붕괴면 둘 다 ~0, R²~0.
# balanced follow rate(참고용)도 같이 보고.
#
# 실행: python s1_rebuttal/score_v3_ols.py --model llava

import os
import sys
import json
import argparse

import numpy as np
import statsmodels.api as sm

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_similarity  # noqa: E402

RESULT_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal")
CKPT_ORDER = ["Base", "S1_fwd", "S1_rev", "Final_fwd_S4", "Final_rev_S0"]
VARIANT = "b_mismatch_v2"


def score_one(records):
    pred, sim_img, sim_text = [], [], []
    n_parse_fail = 0
    for r in records:
        p = parse_similarity(r["output"])
        if p is None:
            n_parse_fail += 1
            continue
        pred.append(p)
        sim_img.append(float(r["sim_img"]))
        sim_text.append(float(r["sim_text"]))

    n = len(records)
    if len(pred) < 5:
        return {"n": n, "parsed": len(pred), "note": "insufficient parsed samples"}

    pred = np.array(pred)
    sim_img = np.array(sim_img)
    sim_text = np.array(sim_text)

    X = sm.add_constant(np.column_stack([sim_img, sim_text]))
    model = sm.OLS(pred, X).fit()
    b0, b_img, b_text = model.params
    p0, p_img, p_text = model.pvalues

    n_image = int(np.sum(np.abs(pred - sim_img) < np.abs(pred - sim_text)))
    n_text = int(np.sum(np.abs(pred - sim_text) < np.abs(pred - sim_img)))
    n_tie = len(pred) - n_image - n_text
    follow_rate = n_image / (n_image + n_text) if (n_image + n_text) else float("nan")

    return {
        "n": n, "parsed": len(pred), "n_parse_fail": n_parse_fail,
        "b_img": float(b_img), "p_img": float(p_img),
        "b_text": float(b_text), "p_text": float(p_text),
        "b0": float(b0), "p0": float(p0),
        "r2": float(model.rsquared),
        "follow_rate": follow_rate, "n_image": n_image, "n_text": n_text, "n_tie": n_tie,
    }


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def sig(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def run(model_name):
    ckpt_dirs = [c for c in CKPT_ORDER if os.path.isdir(os.path.join(RESULT_ROOT, model_name, c))]
    results = {}
    for ckpt in ckpt_dirs:
        path = os.path.join(RESULT_ROOT, model_name, ckpt, f"{VARIANT}.json")
        if not os.path.exists(path):
            print(f"[skip] {ckpt}: {path} 없음")
            continue
        records = json.load(open(path))
        results[ckpt] = score_one(records)

    json.dump(results, open(os.path.join(RESULT_ROOT, model_name, "_summary_v3_ols.json"), "w"), indent=2)

    header = "| ckpt | n(parsed) | beta_img | p_img | beta_text | p_text | R2 | follow_rate(ref) |"
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [f"\n### {model_name} — (b) mismatch_v2 OLS: output ~ b0 + b_img*sim_img + b_text*sim_text\n",
             header, sep]
    for ckpt in ckpt_dirs:
        s = results.get(ckpt)
        if s is None or "note" in s:
            lines.append(f"| {ckpt} | - | - | - | - | - | - | - |")
            continue
        row = (f"| {ckpt} | {s['parsed']}/{s['n']} "
               f"| {fmt(s['b_img'])}{sig(s['p_img'])} | {fmt(s['p_img'], 4)} "
               f"| {fmt(s['b_text'])}{sig(s['p_text'])} | {fmt(s['p_text'], 4)} "
               f"| {fmt(s['r2'])} | {fmt(s['follow_rate'])} |")
        lines.append(row)
    text = "\n".join(lines) + "\n"
    print(text)
    md_path = os.path.join(RESULT_ROOT, model_name, "_table_v3_ols.md")
    open(md_path, "w").write(text)
    print(f"[saved] {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava")
    a = ap.parse_args()
    run(a.model)
