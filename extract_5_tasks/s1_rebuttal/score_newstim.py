# score_newstim.py
# S1/S3 신규 자극 채점, 재추론 없음 — run_s1_newstim.py/run_s3_newstim.py 결과만 사용.
#
# S1: 체크포인트별 {c_nohex(원본), swap, perturb, newpair} 채점 (1-MAE/std/pearson/spearman/
#     skill/exact-match@2decimal). swap은 원본 대비 per-sample |Δ|평균·상관도 추가.
#     newpair는 GT가 CIEDE2000 proxy라 Spearman만 신뢰(스펙 지시).
# S3: 체크포인트별 {orig, perturb} accuracy, chance=1/15 병기.

import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import pearsonr, spearmanr

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_similarity, parse_color  # noqa: E402

S1_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal")
S3_ROOT = os.path.join(EXTRACT5, "result_s3_rebuttal")

S1_MODELS = ["llava", "mipha", "vila_3b", "vila_8b", "openflamingo"]
S1_CKPTS = {
    "llava": ["Base", "S1_fwd", "S1_rev", "S1_fwd_nohex", "S1_rev_nohex"],
    "mipha": ["Base", "S1_fwd", "S1_rev"],
    "vila_3b": ["Base", "S1_fwd", "S1_rev", "S1_fwd_nohex", "S1_rev_nohex"],
    "vila_8b": ["Base", "S1_fwd", "S1_rev"],
    "openflamingo": ["Base", "S1_fwd", "S1_rev"],
}
S1_VARIANTS = ["c_nohex", "swap", "perturb", "newpair"]

S3_MODELS = ["llava", "mipha", "vila_3b", "vila_8b", "openflamingo"]
S3_CKPTS = ["Base", "Final_fwd", "Final_rev"]
S3_CHANCE = 1 / 15

CONST_STD_EPS = 1e-6


def score_s1(records, exact_decimals=2):
    pv, gv, ids = [], [], []
    for r in records:
        p = parse_similarity(r["output"])
        if p is not None:
            pv.append(p); gv.append(float(r["sim_img"])); ids.append(r["id"])
    n = len(records)
    if len(pv) < 2:
        return {"n": n, "parsed": len(pv), "one_minus_mae": float("nan"), "std": float("nan"),
                "pearson": "const_collapse" if len(pv) else "no_data",
                "spearman": "const_collapse" if len(pv) else "no_data",
                "skill": float("nan"), "exact_match": float("nan")}

    pv_a, gv_a = np.array(pv), np.array(gv)
    mae = float(np.mean(np.abs(pv_a - gv_a)))
    std = float(np.std(pv_a))

    gt_all = np.array([float(r["sim_img"]) for r in records])
    const_pred = np.median(gt_all)
    mae_const = float(np.mean(np.abs(const_pred - gt_all)))
    skill = 1 - mae / mae_const if mae_const > 0 else float("nan")

    if std < CONST_STD_EPS:
        pearson, spearman = "const_collapse", "const_collapse"
    else:
        try:
            pearson = float(pearsonr(gv_a, pv_a)[0])
        except Exception:
            pearson = "const_collapse"
        try:
            spearman = float(spearmanr(gv_a, pv_a)[0])
        except Exception:
            spearman = "const_collapse"

    exact = float(np.mean(np.round(pv_a, exact_decimals) == np.round(gv_a, exact_decimals)))

    return {"n": n, "parsed": len(pv), "one_minus_mae": 1 - mae, "std": std,
            "pearson": pearson, "spearman": spearman, "skill": skill,
            "exact_match": exact, "_pred_by_id": dict(zip(ids, pv))}


def score_s1_spearman_only(records):
    pv, gv = [], []
    for r in records:
        p = parse_similarity(r["output"])
        if p is not None:
            pv.append(p); gv.append(float(r["sim_img"]))
    if len(pv) < 2 or np.std(pv) < CONST_STD_EPS:
        return {"n": len(records), "parsed": len(pv), "spearman": "const_collapse" if pv else "no_data"}
    try:
        sp = float(spearmanr(gv, pv)[0])
    except Exception:
        sp = "const_collapse"
    return {"n": len(records), "parsed": len(pv), "spearman": sp}


def load_json(p):
    return json.load(open(p)) if os.path.exists(p) else None


def swap_delta(orig_records, swap_records):
    """원본(c_nohex) vs swap 예측의 per-sample |Δ|, 이미지 basename 으로 매칭."""
    orig_by_base = {}
    for r in orig_records:
        base = os.path.basename(r.get("image", "")) or None
        p = parse_similarity(r["output"])
        if p is not None:
            orig_by_base[str(int(r["id"]) - 1)] = p  # c_nohex id "N" -> labels index N-1

    deltas, pairs = [], []
    for r in swap_records:
        p = parse_similarity(r["output"])
        if p is None:
            continue
        idx = r["id"].replace("swap_", "")
        if idx in orig_by_base:
            o = orig_by_base[idx]
            deltas.append(abs(p - o))
            pairs.append((o, p))
    if not deltas:
        return {"mean_abs_delta": float("nan"), "corr": "no_data", "n_matched": 0}
    deltas = np.array(deltas)
    o_arr = np.array([p[0] for p in pairs]); s_arr = np.array([p[1] for p in pairs])
    if np.std(o_arr) < CONST_STD_EPS or np.std(s_arr) < CONST_STD_EPS:
        corr = "const_collapse"
    else:
        corr = float(pearsonr(o_arr, s_arr)[0])
    return {"mean_abs_delta": float(np.mean(deltas)), "corr": corr, "n_matched": len(deltas)}


def fmt(v, nd=3):
    if isinstance(v, str):
        return v
    if isinstance(v, float) and np.isnan(v):
        return "NaN"
    return f"{v:.{nd}f}"


def run_s1():
    lines = ["## 표 B-1 — S1 신규 자극 (5모델, 전부 hex 제거 프롬프트)\n"]
    header = ("| model | ckpt | 원본 c (1-MAE/pearson/spearman/exact) | swap (1-MAE/pearson/spearman/exact) "
              "| swap |Δ| (mean/corr) | perturb (1-MAE/pearson/spearman/exact) | newpair (Spearman) |")
    sep = "|" + "---|" * 7
    lines += [header, sep]

    all_scores = {}
    for model in S1_MODELS:
        all_scores[model] = {}
        for ckpt in S1_CKPTS[model]:
            base_dir = os.path.join(S1_ROOT, model, ckpt)
            c_rec = load_json(os.path.join(base_dir, "c_nohex.json"))
            swap_rec = load_json(os.path.join(base_dir, "swap.json"))
            pert_rec = load_json(os.path.join(base_dir, "perturb.json"))
            newp_rec = load_json(os.path.join(base_dir, "newpair.json"))
            if c_rec is None:
                continue

            s_c = score_s1(c_rec)
            s_swap = score_s1(swap_rec) if swap_rec else None
            s_pert = score_s1(pert_rec) if pert_rec else None
            s_newp = score_s1_spearman_only(newp_rec) if newp_rec else None
            d_swap = swap_delta(c_rec, swap_rec) if swap_rec else None

            all_scores[model][ckpt] = {"c_nohex": s_c, "swap": s_swap, "perturb": s_pert,
                                       "newpair": s_newp, "swap_delta": d_swap}

            def cell(s):
                if s is None:
                    return "-"
                return f"{fmt(s['one_minus_mae'])}/{fmt(s['pearson'])}/{fmt(s['spearman'])}/{fmt(s['exact_match'])}"

            swap_delta_cell = (f"{d_swap['mean_abs_delta']:.3f}/{fmt(d_swap['corr'])}"
                              if d_swap and d_swap['n_matched'] > 0 else "-")
            newp_cell = fmt(s_newp['spearman']) if s_newp else "-"

            row = f"| {model} | {ckpt} | {cell(s_c)} | {cell(s_swap)} | {swap_delta_cell} | {cell(s_pert)} | {newp_cell} |"
            lines.append(row)

    text = "\n".join(lines) + "\n"
    print(text)
    out_dir = os.path.join(EXTRACT5, "s1_rebuttal", "tables")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "tableB1_s1_newstim.md"), "w").write(text)
    # _pred_by_id 는 json 저장에서 제외(용량/직렬화)
    for m in all_scores:
        for ck in all_scores[m]:
            for v in ["c_nohex", "swap", "perturb"]:
                if all_scores[m][ck].get(v) and "_pred_by_id" in all_scores[m][ck][v]:
                    del all_scores[m][ck][v]["_pred_by_id"]
    json.dump(all_scores, open(os.path.join(out_dir, "tableB1_s1_newstim.json"), "w"), indent=2)
    print(f"[saved] tables/tableB1_s1_newstim.md, .json")


def score_s3_acc(records):
    correct = total = 0
    for r in records:
        p = parse_color(r["output"])
        g = str(r["answer"]).strip().lower()
        correct += (p == g); total += 1
    return correct / total if total else float("nan")


def run_s3():
    lines = [f"## 표 B-2 — S3 칩 perturbation (5모델, chance={S3_CHANCE:.3f})\n"]
    header = "| model | ckpt | 원본 acc | perturb acc |"
    sep = "|---|---|---|---|"
    lines += [header, sep]

    all_scores = {}
    for model in S3_MODELS:
        all_scores[model] = {}
        for ckpt in S3_CKPTS:
            base_dir = os.path.join(S3_ROOT, model, ckpt)
            orig_rec = load_json(os.path.join(base_dir, "orig.json"))
            pert_rec = load_json(os.path.join(base_dir, "perturb.json"))
            if orig_rec is None:
                continue
            acc_o = score_s3_acc(orig_rec)
            acc_p = score_s3_acc(pert_rec) if pert_rec else float("nan")
            all_scores[model][ckpt] = {"orig": acc_o, "perturb": acc_p}
            lines.append(f"| {model} | {ckpt} | {fmt(acc_o)} | {fmt(acc_p)} |")

    text = "\n".join(lines) + "\n"
    print(text)
    out_dir = os.path.join(EXTRACT5, "s1_rebuttal", "tables")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "tableB2_s3_newstim.md"), "w").write(text)
    json.dump(all_scores, open(os.path.join(out_dir, "tableB2_s3_newstim.json"), "w"), indent=2)
    print(f"[saved] tables/tableB2_s3_newstim.md, .json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["s1", "s3"], default=None)
    a = ap.parse_args()
    if a.only in (None, "s1"):
        run_s1()
    if a.only in (None, "s3"):
        run_s3()
