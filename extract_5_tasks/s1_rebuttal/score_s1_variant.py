# score_s1_variant.py
# run_s1_variant.py 가 만든 raw_responses(result_s1_rebuttal/<model>/<ckpt>/<variant>.json) 를
# acc_parsing.parse_similarity() 로 채점한다 (원본 파싱 로직 그대로 재사용).
#
# 실행 (extract_5_tasks/ 안에서):
#   python s1_rebuttal/score_s1_variant.py --model llava

import os
import sys
import csv
import json
import argparse

import numpy as np

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_similarity  # noqa: E402

RESULT_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal")

CKPT_ORDER = ["Base", "S1_fwd", "S1_rev", "Final_fwd_S4", "Final_rev_S0"]
VARIANT_ORDER = ["orig", "a_text_only", "c_nohex", "d_gray_nohex", "b_mismatch"]


def score_plain(records):
    """orig / a_text_only / c_nohex / d_gray_nohex 공통: sim_img 기준 MAE."""
    pv, gv = [], []
    for r in records:
        p = parse_similarity(r["output"])
        if p is not None:
            pv.append(p)
            gv.append(float(r["sim_img"]))
    n = len(records)
    if not pv:
        return {"n": n, "parsed": 0, "parse_ratio": 0.0, "mae": float("nan"), "one_minus_mae": float("nan")}
    pv, gv = np.array(pv), np.array(gv)
    mae = float(np.mean(np.abs(pv - gv)))
    return {"n": n, "parsed": len(pv), "parse_ratio": len(pv) / n, "mae": mae, "one_minus_mae": 1 - mae}


def score_mismatch(records):
    n = len(records)
    err_img, err_text = [], []
    n_image, n_text, n_tie = 0, 0, 0
    details = []
    for r in records:
        p = parse_similarity(r["output"])
        sim_img, sim_text = float(r["sim_img"]), float(r["sim_text"])
        if p is None:
            details.append({"id": r["id"], "sim_img": sim_img, "sim_text": sim_text,
                            "gap": r["gap"], "pred": None, "judgement": "parse_fail"})
            continue
        ei, et = abs(p - sim_img), abs(p - sim_text)
        err_img.append(ei); err_text.append(et)
        if ei < et:
            judge = "image"; n_image += 1
        elif et < ei:
            judge = "text"; n_text += 1
        else:
            judge = "tie"; n_tie += 1
        details.append({"id": r["id"], "sim_img": sim_img, "sim_text": sim_text,
                        "gap": r["gap"], "pred": p, "err_img": ei, "err_text": et,
                        "judgement": judge})
    parsed = len(err_img)
    decided = n_image + n_text
    summary = {
        "n": n, "parsed": parsed, "parse_ratio": parsed / n if n else 0.0,
        "mae_img": float(np.mean(err_img)) if err_img else float("nan"),
        "mae_text": float(np.mean(err_text)) if err_text else float("nan"),
        "n_image": n_image, "n_text": n_text, "n_tie": n_tie,
        "image_following_rate": (n_image / decided) if decided else float("nan"),
    }
    return summary, details


def run(model):
    ckpt_dirs = sorted(d for d in os.listdir(os.path.join(RESULT_ROOT, model))
                       if os.path.isdir(os.path.join(RESULT_ROOT, model, d)))
    all_summaries = {}
    for ckpt in ckpt_dirs:
        ckpt_dir = os.path.join(RESULT_ROOT, model, ckpt)
        all_summaries[ckpt] = {}
        for variant in VARIANT_ORDER:
            path = os.path.join(ckpt_dir, f"{variant}.json")
            if not os.path.exists(path):
                continue
            records = json.load(open(path))
            if variant == "b_mismatch":
                summary, details = score_mismatch(records)
                all_summaries[ckpt][variant] = summary
                csv_path = os.path.join(ckpt_dir, "b_mismatch_detail.csv")
                with open(csv_path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["id", "sim_img", "sim_text", "gap",
                                                       "pred", "err_img", "err_text", "judgement"])
                    w.writeheader()
                    for d in details:
                        w.writerow(d)
                print(f"[saved] {csv_path}")
            else:
                all_summaries[ckpt][variant] = score_plain(records)

    summary_path = os.path.join(RESULT_ROOT, model, "_summary.json")
    json.dump(all_summaries, open(summary_path, "w"), indent=2)
    print(f"[saved] {summary_path}")

    print_table(model, all_summaries)


def fmt_plain(s):
    if s is None:
        return "-"
    if s["parsed"] == 0:
        return "NaN"
    return f"{s['one_minus_mae']:.3f}"


def fmt_mismatch(s):
    if s is None:
        return "-"
    if s["parsed"] == 0:
        return "NaN"
    return (f"MAE_img={s['mae_img']:.3f} MAE_text={s['mae_text']:.3f} "
            f"follow={s['image_following_rate']:.2f}")


def print_table(model, all_summaries):
    ckpts = [c for c in CKPT_ORDER if c in all_summaries] + \
            [c for c in all_summaries if c not in CKPT_ORDER]
    print(f"\n### S1 hex-shortcut rebuttal — {model}\n")
    header = "| ckpt | " + " | ".join(VARIANT_ORDER) + " |"
    sep = "|---" * (len(VARIANT_ORDER) + 1) + "|"
    print(header)
    print(sep)
    lines = [header, sep]
    for ckpt in ckpts:
        row = [ckpt]
        for v in VARIANT_ORDER:
            s = all_summaries[ckpt].get(v)
            row.append(fmt_mismatch(s) if v == "b_mismatch" else fmt_plain(s))
        line = "| " + " | ".join(row) + " |"
        print(line)
        lines.append(line)
    md_path = os.path.join(RESULT_ROOT, model, "_table.md")
    open(md_path, "w").write("\n".join(lines) + "\n")
    print(f"\n[saved] {md_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava")
    a = ap.parse_args()
    run(a.model)
