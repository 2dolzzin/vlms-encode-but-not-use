# score_grayctrl.py
# Experiment A (회색 이미지 control) 채점. acc_parsing.py 의 파서를 그대로 재사용.
# stage별 원본 지표: S0=글자단위acc, S1=1-MAE, S2=acc, S3=acc, S4=acc.
# chance: S0=1/26, S1=상수(median) 예측 1-MAE, S2=0.1, S3=1/15, S4=0.25.

import os
import sys
import json
import glob
import numpy as np

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_letters, parse_similarity, parse_digit, parse_color, parse_mcq  # noqa: E402

RESULT_ROOT = os.path.join(EXTRACT5, "result_grayctrl")
CKPT_ORDER = ["Base", "Final_fwd", "Final_rev", "S1_fwd_nohex", "S1_rev_nohex"]
STAGES = ["S0", "S1", "S2", "S3", "S4"]

CHANCE = {"S0": 1 / 26, "S1": None, "S2": 0.1, "S3": 1 / 15, "S4": 0.25}  # S1은 아래서 데이터 기반 계산


def s1_chance():
    d = json.load(open("/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/dataset.json"))
    gt = np.array([float(c["conversations"][1]["value"]) for c in d])
    med = np.median(gt)
    return 1 - np.mean(np.abs(gt - med))


CHANCE["S1"] = s1_chance()


def score_stage(stage, records):
    if stage == "S0":
        cl = tl = 0
        for r in records:
            pred = parse_letters(r["output"])
            gt = r["expected_output"].strip().upper()
            for i in range(len(gt)):
                p = pred[i] if i < len(pred) else "<none>"
                tl += 1
                cl += (p == gt[i])
        return cl / tl if tl else float("nan")
    if stage == "S1":
        pv, gv = [], []
        for r in records:
            p = parse_similarity(r["output"])
            if p is not None:
                pv.append(p); gv.append(float(r["expected_output"]))
        if not pv:
            return float("nan")
        pv, gv = np.array(pv), np.array(gv)
        return 1 - np.mean(np.abs(pv - gv))
    if stage == "S2":
        correct = total = 0
        for r in records:
            p = parse_digit(r["output"])
            g = str(r["expected_output"]).strip()
            correct += (p == g); total += 1
        return correct / total if total else float("nan")
    if stage == "S3":
        correct = total = 0
        for r in records:
            p = parse_color(r["output"])
            g = str(r["expected_output"]).strip().lower()
            correct += (p == g); total += 1
        return correct / total if total else float("nan")
    if stage == "S4":
        correct = total = 0
        for r in records:
            p = parse_mcq(r["output"])
            g = str(r["expected_output"]).strip().upper()
            correct += (p == g); total += 1
        return correct / total if total else float("nan")
    raise ValueError(stage)


def run(model):
    ckpt_dirs = sorted(glob.glob(os.path.join(RESULT_ROOT, model, "*")))
    ckpt_names = [os.path.basename(d) for d in ckpt_dirs]
    ckpt_names = [c for c in CKPT_ORDER if c in ckpt_names] + [c for c in ckpt_names if c not in CKPT_ORDER]

    print(f"\n### {model} — 회색 이미지 control (원본 지표, chance 표기)\n")
    header = "| ckpt | " + " | ".join(f"{s}(orig/gray/chance)" for s in STAGES) + " |"
    sep = "|---" * (len(STAGES) + 1) + "|"
    lines = [header, sep]
    print(header); print(sep)

    for ckpt in ckpt_names:
        row = [ckpt]
        for stage in STAGES:
            orig_p = os.path.join(RESULT_ROOT, model, ckpt, f"{stage}_orig.json")
            gray_p = os.path.join(RESULT_ROOT, model, ckpt, f"{stage}_gray.json")
            if not (os.path.exists(orig_p) and os.path.exists(gray_p)):
                row.append("-")
                continue
            orig_score = score_stage(stage, json.load(open(orig_p)))
            gray_score = score_stage(stage, json.load(open(gray_p)))
            row.append(f"{orig_score:.3f}/{gray_score:.3f}/{CHANCE[stage]:.3f}")
        line = "| " + " | ".join(row) + " |"
        print(line)
        lines.append(line)

    out = os.path.join(RESULT_ROOT, f"_table_{model}.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    run(a.model)
