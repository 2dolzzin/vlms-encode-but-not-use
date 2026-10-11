# aggregate_expB_s4.py
# Experiment B S4 토큰-레벨(비풀링/그리드풀링) 프로브 결과 집계.
# mean-pooled 대조군(MLP/Linear)은 기존 Exp A 결과(probe_classifier.py, probe_logreg.py)를
# 그대로 인용 — 재추출 없음(probe_expB_s4.py 설계상 이미 명시된 방식).

import os
import glob
import json
import numpy as np

ROOT = "/home/hail/vlm3/probing"
MODELS = ["llava", "llava_reverse", "mipha", "mipha_reverse",
          "vila_3b", "vila_3b_reverse", "vila_8b", "vila_8b_reverse"]
SEEDS = [1, 2, 3]
VARIANTS = ["2x2_linear", "4x4_linear", "2x2_mlp", "4x4_mlp"]


def mean_pooled_mlp(model):
    vals = []
    for s in SEEDS:
        pat = os.path.join(ROOT, model, f"inf_classification_{s}", "*__S4_depth.json")
        for p in glob.glob(pat):
            d = json.load(open(p))
            for ckpt, tasks in d.items():
                if "S4_depth" in tasks and "proj" in tasks["S4_depth"]:
                    vals.append(tasks["S4_depth"]["proj"]["acc"])
    return vals


def mean_pooled_linear(model):
    vals = []
    for s in SEEDS:
        pat = os.path.join(ROOT, model, f"inf_classification_logreg_{s}", "*__S4_depth.json")
        for p in glob.glob(pat):
            d = json.load(open(p))
            for ckpt, tasks in d.items():
                if "S4_depth" in tasks and "proj" in tasks["S4_depth"]:
                    vals.append(tasks["S4_depth"]["proj"]["acc"])
    return vals


def expb_variant(model, variant):
    vals = []
    for run_idx in range(1, 4):
        pat = os.path.join(ROOT, "result_expB_s4", model, f"run_{run_idx}", f"*__{variant}.json")
        for p in glob.glob(pat):
            d = json.load(open(p))
            vals.append(d["acc"])
    return vals


def fmt(vals):
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "-", "-"
    return f"{np.mean(vals):.4f}", f"±{np.std(vals):.4f}"


def main():
    print("### Experiment B: S4_depth 토큰레벨(그리드풀링) vs mean-pooled 대조군")
    print("(각 셀: 11ckpt × 3seed 평균 accuracy ± std, S4_depth 4지선다 chance=0.25)\n")
    cols = ["mean-pool MLP", "mean-pool Linear"] + VARIANTS
    header = "| model | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    print(header)
    print(sep)
    lines = [header, sep]

    totals = {c: [] for c in cols}
    for model in MODELS:
        per_col = {
            "mean-pool MLP": mean_pooled_mlp(model),
            "mean-pool Linear": mean_pooled_linear(model),
        }
        for v in VARIANTS:
            per_col[v] = expb_variant(model, v)
        for c in cols:
            totals[c] += per_col[c]

        cells = []
        for c in cols:
            m, s = fmt(per_col[c])
            cells.append(f"{m} {s}" if m != "-" else "-")
        row = f"| {model} | " + " | ".join(cells) + " |"
        print(row)
        lines.append(row)

    cells = []
    for c in cols:
        m, s = fmt(totals[c])
        cells.append(f"{m} {s}" if m != "-" else "-")
    row = "| **전체평균** | " + " | ".join(cells) + " |"
    print(row)
    lines.append(row)

    out = os.path.join(ROOT, "_expB_s4_table.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
