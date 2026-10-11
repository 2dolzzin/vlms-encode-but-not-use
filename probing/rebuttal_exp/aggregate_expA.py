# aggregate_expA.py
# Experiment A 중간 결과 집계: 완료된 모델만 모아서 과제별 (MLP vs 선형 vs 추론) 비교표.
# projector_vs_lm.csv 와 같은 방식(모델×방향×체크포인트×seed 전체 평균)으로 aggregate.

import os
import glob
import json
import numpy as np

PROBING_ROOT = "/home/hail/vlm3/probing"
BENCH_ROOT = "/home/hail/vlm3/extract_benchmark/result"
TASKS = ["circled_letter", "counting_grid_blank", "counting_grid_word", "nested_squares",
         "olympic_circles", "olympic_pentagons", "subway_connections", "touching_circles"]
SEEDS = [1, 2, 3]


def completed_models():
    done = []
    for d in sorted(glob.glob(os.path.join(PROBING_ROOT, "*_blind"))):
        model = os.path.basename(d)
        logreg_dirs = [os.path.join(d, f"ben_classification_logreg_{s}") for s in SEEDS]
        if all(os.path.isdir(p) for p in logreg_dirs):
            # 완료 판정: 모든 task에 대해 최소 1개 ckpt 파일이라도 존재하는지 대충 체크 (완전판정은 개수로)
            n_files = sum(len(glob.glob(os.path.join(p, "*.json"))) for p in logreg_dirs)
            done.append((model, n_files))
    return done


def load_acc(path_glob, key_getter):
    """glob으로 찾은 파일들에서 acc 값을 모아 평균."""
    vals = []
    for p in glob.glob(path_glob):
        d = json.load(open(p))
        for ckpt, tasks in d.items():
            for task, res in tasks.items():
                v = key_getter(res)
                if v is not None:
                    vals.append(v)
    return vals


def mlp_vals(model, task):
    vals = []
    for s in SEEDS:
        pat = os.path.join(PROBING_ROOT, model, f"ben_classification_{s}", f"*__{task}.json")
        for p in glob.glob(pat):
            d = json.load(open(p))
            for ckpt, tasks in d.items():
                if task in tasks and "proj" in tasks[task]:
                    vals.append(tasks[task]["proj"]["acc"])
    return vals


def logreg_vals(model, task):
    vals = []
    for s in SEEDS:
        pat = os.path.join(PROBING_ROOT, model, f"ben_classification_logreg_{s}", f"*__{task}.json")
        for p in glob.glob(pat):
            d = json.load(open(p))
            for ckpt, tasks in d.items():
                if task in tasks and "proj" in tasks[task]:
                    vals.append(tasks[task]["proj"]["acc"])
    return vals


def inf_vals(model, task):
    vals = []
    p = os.path.join(BENCH_ROOT, model, "accuracy", "_summary_all.json")
    if not os.path.exists(p):
        return vals
    rows = json.load(open(p))
    for r in rows:
        if r.get("task") == task and r.get("accuracy") is not None:
            vals.append(r["accuracy"])
    return vals


def fmt(vals, nd=4):
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "-"
    return f"{np.mean(vals):.{nd}f}"


def main():
    models = completed_models()
    print("완료된 모델(로그 폴더 존재 기준):")
    for m, n in models:
        print(f"  {m}  (files~{n})")
    model_names = [m for m, _ in models]

    print(f"\n### Experiment A 중간 결과 — 완료된 {len(model_names)}/10 모델 기준 (task별 전체 ckpt·seed 평균)\n")
    header = "| task | MLP(proj) | Linear(proj) | Δ(Linear-MLP) | 추론(inf) |"
    sep = "|---|---|---|---|---|"
    print(header)
    print(sep)
    lines = [header, sep]
    for task in TASKS:
        mlp_all, lr_all, inf_all = [], [], []
        for m in model_names:
            mlp_all += mlp_vals(m, task)
            lr_all += logreg_vals(m, task)
            inf_all += inf_vals(m, task)
        mlp_m = np.mean(mlp_all) if mlp_all else None
        lr_m = np.mean(lr_all) if lr_all else None
        delta = f"{lr_m - mlp_m:+.4f}" if (mlp_m is not None and lr_m is not None) else "-"
        row = f"| {task} | {fmt(mlp_all)} | {fmt(lr_all)} | {delta} | {fmt(inf_all)} |"
        print(row)
        lines.append(row)

    out = os.path.join(PROBING_ROOT, "_expA_interim_table.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
