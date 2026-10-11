# score_expB_s4.py
# Experiment B 채점: S4 토큰-레벨(2x2/4x4 그리드) 프로브.
# 2x2/4x4 선형(L2 logistic, C는 CV로 강한 정규화까지 포함해 탐색) + 2x2 MLP(기존 구성 그대로).
# probe_classifier.py(MLP)/probe_logreg.py 스타일의 5-fold OOF, 3 seed, StandardScaler 재사용.

import os
import sys
import json
import time
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

PROBING_DIR = "/home/hail/vlm3/probing"
sys.path.insert(0, PROBING_DIR)
import probe_classifier as pc  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(EXTRACT5, "result_expB_s4_tokenlevel")
OUT_ROOT = os.path.join(EXTRACT5, "result_expB_s4_scored")

MODELS = ["llava", "llava_reverse", "mipha", "mipha_reverse",
         "vila_3b", "vila_3b_reverse", "vila_8b", "vila_8b_reverse"]
CKPTS = ["S0_start", "S0_mid", "S0_end", "S1_mid", "S1_end", "S2_mid", "S2_end",
        "S3_mid", "S3_end", "S4_mid", "S4_end"]
SEEDS = pc.SEEDS  # [53, 9, 144]
N_FOLDS = pc.N_FOLDS  # 5

Cs = np.logspace(-5, 1, 10)
INNER_CV = 3
MAX_ITER = 300


def fit_predict_logreg(X, y, seed):
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    for tr_idx, te_idx in pc._kfold_indices(X, y, N_FOLDS, seed):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = LogisticRegressionCV(Cs=Cs, cv=INNER_CV, scoring="accuracy",
                                   solver="lbfgs", max_iter=MAX_ITER, n_jobs=1, random_state=seed)
        clf.fit(X_tr, y_tr)
        oof_pred[te_idx] = clf.predict(X_te)
    return oof_pred


def score_variant_worker(args):
    model, ckpt, variant, seed, run_idx = args
    out_dir = os.path.join(OUT_ROOT, model, f"run_{run_idx}")
    out_path = os.path.join(out_dir, f"{ckpt}__{variant}.json")
    if os.path.exists(out_path):
        return f"[cached] {model}/{ckpt}/{variant}/seed{seed}"

    pt_path = os.path.join(DATA_ROOT, model, f"{ckpt}.pt")
    if not os.path.exists(pt_path):
        return f"[missing] {pt_path}"

    try:
        data = torch.load(pt_path, map_location="cpu")
        labels, num_classes = pc.make_labels(data["answers"])

        if variant == "2x2_linear":
            X = data["feats_2x2"].float().reshape(len(labels), -1).numpy()
            oof = fit_predict_logreg(X, labels, seed)
        elif variant == "4x4_linear":
            X = data["feats_4x4"].float().reshape(len(labels), -1).numpy()
            oof = fit_predict_logreg(X, labels, seed)
        elif variant == "2x2_mlp":
            X = data["feats_2x2"].float().reshape(len(labels), -1).numpy()
            pc.SEED = seed
            acc, f1 = pc.train_one_layer(X, labels, num_classes, X.shape[1], f"{model}/{ckpt}/2x2_mlp")
            os.makedirs(out_dir, exist_ok=True)
            json.dump({"acc": acc, "macro_f1": f1, "n": len(labels)}, open(out_path, "w"))
            return f"[saved] {model}/{ckpt}/{variant}/seed{seed} acc={acc:.4f}"
        else:
            raise ValueError(variant)

        acc = accuracy_score(labels, oof)
        f1 = f1_score(labels, oof, average="macro")
        os.makedirs(out_dir, exist_ok=True)
        json.dump({"acc": acc, "macro_f1": f1, "n": len(labels)}, open(out_path, "w"))
        return f"[saved] {model}/{ckpt}/{variant}/seed{seed} acc={acc:.4f}"
    except Exception as e:
        import traceback
        err_dir = os.path.join(out_dir, "_errors")
        os.makedirs(err_dir, exist_ok=True)
        open(os.path.join(err_dir, f"{ckpt}__{variant}_s{seed}.txt"), "w").write(traceback.format_exc())
        return f"[ERROR] {model}/{ckpt}/{variant}/seed{seed}: {type(e).__name__}: {e}"


def run(models, variants, n_jobs):
    jobs = []
    for model in models:
        for ckpt in CKPTS:
            pt_path = os.path.join(DATA_ROOT, model, f"{ckpt}.pt")
            if not os.path.exists(pt_path):
                continue
            for variant in variants:
                for run_idx, seed in enumerate(SEEDS, start=1):
                    jobs.append((model, ckpt, variant, seed, run_idx))

    print(f"처리할 작업: {len(jobs)}개, n_jobs={n_jobs}")
    t0 = time.time()
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for msg in pool.imap_unordered(score_variant_worker, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(score_variant_worker(j), flush=True)
    print(f"\n[done] {len(jobs)} jobs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--variants", default="2x2_linear,4x4_linear,2x2_mlp")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    run(a.models.split(","), a.variants.split(","), a.jobs)
