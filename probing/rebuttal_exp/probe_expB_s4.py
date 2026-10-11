# probe_expB_s4.py
# Experiment B: S4 토큰-레벨(비풀링) 프로브. extract_s4_tokenlevel.py 가 만든
# [N,4,d]/[N,16,d] AdaptiveAvgPool2d 특징에 대해:
#   - 2x2 선형(logreg), 4x4 선형(logreg), 2x2 MLP(기존 probe_classifier.py 구성 그대로)
# 를 5-fold outer OOF x 3 seed 로 채점. mean-pooled 대조군은 이미 있는 결과(Exp A 배터리,
# 기존 MLP probe)를 그대로 인용 — 재추출 없음.
#
# 계산량: probe_logreg.py 와 동일 이유로 Cs=8점(logspace(-5,1,8), 사용자 지정 범위),
# inner_cv=3, max_iter=150 사용 (S0_logamr에서 확인된 것과 같은 컴퓨팅 예산 축소, 새
# 튜닝이 아니라 계산 가능하게 만드는 것).
#
# 실행 (probing/ 안에서):
#   python probe_expB_s4.py --jobs 40

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import glob
import time
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # probing/ (probe_classifier 등)
import probe_classifier as pc  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

Cs = np.logspace(-5, 1, 8)
INNER_CV = 3
MAX_ITER = 150
N_FOLDS = 5
SEEDS = pc.SEEDS  # [53, 9, 144]

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/extract_5_tasks" \
    if False else "/home/hail/vlm3/extract_5_tasks"
FEAT_ROOT = os.path.join(EXTRACT5, "result_expB_s4_tokenlevel")
OUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "result_expB_s4")   # probing/result_expB_s4

MODELS = ["llava", "llava_reverse", "mipha", "mipha_reverse",
         "vila_3b", "vila_3b_reverse", "vila_8b", "vila_8b_reverse"]
VARIANTS = ["2x2_linear", "4x4_linear", "2x2_mlp", "4x4_mlp"]


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


def fit_predict_mlp(X, y, num_classes, seed):
    global pc
    pc.SEED = seed
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for tr_idx, te_idx in pc._kfold_indices(X, y, N_FOLDS, seed):
        oof_pred[te_idx] = pc._fit_predict_fold(X[tr_idx], y[tr_idx], X[te_idx],
                                                num_classes, X.shape[1], dev)
    return oof_pred


def process_one(args):
    model, ckpt, variant, seed, run_idx = args
    out_dir = os.path.join(OUT_ROOT, model, f"run_{run_idx}")
    out_path = os.path.join(out_dir, f"{ckpt}__{variant}.json")
    if os.path.exists(out_path):
        return f"[cached] {model}/{ckpt}/{variant}"

    pt_path = os.path.join(FEAT_ROOT, model, f"{ckpt}.pt")
    if not os.path.exists(pt_path):
        return f"[missing feat] {model}/{ckpt}"

    try:
        data = torch.load(pt_path, map_location="cpu")
        answers = data["answers"]
        labels, num_classes = pc.make_labels(answers)

        key = "feats_2x2" if variant.startswith("2x2") else "feats_4x4"
        X = data[key].float().numpy()
        N = X.shape[0]
        X = X.reshape(N, -1)

        t0 = time.time()
        if variant.endswith("linear"):
            oof = fit_predict_logreg(X, labels, seed)
        else:
            oof = fit_predict_mlp(X, labels, num_classes, seed)
        dt = time.time() - t0

        acc = accuracy_score(labels, oof)
        f1 = f1_score(labels, oof, average="macro")
        result = {"acc": acc, "macro_f1": f1, "n": N, "seconds": dt}

        os.makedirs(out_dir, exist_ok=True)
        json.dump(result, open(out_path, "w"), indent=2)
        return f"[saved] {model}/{ckpt}/{variant} run{run_idx} acc={acc:.3f} ({dt:.1f}s)"
    except Exception as e:
        err_dir = os.path.join(out_dir, "_errors")
        os.makedirs(err_dir, exist_ok=True)
        import traceback
        open(os.path.join(err_dir, f"{ckpt}__{variant}.txt"), "w").write(traceback.format_exc())
        return f"[ERROR] {model}/{ckpt}/{variant}: {type(e).__name__}: {e}"


def run(n_jobs, models=None, variants=None):
    models = models or MODELS
    variants = variants or VARIANTS
    jobs = []
    for model in models:
        ckpts = sorted(os.path.basename(p).replace(".pt", "")
                       for p in glob.glob(os.path.join(FEAT_ROOT, model, "*.pt")))
        for ckpt in ckpts:
            for variant in variants:
                for run_idx, seed in enumerate(SEEDS, start=1):
                    jobs.append((model, ckpt, variant, seed, run_idx))

    print(f"처리할 작업: {len(jobs)}개, n_jobs={n_jobs}")
    t0 = time.time()
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for msg in pool.imap_unordered(process_one, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(process_one(j), flush=True)
    print(f"[done] {len(jobs)} jobs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--models", default=None, help="쉼표구분")
    ap.add_argument("--variants", default=None, help="쉼표구분")
    a = ap.parse_args()
    run(a.jobs, a.models.split(",") if a.models else None,
        a.variants.split(",") if a.variants else None)
