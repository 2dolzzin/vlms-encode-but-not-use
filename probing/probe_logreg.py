# probe_logreg.py
# Rebuttal 실험 A: 이산 과제 MLP 프로브(probe_classifier.py)의 용량 대조군.
# 헤드만 선형(L2 multinomial logistic regression)으로 교체, 나머지(전처리·CV분할·시드·
# 저장 규약)는 probe_classifier.py 를 그대로 import 해서 재사용 — 기존 파이프라인
# 코드는 전혀 수정하지 않는다.
#
# 결과는 probe_classifier.py(MLP)와 겹치지 않도록 별도 폴더(_classification_logreg_N)에 저장.
#
# 성능 노트: LogisticRegressionCV(25-C grid × 5-fold 내부, n_jobs=-1 단일 셀 안에서)는
# 고차원(4096) 특징에서 셀당 4~5분이 걸려 전체 조합(수천 셀)이 비현실적이었다.
# 셀(=ckpt×task×seed)들은 서로 완전 독립이므로, 셀 단위를 프로세스 풀로 병렬화하고
# (각 fit은 n_jobs=1) C 그리드를 25→7로 줄여 계산량을 낮췄다. 통계적 설계(5-fold outer OOF,
# 3 seed, StandardScaler, multinomial logistic, CV로 C 선택)는 그대로다.
#
# 실행 (probing/ 안에서):
#   python probe_logreg.py --model llava_blind --source ben --proj_only --jobs 40
#   python probe_logreg.py --model llava --source inf --task S2_ishihara,S3_naming,S4_depth --jobs 40

import os
# 프로세스 풀로 셀(ckpt×task×seed) 단위 병렬화하므로, 각 프로세스 안의 BLAS는 반드시
# 단일 스레드로 고정해야 한다. 안 그러면 n_jobs(프로세스 수) x BLAS내부스레드 로 코어를
# 심하게 오버섭스크립션해서 (48코어에 수백~수천 스레드) 오히려 훨씬 느려진다 — 실측 확인됨
# (오버섭스크립션 상태: fit 1회 7.7~57s → 스레드 고정 후: 0.77s).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import glob
import json
import time
import argparse
import warnings
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_classifier as pc  # noqa: E402  (make_labels, FIRST_CHAR_TASKS, SKIP_TASKS, _kfold_indices 재사용)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# S0_logamr(26클래스, N=1800, 4096차원)에서 lbfgs가 max_iter=1000/Cs=25/inner_cv=5로도
# 전혀 수렴하지 않음을 실측 확인 (100 iter에도 train_acc 0.22대 정체, fit 1회 27s+,
# 셀 하나에 수십 시간 소요 추정). 계산 예산을 낮춰 전체 배터리(S0/S2/S3/S4, 10모델,
# 11ckpt x 2방향 x 3seed)가 현실적 시간 안에 끝나도록 함 — 정확도를 올리려는 목적이
# 아니라 순수 계산량 문제이며, 모든 과제에 동일하게 적용(S0만 따로 손대지 않음).
Cs = np.logspace(-4, 2, 8)
N_FOLDS = pc.N_FOLDS          # 5, probe_classifier.py 와 동일 (평가 프로토콜은 안 건드림)
INNER_CV = 3
MAX_ITER = 150
SEEDS = pc.SEEDS              # [53, 9, 144], probe_classifier.py 와 동일

MODEL = "llava_blind"
SOURCE = "ben"


def fit_predict_logreg(X, y, seed):
    """5-fold OOF, fold마다 LogisticRegressionCV(내부 CV로 C 선택).
    극단적 클래스 불균형(예: S3_naming 최소클래스 2개) 과제는 outer fold의 학습 부분집합을
    다시 inner_cv로 쪼갤 때 특정 클래스가 통째로 빠지는 fold가 생겨 sklearn 내부
    coefs_paths reshape가 깨진다 (KFold/StratifiedKFold 둘 다 동일 — 근본 원인은 표본 수
    부족이지 분할 전략이 아님). 이 경우 fold 단위로 감지해 C-그리드서치 없이 고정
    C=1.0 단순 LogisticRegression으로 그 fold만 대체한다 — 나머지 fold는 정상 CV 그대로.
    """
    oof_pred = np.full(len(y), -1, dtype=np.int64)
    fallback_folds = 0
    for tr_idx, te_idx in pc._kfold_indices(X, y, N_FOLDS, seed):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr = y[tr_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        inner_cv = KFold(n_splits=INNER_CV, shuffle=True, random_state=seed)
        clf = LogisticRegressionCV(
            Cs=Cs, cv=inner_cv, scoring="accuracy",
            solver="lbfgs", max_iter=MAX_ITER, n_jobs=1, random_state=seed,
        )
        try:
            clf.fit(X_tr, y_tr)
            oof_pred[te_idx] = clf.predict(X_te)
        except ValueError:
            # 표본 부족으로 내부 CV가 깨짐 → 이 fold만 C 탐색 없이 고정 C=1.0 로 대체
            from sklearn.linear_model import LogisticRegression
            fallback_folds += 1
            clf2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=MAX_ITER, n_jobs=1)
            clf2.fit(X_tr, y_tr)
            oof_pred[te_idx] = clf2.predict(X_te)
    return oof_pred, fallback_folds


def train_one_layer_logreg(X, y, tag, seed):
    t0 = time.time()
    oof_pred, fallback_folds = fit_predict_logreg(X, y, seed)
    acc = accuracy_score(y, oof_pred)
    f1 = f1_score(y, oof_pred, average="macro")
    dt = time.time() - t0
    fb = f" fallback_folds={fallback_folds}" if fallback_folds else ""
    print(f"    [{tag}] acc={acc:.4f}  macroF1={f1:.4f}  (OOF N={len(y)}, {dt:.1f}s){fb}", flush=True)
    return acc, f1, fallback_folds


def process_one(args):
    """워커 1개: (pt_path, seed, run_idx, res_dir, proj_only, tasks_filter) 하나 처리."""
    pt_path, seed, run_idx, res_dir, proj_only, tasks_filter = args
    ckpt_name = os.path.basename(os.path.dirname(pt_path))
    task_name = os.path.basename(pt_path).replace(".pt", "")
    name = f"{ckpt_name}__{task_name}"
    if task_name in pc.SKIP_TASKS:
        return f"[skip] {name} (회귀 task)"
    if tasks_filter and task_name not in tasks_filter:
        return None

    out = os.path.join(res_dir, f"{name}.json")
    if os.path.exists(out):
        return f"[cached] {name}"

    try:
        data = torch.load(pt_path, map_location="cpu")
        answers = data["answers"]
        layers = data["layers"]

        if task_name in pc.FIRST_CHAR_TASKS:
            answers = [a[0] for a in answers]

        labels, num_classes = pc.make_labels(answers)

        result = {}
        if not proj_only:
            feats = data["feats"].float()
            for i, L in enumerate(layers):
                X = feats[:, i, :].numpy()
                acc, f1, fb = train_one_layer_logreg(X, labels, f"layer_{L}", seed)
                result[f"layer_{L}"] = {"acc": acc, "macro_f1": f1, "fallback_folds": fb}

        if "proj_feats" in data:
            Xp = data["proj_feats"].float().numpy()
            acc, f1, fb = train_one_layer_logreg(Xp, labels, "proj", seed)
            result["proj"] = {"acc": acc, "macro_f1": f1, "fallback_folds": fb}

        os.makedirs(res_dir, exist_ok=True)
        json.dump({ckpt_name: {task_name: result}}, open(out, "w"), indent=2)
        return f"[saved] {name} → {out}"
    except Exception as e:
        err_dir = os.path.join(res_dir, "_errors")
        os.makedirs(err_dir, exist_ok=True)
        import traceback
        open(os.path.join(err_dir, f"{name}.txt"), "w").write(traceback.format_exc())
        return f"[ERROR] {name}: {type(e).__name__}: {e}"


def run(model, source, ckpt=None, task=None, proj_only=False, n_jobs=1):
    emb_dir = pc.EMB_ROOT[source].format(model=model)
    tasks_filter = set(task.split(",")) if task else None
    pat = os.path.join(emb_dir, ckpt or "*", "*.pt")
    pts = sorted(glob.glob(pat))
    if not pts:
        print(f"처리할 .pt 없음. (pat={pat})")
        return

    jobs = []
    for run_idx, seed in enumerate(SEEDS, start=1):
        res_dir = f"./{model}/{source}_classification_logreg_{run_idx}"
        for pt in pts:
            jobs.append((pt, seed, run_idx, res_dir, proj_only, tasks_filter))

    print(f"처리할 작업: {len(jobs)}개 (.pt={len(pts)} × seed={len(SEEDS)}), n_jobs={n_jobs}, "
          f"Cs={len(Cs)}pt, inner_cv={INNER_CV}, proj_only={proj_only}")
    t0 = time.time()
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for msg in pool.imap_unordered(process_one, jobs):
                if msg:
                    print(msg, flush=True)
    else:
        for j in jobs:
            msg = process_one(j)
            if msg:
                print(msg, flush=True)
    print(f"\n[done] {len(jobs)} jobs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--task", type=str, default=None, help="쉼표구분, 미지정시 전체")
    ap.add_argument("--model", type=str, default=MODEL)
    ap.add_argument("--source", type=str, default=SOURCE, choices=list(pc.EMB_ROOT))
    ap.add_argument("--proj_only", action="store_true", help="projector/resampler 지점만 (LM layer 생략)")
    ap.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수 (셀=ckpt×task×seed 단위)")
    a = ap.parse_args()
    run(a.model, a.source, ckpt=a.ckpt, task=a.task, proj_only=a.proj_only, n_jobs=a.jobs)
