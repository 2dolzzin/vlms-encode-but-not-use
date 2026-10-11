# probe_logreg_s0_reduced.py
# S0_logamr 은 N=1800/26-way라 25점 C그리드에서 지점(layer)당 seed 1개 fit이 ~18분(layer_8,
# llava 실측)까지 걸려, 5모델 전체를 6지점(layer 0/8/16/24/32/proj)으로 돌리면 비현실적이다.
# projector(0.13대)와 LM 최종층(0.89대) 두 끝점이면 "선형으로도 재현되는가"라는 질문에는
# 충분하다는 판단으로 이 두 지점만, C그리드도 8점(원래 Exp A 배터리에서 쓰던 축소 그리드,
# 새로 발명한 게 아니라 이미 검증된 값)으로 계산한다. 3 seed 유지.
#
# probe_logreg.py 의 fit_predict_logreg/process_one 은 한 .pt 파일의 전체 layer+proj 를
# 한번에 처리해 하나의 JSON으로 저장하는 구조라 "2개 지점만" 부분처리가 안 됨 — 여기서는
# 별도로 proj/최종층만 뽑아 독립적으로 처리한다.
#
# 실행 (probing/ 안에서):
#   python probe_logreg_s0_reduced.py --model mipha --jobs 12

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time
import argparse
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # probing/ (probe_classifier 등)
import probe_logreg as base  # noqa: E402
import probe_classifier as pc  # noqa: E402

Cs_S0 = np.logspace(-4, 2, 8)   # Exp A 배터리에서 이미 쓰던 축소 그리드 그대로 재사용
TASK = "S0_logamr"


def process_point(args):
    pt_path, seed, run_idx, res_dir, point_name = args
    out = os.path.join(res_dir, f"S0_start__{TASK}__{point_name}.json")
    if os.path.exists(out):
        return f"[cached] {point_name} seed{seed}"

    data = torch.load(pt_path, map_location="cpu")
    answers = data["answers"]
    layers = data["layers"]
    if TASK in pc.FIRST_CHAR_TASKS:   # S0_logamr: 답 문자열 첫 글자만 라벨로 (원본 process_one과 동일)
        answers = [a[0] for a in answers]
    labels, num_classes = pc.make_labels(answers)

    if point_name == "proj":
        X = data["proj_feats"].float().numpy()
    else:  # "final" = 마지막 layer (모델별 100% 깊이)
        li = len(layers) - 1
        X = data["feats"][:, li, :].float().numpy()

    # base.Cs 를 이 호출 동안만 8점으로 바꿔서 base.fit_predict_logreg 그대로 재사용
    orig_cs = base.Cs
    base.Cs = Cs_S0
    t0 = time.time()
    oof_pred, fallback = base.fit_predict_logreg(X, labels, seed)
    base.Cs = orig_cs
    dt = time.time() - t0

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(labels, oof_pred)
    f1 = f1_score(labels, oof_pred, average="macro")

    os.makedirs(res_dir, exist_ok=True)
    result = {"S0_start": {TASK: {point_name: {"acc": acc, "macro_f1": f1,
                                                "fallback_folds": fallback, "seconds": dt}}}}
    json.dump(result, open(out, "w"), indent=2)
    return f"[saved] {point_name} seed{seed} acc={acc:.4f} ({dt:.1f}s)"


def run(model, n_jobs):
    emb_dir = pc.EMB_ROOT["inf"].format(model=model)
    pt_path = os.path.join(emb_dir, "S0_start", f"{TASK}.pt")
    if not os.path.exists(pt_path):
        print(f"없음: {pt_path}"); return

    jobs = []
    for run_idx, seed in enumerate(base.SEEDS, start=1):
        res_dir = f"./{model}/inf_classification_logreg25_{run_idx}"
        for point in ("proj", "final"):
            jobs.append((pt_path, seed, run_idx, res_dir, point))

    print(f"S0 축소 실행: {model}, {len(jobs)}개 작업(2지점 x 3seed), Cs=8점, jobs={n_jobs}")
    t0 = time.time()
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for msg in pool.imap_unordered(process_point, jobs):
                print(msg, flush=True)
    else:
        for j in jobs:
            print(process_point(j), flush=True)
    print(f"[done] {len(jobs)} jobs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    run(a.model, a.jobs)
