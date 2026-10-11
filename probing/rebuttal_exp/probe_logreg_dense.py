# probe_logreg_dense.py
# 1onr W1 대응: 배터리 과제(S0/S2/S3/S4) 선형 probe, Base 체크포인트만, 사용자가 원래 지정한
# C ∈ logspace(-4,2,25) 그대로(축소 안 함) — Base 1개 체크포인트로 범위를 좁혔기 때문에
# 25점 그리드도 계산 가능. probe_logreg.py를 그대로 import해서 재사용하되 Cs와 저장 폴더만
# 바꾼다(기존 8점-그리드 전체배터리 결과와 섞이지 않도록 별도 폴더에 저장).
#
# 실행 (probing/ 안에서):
#   python probe_logreg_dense.py --model llava --jobs 8
#   python probe_logreg_dense.py --model llava --ckpt Final_fwd --jobs 8   # 여유되면 fwd 최종

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # probing/ (probe_classifier 등)
import probe_logreg as base  # noqa: E402

base.Cs = np.logspace(-4, 2, 25)   # 사용자 지정 그리드 그대로(8점 축소 없음)


def run_dense(model, source, ckpt, task, n_jobs):
    import glob
    import time
    from multiprocessing import Pool

    emb_dir = base.pc.EMB_ROOT[source].format(model=model)
    tasks_filter = set(task.split(",")) if task else None
    pat = os.path.join(emb_dir, ckpt or "*", "*.pt")
    pts = sorted(glob.glob(pat))
    if not pts:
        print(f"처리할 .pt 없음. (pat={pat})")
        return

    jobs = []
    for run_idx, seed in enumerate(base.SEEDS, start=1):
        res_dir = f"./{model}/{source}_classification_logreg25_{run_idx}"  # 별도 폴더 (25점 그리드)
        for pt in pts:
            jobs.append((pt, seed, run_idx, res_dir, False, tasks_filter))  # proj_only=False -> 전 레이어+proj

    print(f"처리할 작업: {len(jobs)}개 (.pt={len(pts)} x seed={len(base.SEEDS)}), n_jobs={n_jobs}, "
          f"Cs={len(base.Cs)}pt(dense), ckpt={ckpt}")
    t0 = time.time()
    if n_jobs > 1:
        with Pool(n_jobs) as pool:
            for msg in pool.imap_unordered(base.process_one, jobs):
                if msg:
                    print(msg, flush=True)
    else:
        for j in jobs:
            msg = base.process_one(j)
            if msg:
                print(msg, flush=True)
    print(f"\n[done] {len(jobs)} jobs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", default="inf", choices=list(base.pc.EMB_ROOT))
    ap.add_argument("--ckpt", default="S0_start", help="기본 Base(S0_start)만")
    ap.add_argument("--task", default="S0_logamr,S2_ishihara,S3_naming,S4_depth")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    run_dense(a.model, a.source, a.ckpt, a.task, a.jobs)
