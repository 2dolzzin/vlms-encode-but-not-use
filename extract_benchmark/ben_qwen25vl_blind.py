import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ben_qwen25vl_blind.py
# ben_qwen25vl.py 에서 채점(evaluate_one) 을 뺀 blind 벤치 전용 추론 스크립트.
#   blind 8 task 추론 + 임베딩 추출만 수행 (raw_responses + embeddings).
#   결과는 result/<out>_blind/ 밑에 저장. 채점은 acc_blind.py 가 담당.
#
# 실행 (extract_benchmark/ 안에서, conda activate qwenvl):
#   python ben_qwen25vl_blind.py --run qwen25vl_7b
#   python ben_qwen25vl_blind.py --run all --tasks counting_grid_blank
# 채점:
#   python acc_blind.py --model qwen25vl_7b_blind
#   python acc_blind.py --model qwen25vl_7b_reverse_blind

import sys
import json
import argparse

import torch
from tqdm import tqdm

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
# 로더/ckpt 탐색/데이터 로드는 ben_qwen25vl.py 것을 그대로 재사용 (blind 는 채점만 다름)
import ben_qwen25vl as B  # noqa: E402

LAYERS = B.LAYERS
MAX_NEW_TOKENS = 64          # blind 는 짧은 답 (기존 ben_*_blind.py 와 동일)
B.MAX_NEW_TOKENS = MAX_NEW_TOKENS

RUNS = {
    "qwen25vl_7b":         dict(root=B.RUNS["qwen25vl_7b"]["root"],         out="qwen25vl_7b_blind"),
    "qwen25vl_7b_reverse": dict(root=B.RUNS["qwen25vl_7b_reverse"]["root"], out="qwen25vl_7b_reverse_blind"),
}
# 무작위 순서 런: ben_qwen25vl.py 가 자동 등록한 "qwen25vl_7b_order_<tag>" 를 그대로 승계 (out 만 _blind)
for _k, _v in B.RUNS.items():
    if "_order_" in _k:
        RUNS[_k] = dict(root=_v["root"], out=_k + "_blind")
from order_runs import read_order, describe_ckpts  # noqa: E402  (B 가 sys.path 에 저장소 루트를 넣어둠)

BLIND_ROOT = "/home/hail/HDD/blind/unpacked"
DATASETS = {
    "circled_letter":      f"{BLIND_ROOT}/Circled_Letter/dataset.json",
    "counting_grid_blank": f"{BLIND_ROOT}/Counting_Grid_-_Blank_Grids/dataset.json",
    "counting_grid_word":  f"{BLIND_ROOT}/Counting_Grid_-_Word_Grids/dataset.json",
    "nested_squares":      f"{BLIND_ROOT}/Nested_Squares/dataset.json",
    "olympic_circles":     f"{BLIND_ROOT}/Olympic_Counting_-_Circles/dataset.json",
    "olympic_pentagons":   f"{BLIND_ROOT}/Olympic_Counting_-_Pentagons/dataset.json",
    "subway_connections":  f"{BLIND_ROOT}/Subway_Connections/dataset.json",
    "touching_circles":    f"{BLIND_ROOT}/Touching_Circles/dataset.json",
    # 전이 평가(실험 3)의 나머지 둘. line_plot 은 기존 fwd/rev 결과(result_origin)와 같은 샘플셋을 써서
    # 비교 가능하게 한다 — blind/unpacked/Line_Plot_Intersections 는 같은 task 지만 샘플이 다름(이미지 교집합 0).
    "line_plot": "/home/hail/HDD/vlm_inference/benchmark/Task_lowlevel/line_plot/dataset.json",
    "mmvp":      "/home/hail/HDD/vlm_inference/benchmark/MMVP/dataset.json",
}


def process_dataset(loader, ckpt_name, ds_name, ds_path, temperature, dirs):
    _pp = os.path.join(dirs["pred"], ckpt_name, f"{ds_name}.json")
    _ep = os.path.join(dirs["emb"], ckpt_name, f"{ds_name}.pt")
    if os.path.exists(_pp) and os.path.exists(_ep):
        print(f"[skip] {ckpt_name}/{ds_name} (이미 완료)")
        return

    samples = B.load_dataset(ds_path)
    records, feats, proj_feats = [], [], []
    predictions, answers, ids = [], [], []
    for s in tqdm(samples, desc=f"{ckpt_name}/{ds_name}", ncols=90):
        pred, feat, p_feat = loader.run_one(s["image_path"], s["question"], LAYERS, temperature)
        records.append(dict(id=s["id"], question=s["question"],
                            output=pred, expected_output=s["answer"]))
        feats.append(feat); proj_feats.append(p_feat)
        predictions.append(pred); answers.append(s["answer"]); ids.append(s["id"])

    feats = torch.stack(feats).half()
    proj_feats = torch.stack(proj_feats).half()

    os.makedirs(os.path.dirname(_pp), exist_ok=True)
    with open(_pp, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(_ep), exist_ok=True)
    torch.save({"feats": feats, "proj_feats": proj_feats, "predictions": predictions,
                "answers": answers, "ids": ids, "layers": LAYERS,
                "hidden_size": loader.hidden_size}, _ep)
    print(f"[saved] {ckpt_name}/{ds_name}  feats={tuple(feats.shape)} proj={tuple(proj_feats.shape)}")


def run(run_key, temperature):
    cfg = RUNS[run_key]
    dirs = B.make_dirs(cfg["out"])
    ckpts = B.build_ckpts(cfg["root"])
    print(f"[{run_key}] 처리할 ckpt {len(ckpts)}개: {list(ckpts)}")
    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {run_key} / {ckpt_name} ##########")
        loader = B.QwenInferExtract(ckpt_path)
        for ds_name in DATASETS:
            process_dataset(loader, ckpt_name, ds_name, DATASETS[ds_name], temperature, dirs)
        del loader
        torch.cuda.empty_cache()
    print(f"\n[done] 채점: python acc_blind.py --model {cfg['out']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", choices=list(RUNS) + ["all"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tasks", default=None,
                    help="쉼표구분 task 만 실행 (예: counting_grid_word,circled_letter). 미지정=전체")
    ap.add_argument("--list", action="store_true",
                    help="모델 로드 없이 run 별 체크포인트 11시점만 출력(드라이런)")
    a = ap.parse_args()
    if a.list:
        for rk in (list(RUNS) if a.run == "all" else [a.run]):
            ck = B.build_ckpts(RUNS[rk]["root"]); print(f"[{rk}] {len(ck)} ckpt  root={RUNS[rk]['root']}")
            print(describe_ckpts(ck, read_order(RUNS[rk]["root"]))); print()
        raise SystemExit(0)
    if a.tasks:
        keep = [t.strip() for t in a.tasks.split(",")]
        for _k in list(DATASETS):
            if _k not in keep:
                DATASETS.pop(_k)
        print(f"[tasks] 실행 대상: {list(DATASETS)}")
    for rk in (list(RUNS) if a.run == "all" else [a.run]):
        print(f"\n================= RUN: {rk} =================")
        run(rk, a.temperature)
