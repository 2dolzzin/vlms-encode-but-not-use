#!/usr/bin/env python3
"""학습 / 5-task / blind / 구 벤치 / 프로빙 진행 현황표 (디스크 상태로 판정). 실행: python tools/status.py"""
import os, glob, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (tools/ 의 상위)
sys.path.insert(0, BASE)
from order_runs import read_order

MODELS = [  # (표시명, run 키, 체크포인트 폴더 접두사)
    ("llava", "llava", "llava"), ("vila-3B", "vila_3b", "VILA_3B"), ("vila-8B", "vila_8b", "VILA_8B"),
    ("mipha", "mipha", "Mipha"), ("openflamingo", "openflamingo", "OpenFlamingo"), ("qwen", "qwen25vl_7b", "Qwen25VL_7B"),
]
TAGS = [("14203", "1→4→2→0→3"), ("20431", "2→0→4→3→1"), ("31024", "3→1→0→2→4")]
COLS = ["forward", "backward"] + [t[1] for t in TAGS]

def mark(n, total):
    return "✅" if n >= total else ("❌" if n == 0 else f"🔄 {n}/{total}")

def ckpt_root(prefix, col):
    if col == "forward":  return f"{BASE}/checkpoint/{prefix}"
    if col == "backward": return f"{BASE}/checkpoint/{prefix}_reverse"
    return f"{BASE}/checkpoint/{prefix}_order_{col}_nohex"

def run_key(key, col):
    if col == "forward":  return key
    if col == "backward": return key + "_reverse"
    return f"{key}_order_{col}"

def training(prefix, col):
    root = ckpt_root(prefix, col)
    if not os.path.isdir(root): return 0
    pat = "[0-9]_S*" if col not in ("forward", "backward") else "S*"
    done = 0
    for d in glob.glob(os.path.join(root, pat)):
        if os.path.isfile(os.path.join(d, "flamingo_model.pt")) or glob.glob(os.path.join(d, "end", "*.safetensors")) \
           or os.path.isdir(os.path.join(d, "end", "llm")):
            done += 1
    return min(done, 5)

def n_ckpts(path):  # raw_responses 아래 ckpt 폴더 수 (완료 = 11)
    return len([d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d) and glob.glob(d + "/*.json")])

def probing(run):
    parts = [f"{BASE}/probing/{run}/inf_classification_{i}" for i in (1, 2, 3)] \
          + [f"{BASE}/probing/{run}/inf_regression_{i}/ridge" for i in (1, 2, 3)] \
          + [f"{BASE}/probing/{run}_blind/ben_classification_{i}" for i in (1, 2, 3)]
    return sum(1 for p in parts if glob.glob(p + "/*.json")), len(parts)

def table(title, cell):
    print(f"\n### {title}\n")
    print("|  | " + " | ".join(COLS) + " |"); print("| --- |" + " --- |" * len(COLS))
    for name, key, prefix in MODELS:
        print(f"| {name} | " + " | ".join(cell(key, prefix, c) for c in ("forward", "backward", *[t[0] for t in TAGS])) + " |")

table("학습현황", lambda k, p, c: mark(training(p, c), 5))
table("inference_5task 현황", lambda k, p, c: mark(n_ckpts(f"{BASE}/extract_5_tasks/result/{run_key(k,c)}/inference/raw_responses"), 11))
table("inference_blind 현황", lambda k, p, c: mark(n_ckpts(f"{BASE}/extract_benchmark/result/{run_key(k,c)}_blind/raw_responses"), 11))
def n_ckpts_task(run, task):   # blind 결과 또는 구 result_origin 중 해당 task json 이 있는 ckpt 수의 최대
    best = 0
    for d in (f"{BASE}/extract_benchmark/result/{run}_blind/raw_responses", f"{BASE}/extract_benchmark/result_origin/{run}/raw_responses"):
        best = max(best, len(glob.glob(os.path.join(d, "*", f"{task}.json"))))
    return best
table("inference_mmvp+line_plot 현황 (blind 폴더 또는 result_origin)", lambda k, p, c: mark(min(n_ckpts_task(run_key(k,c), "mmvp"), n_ckpts_task(run_key(k,c), "line_plot")), 11))
table("probing 현황 (inf_clf×3 + inf_reg×3 + blind ben_clf×3 = 9)", lambda k, p, c: mark(*probing(run_key(k, c))))
