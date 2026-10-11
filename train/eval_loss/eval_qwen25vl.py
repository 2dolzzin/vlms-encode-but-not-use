#!/usr/bin/env python
# eval_qwen25vl.py
# Qwen2.5-VL-7B 커리큘럼 11시점 ckpt 별로 5 stage held-out 셋의 eval_loss 를 계산 (망각 측정).
# llava/mipha/vila 는 각 repo 의 train.py 를 몽키패치해서 Trainer.evaluate 를 썼지만,
# Qwen 은 순수 HF 모델이라 train/qwen25vl_curriculum/qwen_data.py 의 데이터셋/콜레이터로 직접 계산한다.
#
# loss 정의는 HF Trainer eval_loss 와 동일: 배치별 (토큰 평균 CE) 의 배치 평균.
# 출력 json 포맷은 results_llava/*.json 과 동일: {"model": tag, "model_path": path, "losses": {S0:..,S4:..}}
#
# 실행 (conda activate qwenvl):
#   python eval_qwen25vl.py --model <ckpt_dir> --tag fwd__S0_mid --out results_qwen25vl_7b/fwd__S0_mid.json
#   (보통은 run_forgetting_qwen25vl.sh 가 11시점 × fwd/rev 를 돌린다)

import os
import sys
import json
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "qwen25vl_curriculum"))
from qwen_data import DataCollatorForQwenVL, LazySupervisedDataset, load_model, load_processor  # noqa: E402

DEFAULT_EVAL_SETS = [
    ["S0", "/home/hail/HDD/vlm_validation/S0_Logmar/dataset/logmar_non_flamingo.json"],
    ["S1", "/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/rebuttal_nohex/color_similarity_non_flamingo_nohex.json"],  # nohex
    ["S2", "/home/hail/HDD/vlm_validation/S2_Ishihara/dataset/ishihara_non_flamingo.json"],
    ["S3", "/home/hail/HDD/vlm_validation/S3_Color_Naming/dataset/color_naming_non_flamingo.json"],
    ["S4", "/home/hail/HDD/vlm_validation/S4_Depth/dataset/depth_non_flamingo.json"],
]


@torch.inference_mode()
def eval_loss(model, ds, collator, batch_size, num_workers, device, desc):
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=collator, pin_memory=True)
    total, n = 0.0, 0
    for batch in tqdm(dl, desc=desc, ncols=90):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.float().item()
        n += 1
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ckpt 디렉토리 (base 또는 checkpoint-*/end)")
    ap.add_argument("--tag", required=True, help="예: fwd__S0_mid")
    ap.add_argument("--out", required=True, help="결과 json 경로")
    ap.add_argument("--eval_sets_json", default=os.environ.get("EVAL_SETS_JSON"),
                    help='[["S0","path"],...] (없으면 기본 5개)')
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--model_max_length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="디버그용: 셋당 앞 N 개만")
    a = ap.parse_args()

    if os.path.exists(a.out):
        print(f"[skip] {a.out} 이미 있음")
        return
    eval_sets = json.loads(a.eval_sets_json) if a.eval_sets_json else DEFAULT_EVAL_SETS

    device = "cuda"
    processor = load_processor(a.model)
    model = load_model(a.model, dtype=torch.bfloat16, device_map=device).eval()
    collator = DataCollatorForQwenVL(processor.tokenizer)

    results = {}
    for name, path in eval_sets:
        ds = LazySupervisedDataset(path, processor, a.model_max_length)
        if a.limit:
            ds.records = ds.records[: a.limit]
        loss = eval_loss(model, ds, collator, a.batch_size, a.num_workers, device, f"{a.tag}/{name}")
        results[name] = loss
        print(f"[eval_qwen25vl] {a.tag} {name} eval_loss = {loss:.4f} (n={len(ds)})", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.tag, "model_path": a.model, "losses": results}, f, indent=2)
    print(f"[eval_qwen25vl] saved → {a.out}")


if __name__ == "__main__":
    main()
