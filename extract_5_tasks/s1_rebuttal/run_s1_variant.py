# run_s1_variant.py
# S1 hex-shortcut rebuttal 실험: 지정한 체크포인트 × 변형 데이터셋(orig/a/b/c/d) 조합으로
# 추론만 수행한다. 디코딩/로딩 로직은 inf_llava.py 의 LLaVAInferExtract.run_one() 을
# 그대로 재사용 — 변형 실험 코드가 원본 파이프라인을 바꾸지 않는다는 걸 보장하기 위함.
#
# 실행 (extract_5_tasks/ 안에서, llava conda env):
#   python s1_rebuttal/run_s1_variant.py --model llava

import os
import sys
import json
import argparse

import torch

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXTRACT5)  # inf_llava.py 내부 상대경로("../Model/LLaVA" 등)가 여길 기준으로 함
sys.path.insert(0, EXTRACT5)

VARIANTS_DIR = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants"
VARIANT_FILES = {
    "orig": "dataset_orig.json",
    "a_text_only": "dataset_a.json",
    "b_mismatch": "dataset_b.json",
    "c_nohex": "dataset_c.json",
    "d_gray_nohex": "dataset_d.json",
    "b_mismatch_v2": "dataset_b_v2.json",
}
OUT_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal")

LAYERS = [0, 8, 16, 24, 32]
TEMPERATURE = 0.0

CKPTS = {
    "llava": {
        "Base":         "/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b",
        "S1_fwd":       "/home/hail/vlm3/checkpoint/llava/S1_Similarity/end",
        "S1_rev":       "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity/end",
        "Final_fwd_S4": "/home/hail/vlm3/checkpoint/llava/S4_Depth/end",
        "Final_rev_S0": "/home/hail/vlm3/checkpoint/llava_reverse/S0_LogMAR/end",
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/llava/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity_nohex/end",
    },
    "vila_3b": {
        "Base":   "/home/hail/vlm3/Model/VILA/VILA1.5-3b",
        "S1_fwd": "/home/hail/vlm3/checkpoint/VILA_3B/S1_Similarity/end",
        "S1_rev": "/home/hail/vlm3/checkpoint/VILA_3B_reverse/S1_Similarity/end",
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/VILA_3B/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/VILA_3B_reverse/S1_Similarity_nohex/end",
    },
}


def load_variant(path):
    with open(path) as f:
        return json.load(f)


def make_loader(model, ckpt_path):
    """모델별 로더 생성. run_one(image_path, question, layers, temperature) 시그니처는 공통."""
    if model == "llava":
        import inf_llava
        return inf_llava.LLaVAInferExtract(ckpt_path)
    if model == "vila_3b":
        import inf_vila
        return inf_vila.VILAInferExtract(ckpt_path, "vicuna_v1")
    if model == "vila_8b":
        import inf_vila
        return inf_vila.VILAInferExtract(ckpt_path, "llama_3")
    raise ValueError(f"unsupported model: {model}")


def run(model, variants=None, ckpt_names=None):
    ckpts = ({k: CKPTS[model][k] for k in ckpt_names} if ckpt_names else CKPTS[model])
    variant_items = ({k: VARIANT_FILES[k] for k in variants} if variants else VARIANT_FILES)

    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {model} / {ckpt_name} ##########")
        loader = make_loader(model, ckpt_path)
        for variant_name, fname in variant_items.items():
            samples = load_variant(os.path.join(VARIANTS_DIR, fname))
            out_dir = os.path.join(OUT_ROOT, model, ckpt_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{variant_name}.json")

            records = []
            for s in samples:
                pred, _feat, _proj = loader.run_one(s["image"], s["question"], LAYERS, TEMPERATURE)
                rec = {"id": s["id"], "question": s["question"], "output": pred,
                       "sim_img": s["sim_img"]}
                if "sim_text" in s:
                    rec["sim_text"] = s["sim_text"]
                    rec["text_source_id"] = s["text_source_id"]
                    rec["gap"] = s["gap"]
                records.append(rec)

            with open(out_path, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"[saved] {ckpt_name}/{variant_name} → {out_path}  n={len(records)}")

        del loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llava", choices=list(CKPTS))
    ap.add_argument("--variants", default=None, help="쉼표구분, 미지정시 전체 (예: b_mismatch_v2)")
    ap.add_argument("--ckpts", default=None, help="쉼표구분, 미지정시 전체 (예: S1_fwd_nohex,S1_rev_nohex)")
    a = ap.parse_args()
    run(a.model, a.variants.split(",") if a.variants else None,
        a.ckpts.split(",") if a.ckpts else None)
