# run_s3_newstim.py
# S3 칩 perturbation 5개 모델 평가 (원본 vs perturb), 추론만. Base/fwd 최종(S4_Depth end)/
# rev 최종(S0_LogMAR end) 3체크포인트 x 5모델. 프롬프트는 원본 S3 origin.json 문구 그대로
# (make_s3_stimuli.py 의 ORIGIN_PROMPT, S3는 애초에 hex 프롬프트가 아니므로 조건 분기 불필요).
#
# 원본 조건도 같은 세트로 같이 돈다(정확도 비교를 위해 동일 체크포인트에서 원본/perturb 나란히).
#
# 실행 (extract_5_tasks/ 안에서, 모델별 conda env):
#   python s1_rebuttal/run_s3_newstim.py --model llava

import os
import sys
import json
import argparse

import torch

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXTRACT5)
sys.path.insert(0, EXTRACT5)

NEWSTIM_DIR = os.path.join(EXTRACT5, "s1_rebuttal", "newstim", "data")

VARIANT_FILES = {
    "orig": "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "perturb": os.path.join(NEWSTIM_DIR, "dataset_s3_perturb.json"),
}

OUT_ROOT = os.path.join(EXTRACT5, "result_s3_rebuttal")

CKPTS = {
    "llava": {
        "Base": "/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b",
        "Final_fwd": "/home/hail/vlm3/checkpoint/llava/S4_Depth/end",
        "Final_rev": "/home/hail/vlm3/checkpoint/llava_reverse/S0_LogMAR/end",
    },
    "mipha": {
        "Base": "/home/hail/vlm3/Model/llava-phi/Mipha-3B-phi2",
        "Final_fwd": "/home/hail/vlm3/checkpoint/Mipha/S4_Depth-phi2/end",
        "Final_rev": "/home/hail/vlm3/checkpoint/Mipha_reverse/S0_LogMAR-phi2/end",
    },
    "vila_3b": {
        "Base": "/home/hail/vlm3/Model/VILA/VILA1.5-3b",
        "Final_fwd": "/home/hail/vlm3/checkpoint/VILA_3B/S4_Depth/end",
        "Final_rev": "/home/hail/vlm3/checkpoint/VILA_3B_reverse/S0_LogMAR/end",
    },
    "vila_8b": {
        "Base": "/home/hail/vlm3/Model/VILA/Llama-3-VILA1.5-8B",
        "Final_fwd": "/home/hail/vlm3/checkpoint/VILA_8B/S4_Depth/end",
        "Final_rev": "/home/hail/vlm3/checkpoint/VILA_8B_reverse/S0_LogMAR/end",
    },
    "openflamingo": {
        "Base": "/home/hail/vlm3/Model/open_flamingo/ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt",
        "Final_fwd": "/home/hail/vlm3/checkpoint/OpenFlamingo/S4_Depth",
        "Final_rev": "/home/hail/vlm3/checkpoint/OpenFlamingo_reverse/S0_LogMAR",
    },
}

TEMPERATURE = 0.0


def load_variant_orig(path):
    """원본 dataset.json 은 conversations 스키마 — {id,image,question,answer}로 정규화."""
    cases = json.load(open(path))
    out = []
    for c in cases:
        out.append({"id": str(c["id"]), "image": c["image"],
                    "question": c["conversations"][0]["value"],
                    "answer": c["conversations"][1]["value"]})
    return out


def load_variant_perturb(path):
    return json.load(open(path))  # 이미 {id,image,question,answer,...} 스키마


def build_loader(model, ckpt_path):
    if model == "llava":
        import inf_llava
        return inf_llava.LLaVAInferExtract(ckpt_path)
    if model == "mipha":
        import inf_mipha
        return inf_mipha.MiphaInferExtract(ckpt_path)
    if model == "vila_3b":
        import inf_vila
        return inf_vila.VILAInferExtract(ckpt_path, "vicuna_v1")
    if model == "vila_8b":
        import inf_vila
        return inf_vila.VILAInferExtract(ckpt_path, "llama_3")
    raise ValueError(model)


def run_one_generic(model, loader, image, question):
    if model == "openflamingo":
        pred, _feat, _proj = loader.run_one(image, question, TEMPERATURE)
    else:
        pred, _feat, _proj = loader.run_one(image, question, [0], TEMPERATURE)
    return pred


def run(model, variants=None, ckpt_names=None):
    ckpts = CKPTS[model]
    if ckpt_names:
        ckpts = {k: v for k, v in ckpts.items() if k in ckpt_names}
    variants = variants or list(VARIANT_FILES)

    of_loader = None
    if model == "openflamingo":
        import inf_openflamingo
        of_loader = inf_openflamingo.OFInferExtract()

    for ckpt_name, ckpt_path in ckpts.items():
        print(f"\n########## {model} / {ckpt_name} ##########")
        if model == "openflamingo":
            of_loader.load_ckpt(ckpt_path)
            loader = of_loader
        else:
            loader = build_loader(model, ckpt_path)

        for variant_name in variants:
            fname = VARIANT_FILES[variant_name]
            samples = load_variant_orig(fname) if variant_name == "orig" else load_variant_perturb(fname)
            out_dir = os.path.join(OUT_ROOT, model, ckpt_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{variant_name}.json")
            if os.path.exists(out_path):
                print(f"  [cached] {variant_name}")
                continue

            records = []
            for s in samples:
                pred = run_one_generic(model, loader, s["image"], s["question"])
                rec = {"id": s["id"], "output": pred, "answer": s["answer"]}
                records.append(rec)
            json.dump(records, open(out_path, "w"), indent=2, ensure_ascii=False)
            print(f"  [saved] {variant_name} n={len(records)} -> {out_path}")

        if model != "openflamingo":
            del loader
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS))
    ap.add_argument("--variants", default=None, help="쉼표구분, 미지정시 전체")
    ap.add_argument("--ckpts", default=None, help="쉼표구분, 미지정시 전체")
    a = ap.parse_args()
    run(a.model, a.variants.split(",") if a.variants else None,
        a.ckpts.split(",") if a.ckpts else None)
