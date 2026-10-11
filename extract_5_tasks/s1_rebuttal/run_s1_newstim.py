# run_s1_newstim.py
# S1 신규 자극(좌우교체/perturbation/새쌍) 5개 모델 평가. 전부 hex 제거 프롬프트, 추론만
# (재학습 없음). "원본 c" 조건(c_nohex, 원본이미지+hex제거)도 같은 체크포인트 세트로 같이 돌려서
# 신규 자극과 나란히 비교할 수 있게 한다 — llava/vila_3b는 이미 있는 c_nohex 결과를 캐시로
# 재사용(파일 존재시 스킵), mipha/vila_8b/openflamingo는 이번에 처음 c_nohex를 돈다.
#
# 로딩/디코딩은 inf_llava.py/inf_mipha.py/inf_vila.py/inf_openflamingo.py 기존 클래스 재사용
# (run_grayctrl.py 와 동일 패턴) — 원본 파이프라인 변경 없음.
#
# 실행 (extract_5_tasks/ 안에서, 모델별 conda env):
#   python s1_rebuttal/run_s1_newstim.py --model llava

import os
import sys
import json
import argparse

import torch

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXTRACT5)
sys.path.insert(0, EXTRACT5)

REBUTTAL_S1_DIR = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants"
NEWSTIM_DIR = os.path.join(EXTRACT5, "s1_rebuttal", "newstim", "data")

VARIANT_FILES = {
    "c_nohex": os.path.join(REBUTTAL_S1_DIR, "dataset_c.json"),
    "swap": os.path.join(NEWSTIM_DIR, "dataset_swap.json"),
    "perturb": os.path.join(NEWSTIM_DIR, "dataset_perturb.json"),
    "newpair": os.path.join(NEWSTIM_DIR, "dataset_newpair.json"),
}

OUT_ROOT = os.path.join(EXTRACT5, "result_s1_rebuttal")

CKPTS = {
    "llava": {
        "Base":   "/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b",
        "S1_fwd": "/home/hail/vlm3/checkpoint/llava/S1_Similarity/end",
        "S1_rev": "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity/end",
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/llava/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity_nohex/end",
    },
    "mipha": {
        "Base":   "/home/hail/vlm3/Model/llava-phi/Mipha-3B-phi2",
        "S1_fwd": "/home/hail/vlm3/checkpoint/Mipha/S1_Similarity-phi2/end",
        "S1_rev": "/home/hail/vlm3/checkpoint/Mipha_reverse/S1_Similarity-phi2/end",
    },
    "vila_3b": {
        "Base":   "/home/hail/vlm3/Model/VILA/VILA1.5-3b",
        "S1_fwd": "/home/hail/vlm3/checkpoint/VILA_3B/S1_Similarity/end",
        "S1_rev": "/home/hail/vlm3/checkpoint/VILA_3B_reverse/S1_Similarity/end",
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/VILA_3B/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/VILA_3B_reverse/S1_Similarity_nohex/end",
    },
    "vila_8b": {
        "Base":   "/home/hail/vlm3/Model/VILA/Llama-3-VILA1.5-8B",
        "S1_fwd": "/home/hail/vlm3/checkpoint/VILA_8B/S1_Similarity/end",
        "S1_rev": "/home/hail/vlm3/checkpoint/VILA_8B_reverse/S1_Similarity/end",
    },
    "openflamingo": {
        "Base":   "/home/hail/vlm3/Model/open_flamingo/ckpts/OpenFlamingo-3B-vitl-mpt1b/checkpoint.pt",
        "S1_fwd": "/home/hail/vlm3/checkpoint/OpenFlamingo/S1_Similarity",
        "S1_rev": "/home/hail/vlm3/checkpoint/OpenFlamingo_reverse/S1_Similarity",
    },
}

TEMPERATURE = 0.0


def load_variant(path):
    return json.load(open(path))


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
        pred, _feat, _proj = loader.run_one(image, question, [0], TEMPERATURE)  # layer 추출 불필요, 최소 1개만
    return pred


def run(model, variants=None, ckpt_names=None):
    ckpts = CKPTS[model]
    if ckpt_names:
        ckpts = {k: v for k, v in ckpts.items() if k in ckpt_names}
    variant_items = ({k: VARIANT_FILES[k] for k in variants} if variants else VARIANT_FILES)

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

        for variant_name, fname in variant_items.items():
            samples = load_variant(fname)
            out_dir = os.path.join(OUT_ROOT, model, ckpt_name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{variant_name}.json")
            if os.path.exists(out_path):
                print(f"  [cached] {variant_name}")
                continue

            records = []
            for s in samples:
                pred = run_one_generic(model, loader, s["image"], s["question"])
                rec = {"id": s["id"], "output": pred, "sim_img": s["sim_img"]}
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
