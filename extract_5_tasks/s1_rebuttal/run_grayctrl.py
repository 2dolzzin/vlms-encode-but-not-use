# run_grayctrl.py
# Experiment A (회색 이미지 control): 5개 모델 x {Base, Final_fwd(S4_end), Final_rev(S0_end)}
# = 15개 체크포인트에 대해, 5개 stage 각각 원본 이미지 vs 회색(128,128,128) 이미지로 추론.
# + LLaVA/VILA-3B의 S1 nohex 체크포인트 4개도 S1 stage만 같은 control.
#
# 디코딩/로딩은 inf_llava.py/inf_mipha.py/inf_vila.py/inf_openflamingo.py 의 기존 클래스를
# 그대로 재사용 (원본 파이프라인 변경 없음).
#
# 실행 (extract_5_tasks/ 안에서, 모델별 conda env):
#   python s1_rebuttal/run_grayctrl.py --model llava --stages S0,S1,S2,S3,S4

import os
import sys
import json
import argparse

import torch

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(EXTRACT5)
sys.path.insert(0, EXTRACT5)

REBUTTAL_S1_DIR = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants"

STAGES = {
    "S0": {
        "orig": "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
        "gray": "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/rebuttal_gray/dataset_gray.json",
    },
    "S1": {
        "orig": "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/dataset.json",
        "gray": os.path.join(REBUTTAL_S1_DIR, "dataset_a.json"),
    },
    "S2": {
        "orig": "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
        "gray": "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/rebuttal_gray/dataset_gray.json",
    },
    "S3": {
        "orig": "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
        "gray": "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/rebuttal_gray/dataset_gray.json",
    },
    "S4": {
        "orig": "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
        "gray": "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/rebuttal_gray/dataset_gray.json",
    },
}

LAYERS = [0, 8, 16, 24, 32]
TEMPERATURE = 0.0
OUT_ROOT = os.path.join(EXTRACT5, "result_grayctrl")

CKPTS = {
    "llava": {
        "Base": "/home/hail/vlm3/Model/LLaVA/llava-v1.5-7b",
        "Final_fwd": "/home/hail/vlm3/checkpoint/llava/S4_Depth/end",
        "Final_rev": "/home/hail/vlm3/checkpoint/llava_reverse/S0_LogMAR/end",
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/llava/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/llava_reverse/S1_Similarity_nohex/end",
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
        "S1_fwd_nohex": "/home/hail/vlm3/checkpoint/VILA_3B/S1_Similarity_nohex/end",
        "S1_rev_nohex": "/home/hail/vlm3/checkpoint/VILA_3B_reverse/S1_Similarity_nohex/end",
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

# S1 stage에만 적용할 nohex 체크포인트 (llava, vila_3b 만 존재)
NOHEX_ONLY_S1 = {"S1_fwd_nohex", "S1_rev_nohex"}


def load_stage_samples(path):
    cases = json.load(open(path))
    out = []
    for c in cases:
        if "conversations" in c:
            out.append({"id": str(c.get("id", "")), "image": c["image"],
                       "question": c["conversations"][0]["value"],
                       "answer": c["conversations"][1]["value"]})
        else:
            out.append({"id": str(c.get("id", "")), "image": c["image"],
                       "question": c["question"],
                       "answer": c.get("answer", c.get("sim_img"))})
    return out


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
        pred, _feat, _proj = loader.run_one(image, question, LAYERS, TEMPERATURE)
    return pred


def run(model, stages, ckpt_names=None):
    ckpts = CKPTS[model]
    if ckpt_names:
        ckpts = {k: v for k, v in ckpts.items() if k in ckpt_names}

    of_loader = None  # openflamingo: 모델 1회 빌드, weight만 교체
    if model == "openflamingo":
        import inf_openflamingo
        of_loader = inf_openflamingo.OFInferExtract()

    for ckpt_name, ckpt_path in ckpts.items():
        stage_list = [s for s in stages if not (ckpt_name in NOHEX_ONLY_S1 and s != "S1")]
        print(f"\n########## {model} / {ckpt_name} ##########  stages={stage_list}")

        if model == "openflamingo":
            of_loader.load_ckpt(ckpt_path)
            loader = of_loader
        else:
            loader = build_loader(model, ckpt_path)

        for stage in stage_list:
            for cond in ["orig", "gray"]:
                samples = load_stage_samples(STAGES[stage][cond])
                out_dir = os.path.join(OUT_ROOT, model, ckpt_name)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{stage}_{cond}.json")
                if os.path.exists(out_path):
                    print(f"  [cached] {stage}/{cond}")
                    continue

                records = []
                for s in samples:
                    pred = run_one_generic(model, loader, s["image"], s["question"])
                    records.append({"id": s["id"], "output": pred, "expected_output": s["answer"]})
                json.dump(records, open(out_path, "w"), indent=2, ensure_ascii=False)
                print(f"  [saved] {stage}/{cond} n={len(records)} → {out_path}")

        if model != "openflamingo":
            del loader
            torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CKPTS))
    ap.add_argument("--stages", default="S0,S1,S2,S3,S4")
    ap.add_argument("--ckpts", default=None, help="쉼표구분, 미지정시 전체")
    a = ap.parse_args()
    run(a.model, a.stages.split(","), a.ckpts.split(",") if a.ckpts else None)
