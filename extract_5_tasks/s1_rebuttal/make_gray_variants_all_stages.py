# make_gray_variants_all_stages.py
# Experiment A (회색 이미지 control): S0/S2/S3/S4 검증셋의 이미지를 균일 회색(128,128,128)으로
# 바꾼 변형 데이터셋을 만든다. 프롬프트/정답은 원본 그대로. S1은 기존
# rebuttal_variants/dataset_a.json(hex 포함 프롬프트 + gray 이미지)을 그대로 재사용.
#
# 원본 Task_5_stage 데이터/이미지는 절대 건드리지 않음.

import os
import json
from PIL import Image

STAGES = {
    "S0_Logmar": "/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json",
    "S2_Ishihara": "/home/hail/HDD/vlm_inference/Task_5_stage/S2_Ishihara/dataset.json",
    "S3_Color_Naming": "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/dataset.json",
    "S4_Depth": "/home/hail/HDD/vlm_inference/Task_5_stage/S4_Depth/dataset.json",
}
OUT_ROOT_TMPL = "/home/hail/HDD/vlm_inference/Task_5_stage/{stage}/rebuttal_gray"


def main():
    for stage, path in STAGES.items():
        cases = json.load(open(path))
        im0 = Image.open(cases[0]["image"])
        size, mode = im0.size, im0.mode

        out_root = OUT_ROOT_TMPL.format(stage=stage)
        os.makedirs(os.path.join(out_root, "images"), exist_ok=True)
        gray_path = os.path.join(out_root, "images", "gray_128.png")

        if mode == "RGBA":
            gray = Image.new("RGBA", size, (128, 128, 128, 255))
        else:
            gray = Image.new("RGB", size, (128, 128, 128))
        gray.save(gray_path)

        out = []
        for c in cases:
            out.append({
                "id": c["id"],
                "image": gray_path,
                "question": c["conversations"][0]["value"],
                "answer": c["conversations"][1]["value"],
            })

        out_path = os.path.join(out_root, "dataset_gray.json")
        json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"[{stage}] N={len(out)} size={size} mode={mode} → {out_path}")


if __name__ == "__main__":
    main()
