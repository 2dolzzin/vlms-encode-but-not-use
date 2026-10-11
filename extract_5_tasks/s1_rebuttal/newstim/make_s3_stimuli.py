# make_s3_stimuli.py
# S3 칩 perturbation 자극 생성 (원본 330칩 이미지/데이터/파이프라인은 건드리지 않음, 새 폴더에만 저장)
#
# 확인된 사실(사전에 원본 이미지를 직접 열어 검증): S3 이미지는 "15개 참조색 + 대상칩" 합성이
# 아니라, 대상 칩 1개만 담긴 336x336 단색 정사각형이다. 15개 참조 색 이름(Red/Blue/... /Silver)은
# origin.json의 프롬프트 텍스트로만 주어지고 이미지에는 전혀 그려지지 않는다. 즉 "참조 15색은
# 건드리지 않는다"는 원 스펙 조건은 애초에 이미지에 참조색 픽셀이 없어 자동으로 충족된다 —
# 이 정정은 결과 보고 시 함께 전달한다.
#
# perturbation: RGB 채널별 동일방향 ±8 (worst case로 렌더). 카테고리 경계 근접 칩은 축소/제외.
#   기준: "다른 카테고리 중 최근접 칩까지의 CIEDE2000 거리"(nearest_other_cat_de2000, 이미
#   s3_boundary_distances.json 에 330개 전부 계산되어 있음) 를 실측한 ±8/±4 perturbation
#   자체의 CIEDE2000 크기(각각 ~2.3~3.0 / ~1.1~1.5, n=30 표본)와 비교해 정함:
#     - dist <  2.0  : 제외 (축소해도 경계를 넘나들 위험)
#     - 2.0<=dist<4.0: ±4로 축소 (완충 마진 확보)
#     - dist >= 4.0  : 원래 ±8
#   결과: 제외 17개 / ±4 51개 / ±8 262개 (s3_perturbation_plan.json에 이미 저장됨).
#   RNG는 필요없음(perturbation 방향은 항상 원 hex를 밝은 방향으로 +8/+4, 결정적) — 재현성 자명.

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_common import render_chip, hex_to_rgb, rgb_to_hex, clamp255

IMAGES_ROOT = "/home/hail/HDD/vlm_inference/Task_5_stage/S3_Color_Naming/images"
BOUNDARY_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_boundary_distances.json")

OUT_ROOT = "/home/hail/vlm3/extract_5_tasks/s1_rebuttal/newstim/data"
IMG_S3 = os.path.join(OUT_ROOT, "images_s3_perturb")

TH_EXCLUDE = 2.0
TH_REDUCE = 4.0
FULL_DELTA = 8
REDUCED_DELTA = 4

# S3 원본 origin.json 프롬프트 그대로 재사용 (build.py 에서 확인한 고정 문구)
ORIGIN_PROMPT = ("<image> What color is the 16th color in the image? "
                  "The first 15 colors, from left to right, are : Red, Blue, Yellow, Green, Orange, "
                  "Purple, Pink, Brown, Black, White, Gray, Beige, Turquoise, Gold, Silver. "
                  "Choose the most similar color from this list.")


def perturb_hex(h, delta):
    r, g, b = hex_to_rgb(h)
    r, g, b = clamp255(r + delta), clamp255(g + delta), clamp255(b + delta)
    return rgb_to_hex((r, g, b))


def main():
    os.makedirs(IMG_S3, exist_ok=True)
    boundary = json.load(open(BOUNDARY_JSON))["chips"]

    out = []
    n_excl = n_reduced = n_full = 0
    for c in boundary:
        cat, h, dist = c["category"], c["hex"], c["nearest_other_cat_de2000"]
        if dist < TH_EXCLUDE:
            n_excl += 1
            continue
        delta = REDUCED_DELTA if dist < TH_REDUCE else FULL_DELTA
        if delta == REDUCED_DELTA:
            n_reduced += 1
        else:
            n_full += 1
        h_pert = perturb_hex(h, delta)
        fname = f"{h}_pert{delta}.png"
        out_path = os.path.join(IMG_S3, cat, fname)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        render_chip(h_pert).save(out_path)
        out.append({
            "id": f"s3pert_{cat}_{h}",
            "image": out_path,
            "question": ORIGIN_PROMPT,
            "answer": cat,
            "hex_orig": h, "hex_pert": h_pert, "delta": delta,
            "nearest_other_cat_de2000": dist,
        })

    json.dump(out, open(os.path.join(OUT_ROOT, "dataset_s3_perturb.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[s3 perturb] 제외 {n_excl}개 / ±{REDUCED_DELTA} {n_reduced}개 / ±{FULL_DELTA} {n_full}개 "
          f"(총 생성 {len(out)}/330) -> dataset_s3_perturb.json")


if __name__ == "__main__":
    main()
