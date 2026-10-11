# make_s1_stimuli.py
# S1 신규 자극 3종 생성 (원본 78쌍 이미지/데이터/파이프라인은 전혀 건드리지 않음, 전부 이 아래
# newstim/data/ 밑 새 폴더에만 저장). 프롬프트는 dataset_c.json(원본+hex 제거)에서 실제 쓰인
# 고정 문구를 그대로 재사용 — 78개 전부 동일한 1개 문장이라 조회는 불필요, 상수로 박아둔다.
#
#  a) swap    : 78쌍 각각 좌우 패치 위치 교체. GT(similarity) 동일.
#  b) perturb : 각 패치 RGB 채널별 ±10 무작위(seed 고정). GT 원본 유지.
#  c) newpair : Ekman 14색(고유 hex 13개) 밖의 새 sRGB 색으로 55쌍.
#               원본 Ekman 색이 전부 순수 스펙트럼색(S=1.0, V=1.0)이므로 신규 색도 S=1.0/V=1.0
#               유지, hue만 새로 샘플링(기존 hue와 최소 8도 이격, seed 고정 거절샘플링).
#               GT = CIEDE2000 proxy (color_common.ciede2000_sim, D_REF=흑백 CIEDE2000=100).
#               주의: 절대값이 아니라 proxy이므로 평가 시 Spearman만 사용(스펙 상 지시).

import os
import json
import random
import colorsys
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from color_common import render_pair, hex_to_rgb, rgb_to_hex, clamp255, ciede2000_sim

LABELS = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/images/labels.json"
DATASET_C = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants/dataset_c.json"
EKMAN_COLORS_CSV = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/어디서왔는지/ekman_colors.csv"

OUT_ROOT = "/home/hail/vlm3/extract_5_tasks/s1_rebuttal/newstim/data"
IMG_SWAP = os.path.join(OUT_ROOT, "images_swap")
IMG_PERTURB = os.path.join(OUT_ROOT, "images_perturb")
IMG_NEWPAIR = os.path.join(OUT_ROOT, "images_newpair")

SEED = 0
PERTURB_RANGE = 10
N_NEW_COLORS = 16
N_NEW_PAIRS = 55
MIN_HUE_GAP_DEG = 8.0

# dataset_c.json 78개 전부 동일 — hex 제거된 고정 프롬프트 (question 필드에서 확인, "<image> " 접두 포함)
NOHEX_QUESTION = ("<image> People described pairs of colors using their hex codes. "
                   "The similarity between two colors is measured on a scale from 0 to 1, "
                   "where 0 means the colors are completely different, and 1 means the colors "
                   "are exactly the same. Respond the similarity rating between the colors in the image.")


def make_swap(labels):
    os.makedirs(IMG_SWAP, exist_ok=True)
    out = []
    for lb in labels:
        base = os.path.basename(lb["image_path"])
        h1, h2 = lb["hex_code_1"], lb["hex_code_2"]
        sim = lb["similarity"]
        swapped_name = f"swap_{base}"
        im = render_pair(h2, h1)  # 좌우 교체: 왼쪽=h2, 오른쪽=h1
        out_path = os.path.join(IMG_SWAP, swapped_name)
        im.save(out_path)
        out.append({
            "id": f"swap_{lb['index']}",
            "image": out_path,
            "question": NOHEX_QUESTION,
            "sim_img": f"{sim:.2f}",
            "orig_image_basename": base,
            "hex_left": h2, "hex_right": h1,
        })
    json.dump(out, open(os.path.join(OUT_ROOT, "dataset_swap.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[swap] {len(out)}개 생성 -> dataset_swap.json")
    return out


def make_perturb(labels):
    os.makedirs(IMG_PERTURB, exist_ok=True)
    rng = random.Random(SEED)
    out = []
    for lb in labels:
        base = os.path.basename(lb["image_path"])
        h1, h2 = lb["hex_code_1"], lb["hex_code_2"]
        sim = lb["similarity"]

        def perturb_hex(h):
            r, g, b = hex_to_rgb(h)
            r = clamp255(r + rng.randint(-PERTURB_RANGE, PERTURB_RANGE))
            g = clamp255(g + rng.randint(-PERTURB_RANGE, PERTURB_RANGE))
            b = clamp255(b + rng.randint(-PERTURB_RANGE, PERTURB_RANGE))
            return rgb_to_hex((r, g, b))

        h1p, h2p = perturb_hex(h1), perturb_hex(h2)
        pname = f"pert_{base}"
        out_path = os.path.join(IMG_PERTURB, pname)
        render_pair(h1p, h2p).save(out_path)
        out.append({
            "id": f"pert_{lb['index']}",
            "image": out_path,
            "question": NOHEX_QUESTION,
            "sim_img": f"{sim:.2f}",
            "orig_image_basename": base,
            "hex_left_orig": h1, "hex_right_orig": h2,
            "hex_left_pert": h1p, "hex_right_pert": h2p,
        })
    json.dump(out, open(os.path.join(OUT_ROOT, "dataset_perturb.json"), "w"), indent=2, ensure_ascii=False)
    print(f"[perturb] {len(out)}개 생성 (±{PERTURB_RANGE} RGB, seed={SEED}) -> dataset_perturb.json")
    return out


def sample_new_hues(existing_hexes, n, min_angle_deg, seed):
    existing_h = []
    for h in existing_hexes:
        r, g, b = [c / 255 for c in hex_to_rgb(h)]
        hh, _, _ = colorsys.rgb_to_hsv(r, g, b)
        existing_h.append(hh * 360)

    rng = random.Random(seed)
    chosen = []
    attempts = 0
    while len(chosen) < n and attempts < 200000:
        attempts += 1
        cand = rng.uniform(0, 360)
        if any(min(abs(cand - e), 360 - abs(cand - e)) < min_angle_deg for e in existing_h + chosen):
            continue
        chosen.append(cand)
    if len(chosen) < n:
        raise RuntimeError(f"hue 샘플링 실패: {len(chosen)}/{n} (min_angle_deg={min_angle_deg} 너무 큼)")
    return [rgb_to_hex(tuple(c * 255 for c in colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0))) for hue in chosen]


def make_newpair():
    os.makedirs(IMG_NEWPAIR, exist_ok=True)
    ekman = pd.read_csv(EKMAN_COLORS_CSV)
    existing_hexes = ekman["hex"].tolist()  # 14개 (고유 13개, ff0000 중복)

    new_hexes = sample_new_hues(existing_hexes, N_NEW_COLORS, MIN_HUE_GAP_DEG, SEED)

    rng = random.Random(SEED + 1)
    all_pairs = [(i, j) for i in range(len(new_hexes)) for j in range(i + 1, len(new_hexes))]
    rng.shuffle(all_pairs)
    chosen_pairs = all_pairs[:N_NEW_PAIRS]

    out = []
    for k, (i, j) in enumerate(chosen_pairs):
        h1, h2 = new_hexes[i], new_hexes[j]
        proxy_sim = ciede2000_sim(h1, h2)
        fname = f"new_{h1}_{h2}.png"
        out_path = os.path.join(IMG_NEWPAIR, fname)
        render_pair(h1, h2).save(out_path)
        out.append({
            "id": f"newpair_{k:03d}",
            "image": out_path,
            "question": NOHEX_QUESTION,
            "sim_img": f"{proxy_sim:.4f}",  # proxy GT — 절대값 신뢰 불가, Spearman 전용
            "hex_left": h1, "hex_right": h2,
            "gt_type": "ciede2000_proxy",
        })
    json.dump(out, open(os.path.join(OUT_ROOT, "dataset_newpair.json"), "w"), indent=2, ensure_ascii=False)
    json.dump({"new_colors": new_hexes},
              open(os.path.join(OUT_ROOT, "newpair_colors.json"), "w"), indent=2)
    print(f"[newpair] 신규색 {len(new_hexes)}개 -> 쌍 {len(out)}개 생성 -> dataset_newpair.json")
    return out


if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)
    labels = json.load(open(LABELS))
    make_swap(labels)
    make_perturb(labels)
    make_newpair()
