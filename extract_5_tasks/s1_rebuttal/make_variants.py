# make_variants.py
# S1(Color Similarity) 검증셋(78쌍)에서 hex-code shortcut 검증용 변형 데이터셋을 만든다.
# 원본 dataset.json/images 는 절대 수정하지 않고, rebuttal_variants/ 밑에 새로 생성한다.
#
# 만드는 것:
#   orig   : 원본 그대로 (비교 기준)
#   a      : 이미지 → 균일 회색(128,128,128), 프롬프트(hex 포함) 그대로       [text-only 상한]
#   c      : 이미지 원본 유지, 프롬프트에서 hex 구절만 제거
#   d      : 이미지 → 회색, 프롬프트에서 hex 구절도 제거                      [정보 0 바닥선]
#   b      : 이미지 원본 유지, 프롬프트의 hex만 다른 샘플 것으로 교체 (mismatch)
#
# 실행: python make_variants.py

import os
import re
import json

SRC_JSON = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/dataset.json"
OUT_ROOT = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants"
MISMATCH_MIN_GAP = 0.3

HEX_PAREN_RE = re.compile(
    r'\(Left\s*:\s*#([0-9a-fA-F]{6})\s*,\s*Right\s*:\s*#([0-9a-fA-F]{6})\)'
)


def load_cases():
    with open(SRC_JSON) as f:
        return json.load(f)


def make_gray_image():
    from PIL import Image
    img_dir = os.path.join(OUT_ROOT, "images")
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(img_dir, "gray_128.png")
    Image.new("RGB", (224, 224), (128, 128, 128)).save(path)
    return path


def strip_hex(question):
    q = HEX_PAREN_RE.sub("", question)
    q = re.sub(r"\s+\.", ".", q)   # "image ()." → "image ()" 제거 후 남는 " ." 정리
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def swap_hex(question, new_left, new_right):
    return HEX_PAREN_RE.sub(f"(Left : #{new_left}, Right : #{new_right})", question, count=1)


def build_mismatch_pairs(cases):
    """각 샘플 i 에 대해 |sim_i - sim_j| 가 가장 큰(>= MISMATCH_MIN_GAP) j 를 찾는다."""
    sims = [float(c["conversations"][1]["value"]) for c in cases]
    n = len(cases)
    pairs = {}
    unmatched = []
    for i in range(n):
        best_j, best_gap = None, -1.0
        for j in range(n):
            if j == i:
                continue
            gap = abs(sims[i] - sims[j])
            if gap > best_gap:
                best_gap, best_j = gap, j
        if best_gap >= MISMATCH_MIN_GAP:
            pairs[i] = best_j
        else:
            unmatched.append((i, best_j, best_gap))
    return pairs, unmatched


def main():
    cases = load_cases()
    n = len(cases)
    gray_path = make_gray_image()
    print(f"[gray image] {gray_path}")

    variants = {"orig": [], "a": [], "c": [], "d": []}
    for c in cases:
        q = c["conversations"][0]["value"]
        img = c["image"]
        ans = c["conversations"][1]["value"]
        base = {"id": c["id"]}

        variants["orig"].append({**base, "image": img, "question": q, "sim_img": ans})
        variants["a"].append({**base, "image": gray_path, "question": q, "sim_img": ans,
                               "note": "image replaced with uniform gray(128,128,128)"})
        variants["c"].append({**base, "image": img, "question": strip_hex(q), "sim_img": ans})
        variants["d"].append({**base, "image": gray_path, "question": strip_hex(q), "sim_img": ans,
                               "note": "image replaced with uniform gray(128,128,128)"})

    pairs, unmatched = build_mismatch_pairs(cases)
    print(f"[mismatch] matched {len(pairs)}/{n} (gap>={MISMATCH_MIN_GAP}), unmatched={len(unmatched)}")
    for i, j, gap in unmatched:
        print(f"  [warn] id={cases[i]['id']} 최대 gap={gap:.3f} < {MISMATCH_MIN_GAP} (best candidate id={cases[j]['id'] if j is not None else None})")

    variant_b = []
    for i, c in enumerate(cases):
        if i not in pairs:
            continue
        j = pairs[i]
        q = c["conversations"][0]["value"]
        m = HEX_PAREN_RE.search(cases[j]["conversations"][0]["value"])
        new_left, new_right = m.group(1), m.group(2)
        q_swapped = swap_hex(q, new_left, new_right)
        variant_b.append({
            "id": c["id"],
            "image": c["image"],
            "question": q_swapped,
            "sim_img": c["conversations"][1]["value"],
            "sim_text": cases[j]["conversations"][1]["value"],
            "text_source_id": cases[j]["id"],
            "gap": abs(float(c["conversations"][1]["value"]) - float(cases[j]["conversations"][1]["value"])),
        })

    os.makedirs(OUT_ROOT, exist_ok=True)
    for name, data in variants.items():
        path = os.path.join(OUT_ROOT, f"dataset_{name}.json")
        json.dump(data, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"[saved] {path}  n={len(data)}")

    path_b = os.path.join(OUT_ROOT, "dataset_b.json")
    json.dump(variant_b, open(path_b, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {path_b}  n={len(variant_b)}")

    # 눈으로 확인용 샘플 출력
    print("\n--- sample check ---")
    print("orig[0] :", variants["orig"][0]["question"])
    print("c[0]    :", variants["c"][0]["question"])
    print("b[0]    :", variant_b[0]["question"], "| sim_img=", variant_b[0]["sim_img"],
          " sim_text=", variant_b[0]["sim_text"], " gap=", round(variant_b[0]["gap"], 3))


if __name__ == "__main__":
    main()
