# make_variants_v2.py
# (b) mismatch 재설계: 이전 greedy(최대 gap) 매칭은 sim_img/sim_text 사이에 강한 음의
# 공선성을 만들어 회귀/follow-rate를 둘 다 무효화시켰다 (corr 확인 결과 반영).
# 대신 무작위 순열 + rejection sampling 으로 sim_img·sim_text 를 독립에 가깝게 만든다.
#
# 규칙:
#   - 78개 이미지 각각에 대해 hex 텍스트 파트너를 뽑는다 (자기 자신 제외).
#   - |sim_img - sim_text| >= GAP_MIN(0.2) 만족할 때까지 무작위 재시도(rejection).
#   - 이미지당 최대 MAX_PARTNERS(2)개 파트너 부여 (가능하면 하나는 sim_text > sim_img,
#     하나는 sim_text < sim_img 가 되도록 시도 — 그래도 실패하면 있는 것만 채택).
#   - seed 고정, 재현 가능.
#
# 실행: python make_variants_v2.py

import os
import re
import json
import random

SRC_JSON = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/dataset.json"
OUT_ROOT = "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_variants"
GAP_MIN = 0.2
MAX_PARTNERS = 2
SEED = 0
MAX_TRIES_PER_SLOT = 200

HEX_PAREN_RE = re.compile(
    r'\(Left\s*:\s*#([0-9a-fA-F]{6})\s*,\s*Right\s*:\s*#([0-9a-fA-F]{6})\)'
)


def load_cases():
    with open(SRC_JSON) as f:
        return json.load(f)


def swap_hex(question, new_left, new_right):
    return HEX_PAREN_RE.sub(f"(Left : #{new_left}, Right : #{new_right})", question, count=1)


def build_pairs(cases, rng):
    n = len(cases)
    sims = [float(c["conversations"][1]["value"]) for c in cases]
    out = []
    fail_count = 0

    for i in range(n):
        wanted_signs = [+1, -1] if MAX_PARTNERS >= 2 else [0]
        used_j = set()
        got = 0
        for want in wanted_signs[:MAX_PARTNERS]:
            found = None
            for _try in range(MAX_TRIES_PER_SLOT):
                j = rng.randrange(n)
                if j == i or j in used_j:
                    continue
                gap = sims[j] - sims[i]
                if abs(gap) < GAP_MIN:
                    continue
                if want == +1 and gap <= 0:
                    continue
                if want == -1 and gap >= 0:
                    continue
                found = j
                break
            if found is None:
                # 부호 제약 없이 gap 조건만으로 재시도
                for _try in range(MAX_TRIES_PER_SLOT):
                    j = rng.randrange(n)
                    if j == i or j in used_j:
                        continue
                    if abs(sims[j] - sims[i]) >= GAP_MIN:
                        found = j
                        break
            if found is None:
                fail_count += 1
                continue
            used_j.add(found)
            got += 1

            c, cj = cases[i], cases[found]
            q = c["conversations"][0]["value"]
            m = HEX_PAREN_RE.search(cj["conversations"][0]["value"])
            new_left, new_right = m.group(1), m.group(2)
            out.append({
                "id": f"{c['id']}_x{cj['id']}",
                "image": c["image"],
                "question": swap_hex(q, new_left, new_right),
                "sim_img": c["conversations"][1]["value"],
                "sim_text": cj["conversations"][1]["value"],
                "text_source_id": cj["id"],
                "gap": abs(sims[i] - sims[found]),
            })
    return out, fail_count


def main():
    rng = random.Random(SEED)
    cases = load_cases()
    pairs, fail_count = build_pairs(cases, rng)

    sim_img = [float(p["sim_img"]) for p in pairs]
    sim_text = [float(p["sim_text"]) for p in pairs]
    from scipy.stats import pearsonr
    r = pearsonr(sim_img, sim_text)[0]
    lt = sum(1 for a, b in zip(sim_img, sim_text) if a < b) / len(pairs)

    print(f"[b_v2] n_pairs={len(pairs)}  fail_slots={fail_count}")
    print(f"[b_v2] corr(sim_img, sim_text) = {r:.4f}")
    print(f"[b_v2] sim_img < sim_text 비율 = {lt:.3f} (참고용)")

    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, "dataset_b_v2.json")
    json.dump(pairs, open(path, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {path}  n={len(pairs)}")

    if abs(r) > 0.3:
        print(f"[WARN] |corr|={abs(r):.3f} > 0.3 — 재설계 필요 (seed/GAP_MIN 조정 고려)")


if __name__ == "__main__":
    main()
