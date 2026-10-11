# make_s1_flamingo_nohex.py
# OpenFlamingo 용 S1 nohex 학습 데이터 생성.
# make_s1_prompts_nohex.py (non_flamingo 판) 과 동일한 변환을 flamingo 포맷 파일에 적용한다.
#   - 프롬프트의 "(Left : #xxxxxx, Right : #xxxxxx)" 구절만 제거
#   - 이미지/정답(GT)은 그대로. flamingo 포맷의 짧은 정답("0.86")도 손대지 않는다.
#   - 원본 파일은 건드리지 않고 rebuttal_nohex/ 에 새로 쓴다.
#
# 실행: python make_s1_flamingo_nohex.py

import re
import json
import os

SRC = "/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/similarity_flamingo.json"
OUT = "/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_flamingo_nohex.json"

HEX_PAREN_RE = re.compile(
    r'\(Left\s*:\s*#([0-9a-fA-F]{6})\s*,\s*Right\s*:\s*#([0-9a-fA-F]{6})\)'
)


def strip_hex(question):
    q = HEX_PAREN_RE.sub("", question)
    q = re.sub(r"\s+\.", ".", q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def main():
    assert not os.path.exists(OUT), f"이미 존재함: {OUT}"
    cases = json.load(open(SRC))
    no_match = 0
    for c in cases:
        q = c["conversations"][0]["value"]
        if not HEX_PAREN_RE.search(q):
            no_match += 1
        c["conversations"][0]["value"] = strip_hex(q)

    print(f"[make_s1_flamingo_nohex] n={len(cases)}  hex 패턴 없던 샘플={no_match}")
    json.dump(cases, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT}")

    print("\n--- 샘플 3개 (변형 전 → 후) ---")
    import random
    random.seed(0)
    orig = json.load(open(SRC))
    for c in random.sample(orig, 3):
        before = c["conversations"][0]["value"]
        print(f"id={c['id']}  gpt={c['conversations'][1]['value']!r}")
        print(f"  before: {before}")
        print(f"  after : {strip_hex(before)}")


if __name__ == "__main__":
    main()
