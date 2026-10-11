# make_s1_prompts_nohex.py
# S1 학습 데이터(similarity_non_flamingo.json, 50000개)에서 프롬프트의 hex 코드 구절만
# 제거한 재학습용 데이터를 만든다. 원본 파일은 건드리지 않음. 이미지/정답(GT)은 그대로.
#
# 실행: python make_s1_prompts_nohex.py

import re
import json

SRC = "/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/similarity_non_flamingo.json"
OUT = "/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_non_flamingo_nohex.json"

HEX_PAREN_RE = re.compile(
    r'\(Left\s*:\s*#([0-9a-fA-F]{6})\s*,\s*Right\s*:\s*#([0-9a-fA-F]{6})\)'
)


def strip_hex(question):
    q = HEX_PAREN_RE.sub("", question)
    q = re.sub(r"\s+\.", ".", q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def main():
    cases = json.load(open(SRC))
    no_match = 0
    for c in cases:
        q = c["conversations"][0]["value"]
        if not HEX_PAREN_RE.search(q):
            no_match += 1
        c["conversations"][0]["value"] = strip_hex(q)

    print(f"[make_s1_prompts_nohex] n={len(cases)}  hex 패턴 없던 샘플={no_match}")
    json.dump(cases, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {OUT}")

    print("\n--- 샘플 10개 (변형 전 → 후) ---")
    import random
    random.seed(0)
    orig = json.load(open(SRC))
    for c in random.sample(orig, 10):
        before = c["conversations"][0]["value"]
        after = strip_hex(before)
        print(f"id={c['id']}")
        print(f"  before: {before}")
        print(f"  after : {after}")


if __name__ == "__main__":
    main()
