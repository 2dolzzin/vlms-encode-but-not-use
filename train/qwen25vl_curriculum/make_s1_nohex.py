#!/usr/bin/env python
# make_s1_nohex.py
# S1 Color Similarity 의 "검증셋"과 "5-task 추론셋" 에서 프롬프트의 hex 코드 구절만 제거한
# nohex 버전을 새로 만든다. 원본 파일은 절대 건드리지 않고 새 경로에만 쓴다.
#
# 학습셋 nohex 는 이미 있음:
#   /home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_non_flamingo_nohex.json
#   (train/s1_nohex_retrain/make_s1_prompts_nohex.py 가 생성)
# 이 스크립트는 그와 "동일한 정규식/정리 규칙" 을 써서 검증셋·추론셋도 같은 형태로 맞춘다.
#
# 규칙 (기존 make_s1_prompts_nohex.py 와 동일):
#   "(Left : #2800ff, Right : #0028ff)" 같은 괄호 구절만 삭제.
#   "People described pairs of colors using their hex codes." 문장은 그대로 둔다
#   — 실제 hex 값이 없으므로 정답 누출이 없고, 기존 nohex 학습셋도 이 문장을 유지한다.
#   이미지와 정답(GT)은 손대지 않는다.
#
# 실행: python make_s1_nohex.py           # 생성
#       python make_s1_nohex.py --check   # 생성 안 하고 현재 상태만 점검

import os
import re
import json
import argparse

HEX_PAREN_RE = re.compile(
    r'\(Left\s*:\s*#([0-9a-fA-F]{6})\s*,\s*Right\s*:\s*#([0-9a-fA-F]{6})\)'
)
ANY_HEX_RE = re.compile(r'#[0-9a-fA-F]{6}')

JOBS = [
    # (설명, 원본, 출력)
    ("validation",
     "/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/color_similarity_non_flamingo.json",
     "/home/hail/HDD/vlm_validation/S1_Color_Similarity/dataset/rebuttal_nohex/color_similarity_non_flamingo_nohex.json"),
    ("inference(5-task)",
     "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/dataset.json",
     "/home/hail/HDD/vlm_inference/Task_5_stage/S1_Color_Similarity/rebuttal_nohex/dataset_nohex.json"),
]

TRAIN_NOHEX = "/home/hail/HDD/vlm_training/S1_Color_Similarity/dataset/rebuttal_nohex/similarity_non_flamingo_nohex.json"


def strip_hex(question):
    q = HEX_PAREN_RE.sub("", question)
    q = re.sub(r"\s+\.", ".", q)
    q = re.sub(r"\s{2,}", " ", q).strip()
    return q


def count_hex(cases):
    """프롬프트에 실제 hex 값(#rrggbb)이 남아 있는 샘플 수."""
    return sum(1 for c in cases if ANY_HEX_RE.search(c["conversations"][0]["value"]))


def convert(tag, src, out, write=True):
    cases = json.load(open(src))
    before_hex = count_hex(cases)
    for c in cases:
        c["conversations"][0]["value"] = strip_hex(c["conversations"][0]["value"])
    after_hex = count_hex(cases)

    print(f"\n[{tag}] n={len(cases)}  hex 포함 샘플: {before_hex} → {after_hex}")
    print(f"  src = {src}")
    print(f"  out = {out}")
    print(f"  예시 Q: {cases[0]['conversations'][0]['value'][:150]}")
    print(f"  예시 A: {cases[0]['conversations'][1]['value']}")
    if after_hex:
        raise SystemExit(f"[ERR] {tag}: hex 가 {after_hex}개 남았다. 정규식 확인 필요.")

    if write:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(cases, open(out, "w"), indent=2, ensure_ascii=False)
        print(f"  [saved] {out}")
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="파일을 쓰지 않고 점검만")
    a = ap.parse_args()

    for tag, src, out in JOBS:
        if not a.check and os.path.exists(out):
            cases = json.load(open(out))
            print(f"\n[{tag}] 이미 있음 (n={len(cases)}, 남은 hex={count_hex(cases)}) → {out}")
            continue
        convert(tag, src, out, write=not a.check)

    if os.path.exists(TRAIN_NOHEX):
        tr = json.load(open(TRAIN_NOHEX))
        print(f"\n[training] 기존 nohex 학습셋 n={len(tr)}  남은 hex={count_hex(tr)}")
        print(f"  예시 Q: {tr[0]['conversations'][0]['value'][:150]}")


if __name__ == "__main__":
    main()
