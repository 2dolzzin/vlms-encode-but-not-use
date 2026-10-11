# analyze_gray_outputs.py
# 표 A 각주용: 회색 이미지 조건에서 실제 raw 출력이 뭘 뱉는지 집계 (재추론 없음, 기존
# result_grayctrl/*/*/{S3,S2,S0}_gray.json 만 사용).
#   S3: parse_color 적용 후 top-3 색 이름과 비율 (Gray 비율이 핵심 관심사)
#   S2: parse_digit 적용 후 분포, 상수면 어느 숫자인지
#   S0: 전부 동일 문자열인지(문자 단위 아니라 전체 output 문자열 기준 상수 여부)

import os
import sys
import json
from collections import Counter

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_color, parse_digit  # noqa: E402

RESULT_ROOT = os.path.join(EXTRACT5, "result_grayctrl")
MODELS = ["llava", "mipha", "vila_3b", "vila_8b", "openflamingo"]
CKPTS = ["Base", "Final_fwd", "Final_rev"]


def analyze_s3(model, ckpt):
    p = os.path.join(RESULT_ROOT, model, ckpt, "S3_gray.json")
    if not os.path.exists(p):
        return None
    recs = json.load(open(p))
    cnt = Counter(parse_color(r["output"]) for r in recs)
    total = sum(cnt.values())
    top3 = cnt.most_common(3)
    gray_ratio = cnt.get("gray", 0) / total if total else 0.0
    return {"top3": [(name, n / total) for name, n in top3], "gray_ratio": gray_ratio, "n": total}


def analyze_s2(model, ckpt):
    p = os.path.join(RESULT_ROOT, model, ckpt, "S2_gray.json")
    if not os.path.exists(p):
        return None
    recs = json.load(open(p))
    cnt = Counter(parse_digit(r["output"]) for r in recs)
    total = sum(cnt.values())
    top3 = cnt.most_common(3)
    is_const = len(cnt) == 1
    return {"top3": [(d, n / total) for d, n in top3], "is_const": is_const, "n": total}


def analyze_s0(model, ckpt):
    p = os.path.join(RESULT_ROOT, model, ckpt, "S0_gray.json")
    if not os.path.exists(p):
        return None
    recs = json.load(open(p))
    outs = [r["output"].strip() for r in recs]
    cnt = Counter(outs)
    total = len(outs)
    is_const = len(cnt) == 1
    top1 = cnt.most_common(1)[0] if cnt else (None, 0)
    return {"is_const": is_const, "n_unique": len(cnt), "top1": (top1[0], top1[1] / total), "n": total}


def main():
    lines = ["## 회색(gray) 조건 raw 출력 분포 — 표 A 각주용\n"]

    lines.append("### S3 (색 이름 응답, parse_color 적용)\n")
    lines.append("| model | ckpt | n | top-3 (비율) | gray 비율 |")
    lines.append("|---|---|---|---|---|")
    for m in MODELS:
        for ck in CKPTS:
            r = analyze_s3(m, ck)
            if r is None:
                continue
            top3s = ", ".join(f"{n}({p:.2f})" for n, p in r["top3"])
            lines.append(f"| {m} | {ck} | {r['n']} | {top3s} | {r['gray_ratio']:.3f} |")

    lines.append("\n### S2 (숫자 응답, parse_digit 적용)\n")
    lines.append("| model | ckpt | n | top-3 (비율) | 상수? |")
    lines.append("|---|---|---|---|---|")
    for m in MODELS:
        for ck in CKPTS:
            r = analyze_s2(m, ck)
            if r is None:
                continue
            top3s = ", ".join(f"{d}({p:.2f})" for d, p in r["top3"])
            lines.append(f"| {m} | {ck} | {r['n']} | {top3s} | {'예' if r['is_const'] else '아니오'} |")

    lines.append("\n### S0 (원시 출력 문자열 기준 상수 여부)\n")
    lines.append("| model | ckpt | n | 고유 출력 수 | 상수? | 최빈 출력(비율) |")
    lines.append("|---|---|---|---|---|---|")
    for m in MODELS:
        for ck in CKPTS:
            r = analyze_s0(m, ck)
            if r is None:
                continue
            top1_str = f"{r['top1'][0]!r}({r['top1'][1]:.2f})" if r['top1'][0] is not None else "-"
            lines.append(f"| {m} | {ck} | {r['n']} | {r['n_unique']} | {'예' if r['is_const'] else '아니오'} | {top1_str} |")

    text = "\n".join(lines) + "\n"
    print(text)
    out = os.path.join(EXTRACT5, "s1_rebuttal", "tables", "tableA_gray_output_distribution.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(text)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
