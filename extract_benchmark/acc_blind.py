# acc_blind.py
# ben_*_blind.py 가 저장한 raw_responses/{ckpt}/{task}.json 의 모델 출력(output)을
# 파싱해서 정답(expected_output)과 비교 → task별 지표 계산.
# (ben_*_blind.py 는 추론+임베딩만 하고 채점을 안 하므로, 채점은 이 스크립트가 담당)
#
# 구조/컨벤션은 extract_5_tasks/acc_parsing.py 와 동일:
#   result/{model}/raw_responses/{ckpt}/{task}.json  →  읽기
#   result/{model}/accuracy/{ckpt}__{task}.json      →  요약 저장
#   result/{model}/parsing/{ckpt}__{task}.json       →  샘플별 상세 저장
#   result/{model}/accuracy/_summary_all.json        →  전체 요약
#
# blind 9 task 지표 (전부 accuracy 기준 — blind 논문 컨벤션):
#   circled_letter         : {글자} 파싱 → accuracy + macro F1 (A-Z)
#   touching_circles       : Yes/No 파싱 → accuracy + macro F1
#   line/nested/olympic/subway : {정수} 카운트 → accuracy + macro F1
#   counting_grid(blank/word)  : {row},{col} 파싱 → accuracy(완전일치) + 축별 row/col_acc
#   (미파싱 = 오답으로 accuracy 반영. 파싱 전멸 = 형식 붕괴 → 지표 NaN, 0점과 구분)
#
# 실행 (extract_benchmark/ 안에서):
#   python acc_blind.py --model llava_blind                 # 모든 ckpt × task
#   python acc_blind.py --model vila_8b_blind --ckpt S3_end # 특정 ckpt
#   python acc_blind.py --model mipha_blind --task touching_circles

import os
import re
import glob
import json
import argparse
import numpy as np
from sklearn.metrics import f1_score

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL = "llava_blind"
RESULT_ROOT = "result"      # --root: result(blind 파이프라인) / result_origin(구 fwd/rev 벤치 결과)
RAW_DIR = os.path.join(BASE, RESULT_ROOT, MODEL, "raw_responses")
ACC_DIR = os.path.join(BASE, RESULT_ROOT, MODEL, "accuracy")
PARSE_DIR = os.path.join(BASE, RESULT_ROOT, MODEL, "parsing")


def set_model(model, root=None):
    global RESULT_ROOT
    if root:
        RESULT_ROOT = root
    global MODEL, RAW_DIR, ACC_DIR, PARSE_DIR
    MODEL = model
    RAW_DIR = os.path.join(BASE, RESULT_ROOT, model, "raw_responses")
    ACC_DIR = os.path.join(BASE, RESULT_ROOT, model, "accuracy")
    PARSE_DIR = os.path.join(BASE, RESULT_ROOT, model, "parsing")


# ============================================================
# 파서 — blind 프롬프트는 "answer in curly brackets, e.g. {a}" 스타일이라
#        {} 안을 최우선으로 뽑고, 없으면 자유 텍스트에서 폴백 추출.
# ============================================================
def parse_letter(text):
    """circled_letter: 단일 글자 → 대문자. {S} 우선, 없으면 고립된 A-Z."""
    if not text:
        return None
    m = re.findall(r'\{\s*([A-Za-z])\s*\}', text)
    if m:
        return m[-1].upper()
    # 폴백: 고립된 단일 글자 중 마지막 것(답은 보통 문장 끝). "I"/"a" 오매칭 완화.
    m = re.findall(r'\b([A-Za-z])\b', text.strip())
    if m:
        return m[-1].upper()
    return None


def parse_binary(text):
    """touching_circles: yes/no. {yes} 우선, 없으면 단어 등장 순."""
    if not text:
        return None
    t = text.lower()
    m = re.findall(r'\{\s*(yes|no)\s*\}', t)
    if m:
        return m[-1]
    y = re.search(r'\byes\b', t)
    n = re.search(r'\bno\b', t)
    if y and n:
        return "yes" if y.start() < n.start() else "no"   # 먼저 나온 쪽
    if y:
        return "yes"
    if n:
        return "no"
    return None


def parse_count(text):
    """정수 카운트. {n} 우선, 없으면 마지막 정수."""
    if not text:
        return None
    m = re.findall(r'\{\s*(-?\d+)\s*\}', text)
    if m:
        return int(m[-1])
    m = re.findall(r'-?\d+', text)
    if m:
        return int(m[-1])
    return None


def parse_grid(text):
    """counting_grid: (row, col). rows/cols 키워드 → {r},{c} → 앞의 두 정수."""
    if not text:
        return None
    r = re.search(r'row\w*\s*[=:]?\s*\{?\s*(\d+)', text, re.I)
    c = re.search(r'col\w*\s*[=:]?\s*\{?\s*(\d+)', text, re.I)
    if r and c:
        return (int(r.group(1)), int(c.group(1)))
    nums = re.findall(r'\{\s*(\d+)\s*\}', text)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    nums = re.findall(r'\d+', text)
    if len(nums) >= 2:
        return (int(nums[0]), int(nums[1]))
    return None


# ============================================================
# task 종류 판별 (ben_*_blind.py 의 DATASETS 키 기준)
# ============================================================
def task_kind(task_name):
    t = task_name.lower()
    if "mmvp" in t:                       # (a)/(b) 선택. 질문 단위 acc + pair 단위 acc
        return "choice"
    if "letter" in t or "circled" in t:
        return "letter"
    if "grid" in t:                       # counting_grid_blank / counting_grid_word
        return "grid"
    if "touching" in t:
        return "binary"
    # line_plot_intersect / nested_squares / olympic_* / subway_connections
    return "count"


def parse_choice(text, valid="ab"):
    """MMVP: '(a) Open' / 'a' / 'A.' 등에서 선택지 문자. (ben_llava.py 의 parse_choice 와 동일)"""
    if not text:
        return None
    t = text.strip()
    m = re.search(r'[\(\[]?\s*([a-d])\s*[\)\].:]', t, re.I)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    m = re.match(r'\s*([a-d])\b', t, re.I)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    for m in re.finditer(r'\b([a-d])\b', t, re.I):
        if m.group(1).lower() in valid:
            return m.group(1).lower()
    return None


# ============================================================
# 분류 지표: accuracy + macro F1 (파싱 실패는 <none> 로)
# ============================================================
def clf_metrics(preds, gts, labels):
    preds = [p if p is not None else "<none>" for p in preds]
    acc = float(np.mean([p == g for p, g in zip(preds, gts)]))
    f1 = float(f1_score(gts, preds, labels=labels, average="macro", zero_division=0))
    return acc, f1


# ============================================================
# json 1개 평가
# ============================================================
def eval_one(json_path):
    ckpt = os.path.basename(os.path.dirname(json_path))
    task = os.path.basename(json_path).replace(".json", "")
    kind = task_kind(task)
    recs = json.load(open(json_path))
    n = len(recs)

    details = []
    summary = {"ckpt": ckpt, "task": task, "kind": kind, "n": n}

    if kind in ("letter", "binary"):
        # 분류: accuracy + macro F1
        parse_fn = parse_letter if kind == "letter" else parse_binary
        norm_gt = (lambda g: str(g).strip().upper()) if kind == "letter" \
            else (lambda g: str(g).strip().lower())
        preds, gts = [], []
        for r in recs:
            p = parse_fn(r["output"])
            g = norm_gt(r["expected_output"])
            preds.append(p)
            gts.append(g)
            details.append({"id": r["id"], "pred": p, "gt": g, "correct": p == g})
        labels = sorted(set(gts))
        parsed = sum(p is not None for p in preds)
        if parsed == 0:
            acc, f1 = float("nan"), float("nan")
            print(f"  [{ckpt}/{task}] ALL PARSE FAILED (n={n}) → NaN")
        else:
            acc, f1 = clf_metrics(preds, gts, labels)
            print(f"  [{ckpt}/{task}] ({kind}) acc={acc:.4f} F1={f1:.4f} parse={parsed}/{n}")
        summary.update({"accuracy": acc, "f1": f1,
                        "parsed": parsed, "parse_ratio": parsed / n if n else 0.0,
                        "correct": int(sum(p == g for p, g in zip(preds, gts)))})

    elif kind == "count":
        # 카운트: 정수 분류로 보고 accuracy + macro F1 (미파싱=오답)
        preds, gts = [], []
        for r in recs:
            p = parse_count(r["output"])
            g = str(int(str(r["expected_output"]).strip()))
            preds.append(str(p) if p is not None else None)
            gts.append(g)
            details.append({"id": r["id"], "pred": p, "gt": int(g),
                            "correct": (str(p) == g)})
        labels = sorted(set(gts), key=int)
        parsed = sum(p is not None for p in preds)
        if parsed == 0:
            acc, f1 = float("nan"), float("nan")
            print(f"  [{ckpt}/{task}] ALL PARSE FAILED (n={n}) → NaN")
        else:
            acc, f1 = clf_metrics(preds, gts, labels)
            print(f"  [{ckpt}/{task}] (count) acc={acc:.4f} F1={f1:.4f} parse={parsed}/{n}")
        summary.update({"accuracy": acc, "f1": f1,
                        "parsed": parsed, "parse_ratio": parsed / n if n else 0.0,
                        "correct": int(sum(p == g for p, g in zip(preds, gts)))})

    elif kind == "grid":
        # (row,col) 완전일치 accuracy (미파싱=오답). 참고로 축별 정확도도 기록.
        parsed = exact = row_ok = col_ok = 0
        for r in recs:
            pr = parse_grid(r["output"])
            g = tuple(int(x) for x in str(r["expected_output"]).split(","))
            ok = pr is not None
            parsed += ok
            if ok:
                exact += (pr == g)
                row_ok += (pr[0] == g[0])
                col_ok += (pr[1] == g[1])
            details.append({"id": r["id"], "pred": pr, "gt": g,
                            "correct": (ok and pr == g)})
        if parsed == 0:
            summary.update({"parsed": 0, "parse_ratio": 0.0, "accuracy": float("nan"),
                            "row_acc": float("nan"), "col_acc": float("nan")})
            print(f"  [{ckpt}/{task}] ALL PARSE FAILED (n={n}) → NaN")
        else:
            summary.update({
                "parsed": parsed, "parse_ratio": parsed / n if n else 0.0,
                "accuracy": exact / n if n else 0.0,   # (row,col) 완전일치
                "row_acc": row_ok / n if n else 0.0,
                "col_acc": col_ok / n if n else 0.0,
            })
            print(f"  [{ckpt}/{task}] (grid) acc={summary['accuracy']:.4f} "
                  f"(row={summary['row_acc']:.4f} col={summary['col_acc']:.4f}) parse={parsed}/{n}")
    elif kind == "choice":
        # MMVP: 질문 단위 accuracy(N=300, chance 0.5) + pair 단위 accuracy(150쌍 둘 다 맞아야, chance 0.25).
        # 쌍은 id (2k-1, 2k) — 같은 질문을 두 이미지에 묻는 MMVP 구성.
        preds, gts, by_id = [], [], {}
        for r in recs:
            p = parse_choice(r["output"])
            g = str(r["expected_output"]).split(":")[0].strip().lower()
            preds.append(p); gts.append(g); by_id[int(r["id"])] = (p == g)
            details.append({"id": r["id"], "pred": p, "gt": g, "correct": p == g})
        labels = sorted(set(gts))
        parsed = sum(p is not None for p in preds)
        pairs = [(i, i + 1) for i in range(1, max(by_id) + 1, 2) if i in by_id and i + 1 in by_id]
        pair_acc = float(np.mean([by_id[a] and by_id[b] for a, b in pairs])) if pairs else float("nan")
        if parsed == 0:
            acc, f1, pair_acc = float("nan"), float("nan"), float("nan")
            print(f"  [{ckpt}/{task}] ALL PARSE FAILED (n={n}) → NaN")
        else:
            acc, f1 = clf_metrics(preds, gts, labels)
            print(f"  [{ckpt}/{task}] (choice) acc={acc:.4f} pair_acc={pair_acc:.4f} F1={f1:.4f} parse={parsed}/{n}")
        summary.update({"accuracy": acc, "f1": f1, "pair_accuracy": pair_acc, "n_pairs": len(pairs),
                        "parsed": parsed, "parse_ratio": parsed / n if n else 0.0,
                        "correct": int(sum(p == g for p, g in zip(preds, gts)))})
    else:
        print(f"  [skip] {ckpt}/{task}: unknown kind")
        return None

    os.makedirs(ACC_DIR, exist_ok=True)
    os.makedirs(PARSE_DIR, exist_ok=True)
    json.dump(summary, open(os.path.join(ACC_DIR, f"{ckpt}__{task}.json"), "w"), indent=2)
    json.dump(details, open(os.path.join(PARSE_DIR, f"{ckpt}__{task}.json"), "w"),
              indent=2, ensure_ascii=False)
    return summary


# ============================================================
# main
# ============================================================
def run(ckpt=None, task=None):
    pat = os.path.join(RAW_DIR, ckpt or "*", f"{task or '*'}.json")
    paths = sorted(glob.glob(pat))
    if not paths:
        print(f"처리할 json 없음. (pat={pat})")
        return
    print(f"[{MODEL}] 처리할 json: {len(paths)}개")
    rows = []
    for p in paths:
        s = eval_one(p)
        if s:
            rows.append(s)

    os.makedirs(ACC_DIR, exist_ok=True)
    out = os.path.join(ACC_DIR, "_summary_all.json")
    if (ckpt or task) and os.path.exists(out):
        # 일부만 재채점한 경우: 기존 요약을 읽어 같은 (ckpt, task) 행만 교체 — 다른 task 결과를 지우지 않는다
        old = json.load(open(out))
        new_keys = {(r["ckpt"], r["task"]) for r in rows}
        rows = [r for r in old if (r["ckpt"], r["task"]) not in new_keys] + rows
        rows.sort(key=lambda r: (r["ckpt"], r["task"]))
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\n전체 요약 → {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL,
                    help="result/ 밑 폴더명 (예: llava_blind, vila_8b_blind)")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--root", type=str, default=None, choices=["result", "result_origin"],
                    help="결과 루트. 기본 result. 구 fwd/rev 벤치(line_plot/mmvp) 재채점은 result_origin")
    a = ap.parse_args()
    set_model(a.model, a.root)
    run(ckpt=a.ckpt, task=a.task)
