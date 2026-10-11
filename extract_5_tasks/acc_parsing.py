# eval_raw.py
# raw_responses/{ckpt}/{task}.json 의 모델 출력(output)을 파싱해서
# 정답(expected_output)과 비교 → accuracy + (macro) F1 계산.
#
# 파서/지표 컨벤션은 /home/hail/vlm2/evaluation (parsers.py, metrics/*) 참고.
# raw_response 형식이 모든 ckpt에서 동일하므로 이 스크립트 하나로 전부 처리.
#
# task별 처리:
#   logmar      : 6글자 파싱 → 글자단위 정확도 + 완전일치 정확도 + 글자단위 macro F1(A-Z)
#   similarity  : float 파싱 → 회귀라 acc/F1 대신 MAE/RMSE/Pearson/Spearman/R²
#   ishihara    : 숫자 파싱 → accuracy + macro F1 (0-9)
#   naming      : 색이름 파싱 → accuracy + macro F1 (15색, 대소문자 무시)
#   depth_mcq   : A-D 파싱 → accuracy + macro F1 (A/B/C/D)
#
# 실행:
#   python eval_raw.py                    # 모든 ckpt × 모든 task
#   python eval_raw.py --ckpt S0_end      # 특정 ckpt
#   python eval_raw.py --task S3_naming   # 특정 task

import os
import re
import glob
import json
import argparse
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr, spearmanr

MODEL = "openflamingo_reverse"   # Change by "CLI --model"
RAW_DIR = f"/home/hail/vlm3/extract_5_tasks/result/{MODEL}/inference/raw_responses"
ACC_DIR = f"/home/hail/vlm3/extract_5_tasks/result/{MODEL}/inference/accuracy"
PARSE_DIR = f"/home/hail/vlm3/extract_5_tasks/result/{MODEL}/inference/parsing"


def set_model(model):
    global MODEL, RAW_DIR, ACC_DIR, PARSE_DIR
    MODEL = model
    RAW_DIR = f"/home/hail/vlm3/extract_5_tasks/result/{model}/inference/raw_responses"
    ACC_DIR = f"/home/hail/vlm3/extract_5_tasks/result/{model}/inference/accuracy"
    PARSE_DIR = f"/home/hail/vlm3/extract_5_tasks/result/{model}/inference/parsing"

COLOR_NAMES_15 = ["Red", "Blue", "Yellow", "Green", "Orange", "Purple", "Pink",
                  "Brown", "Black", "White", "Gray", "Beige", "Turquoise",
                  "Gold", "Silver"]


# ============================================================
# 파서 (vlm2/evaluation/parsers.py 기반)
# ============================================================
def parse_letters(text, n=6):
    if not text:
        return ""
    text = text.strip().upper()
    m = re.search(r'\b([A-Z])[\s,]+([A-Z])[\s,]+([A-Z])[\s,]+([A-Z])[\s,]+([A-Z])[\s,]+([A-Z])\b', text)
    if m:
        return "".join(m.groups())
    matches = re.findall(r'\b[A-Z]{%d,}\b' % n, text)
    if matches:
        return matches[-1][:n]
    letters = re.findall(r'[A-Z]', text)
    return "".join(letters[:n])


def parse_similarity(text):
    if not text:
        return None
    text = text.strip()
    for m in re.findall(r'0?\.\d+|1\.0{0,2}', text):
        v = float(m)
        if 0 <= v <= 1:
            return v
    if text in ("0", "1"):
        return float(text)
    pct = re.search(r'(\d{1,3})\s*%', text)
    if pct:
        v = float(pct.group(1)) / 100.0
        if 0 <= v <= 1:
            return v
    frac = re.search(r'([\d.]+)\s*(?:/|out of)\s*1', text)
    if frac:
        try:
            v = float(frac.group(1))
            if 0 <= v <= 1:
                return v
        except ValueError:
            pass
    # 답변 선언형 정수: "The answer is 0.", "similarity is 1" 처럼 소수점 없이
    # 0/1 로만 답한 경우(문장 끝 마침표 허용). 뒤에 숫자·'/'가 오면 소수/분수라
    # 제외(0.86, 1/2 등은 위에서 이미 처리되거나 오매칭이므로 배제).
    m = re.search(r'\bis\s*[:=]?\s*([01])(?![\d/])', text, re.I)
    if m:
        return float(m.group(1))
    return None


def parse_digit(text):
    m = re.findall(r'\d', (text or "").strip())
    return m[0] if m else "-1"


def parse_color(text):
    tl = (text or "").lower().strip()
    for c in COLOR_NAMES_15:
        if c.lower() in tl:
            return c.lower()
    return "unknown"


def parse_mcq(text):
    if not text:
        return None
    m = re.search(r'\b([A-D])\b', text, re.I)
    if not m:
        m = re.search(r'([A-D])', text, re.I)
    return m.group(1).upper() if m else None


# ============================================================
# task 종류 판별
# ============================================================
def task_kind(task_name):
    t = task_name.lower()
    if "logmar" in t or "logamr" in t:
        return "logmar"
    if "similar" in t:
        return "similarity"
    if "ishihara" in t:
        return "ishihara"
    if "naming" in t:
        return "naming"
    if "depth" in t:
        return "depth_mcq"
    return "unknown"


# ============================================================
# 분류 지표: accuracy + macro F1
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

    details = []
    summary = {"ckpt": ckpt, "task": task, "kind": kind, "n": len(recs)}

    if kind == "similarity":
        # 회귀: float 파싱 → MSE/MAE/RMSE/Pearson/Pearson²/Spearman/R²
        pv, gv, parsed = [], [], 0
        for r in recs:
            p = parse_similarity(r["output"])
            g = float(r["expected_output"])
            ok = p is not None
            parsed += ok
            if ok:
                pv.append(p); gv.append(g)
            details.append({"id": r["id"], "pred": p, "gt": g})
        pv, gv = np.array(pv), np.array(gv)
        if len(pv) == 0:
            summary.update({
                "parsed": 0, "parse_ratio": 0.0,
                "mse": float('nan'), "mae": float('nan'), "rmse": float('nan'),
                "r2": float('nan'), "pearson": float('nan'), "spearman": float('nan'),
                "accuracy": None, "f1": None,
            })
            print(f"  [{ckpt}/{task}] (reg) ALL PARSE FAILED ({len(recs)} samples)")
        else:
            ss_res = np.sum((gv - pv) ** 2)
            ss_tot = np.sum((gv - gv.mean()) ** 2)
            summary.update({
                "parsed": parsed, "parse_ratio": parsed / len(recs),
                "mse": float(np.mean((pv - gv) ** 2)),
                "mae": float(np.mean(np.abs(pv - gv))),
                "rmse": float(np.sqrt(np.mean((pv - gv) ** 2))),
                "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan'),
                "pearson": float(pearsonr(gv, pv)[0]),
                "pearson_sq": float(pearsonr(gv, pv)[0] ** 2),
                "spearman": float(spearmanr(gv, pv)[0]),
                "accuracy": None, "f1": None,
            })
            print(f"  [{ckpt}/{task}] (reg) MSE={summary['mse']:.4f} Pearson={summary['pearson']:.4f} "
                f"(MAE={summary['mae']:.4f} R2={summary['r2']:.4f} Pearson²={summary['pearson_sq']:.4f} "
                f"Spearman={summary['spearman']:.4f}) parse={parsed}/{len(recs)}")

    elif kind == "logmar":
        # 글자단위 정확도 + 완전일치 + 글자단위 macro F1
        cl = tl = exact = 0
        char_pred, char_gt = [], []
        for r in recs:
            pred = parse_letters(r["output"])
            gt = r["expected_output"].strip().upper()
            if pred == gt:
                exact += 1
            for i in range(len(gt)):
                p = pred[i] if i < len(pred) else "<none>"
                char_pred.append(p); char_gt.append(gt[i])
                if p == gt[i]:
                    cl += 1
                tl += 1
            details.append({"id": r["id"], "pred": pred, "gt": gt, "exact": pred == gt})
        labels = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        f1 = float(f1_score(char_gt, char_pred, labels=labels, average="macro", zero_division=0))
        summary.update({
            "accuracy": cl / tl if tl else 0,        # 글자단위 정확도
            "exact_match": exact / len(recs),         # 6글자 완전일치
            "f1": f1,                                 # 글자단위 macro F1
            "correct_letters": cl, "total_letters": tl,
        })
        print(f"  [{ckpt}/{task}] charAcc={summary['accuracy']:.4f} "
              f"exact={summary['exact_match']:.4f} F1={f1:.4f}")

    else:
        # 분류 (ishihara / naming / depth_mcq)
        if kind == "ishihara":
            parse_fn = parse_digit
            norm_gt = lambda g: str(g).strip()
            labels = [str(d) for d in range(10)]
        elif kind == "naming":
            parse_fn = parse_color
            norm_gt = lambda g: str(g).strip().lower()
            labels = [c.lower() for c in COLOR_NAMES_15]
        elif kind == "depth_mcq":
            parse_fn = parse_mcq
            norm_gt = lambda g: str(g).strip().upper()
            labels = ["A", "B", "C", "D"]
        else:
            print(f"  [skip] {ckpt}/{task}: unknown kind")
            return None

        preds, gts = [], []
        for r in recs:
            p = parse_fn(r["output"])
            g = norm_gt(r["expected_output"])
            preds.append(p); gts.append(g)
            details.append({"id": r["id"], "pred": p, "gt": g, "correct": p == g})
        # '유효 파싱' = 실제 클래스(labels)로 답한 것.
        # parse_color 는 실패 시 "unknown", parse_digit 는 "-1" 을 돌려주므로
        # None 체크로는 붕괴를 못 잡는다 → labels 소속 여부로 판정한다.
        label_set = set(labels)
        parsed = sum(p in label_set for p in preds)
        if parsed == 0:
            # 형식 전멸(유효 클래스 0개) = 출력 붕괴 → 회귀(similarity)와 대칭으로 NaN 처리.
            # (0으로 두면 '전부 틀림'과 '형식 붕괴'가 구분 안 되므로 빈칸으로 뚫는다)
            acc, f1 = float('nan'), float('nan')
            print(f"  [{ckpt}/{task}] ALL PARSE FAILED (n={len(recs)}) → NaN")
        else:
            acc, f1 = clf_metrics(preds, gts, labels)
            print(f"  [{ckpt}/{task}] acc={acc:.4f} F1={f1:.4f} parse={parsed}/{len(recs)}")
        summary.update({"accuracy": acc, "f1": f1,
                        "parsed": parsed, "parse_ratio": parsed / len(recs) if recs else 0.0,
                        "correct": int(sum(p == g for p, g in zip(preds, gts)))})

    # 저장
    os.makedirs(ACC_DIR, exist_ok=True)
    os.makedirs(PARSE_DIR, exist_ok=True)
    json.dump(summary, open(os.path.join(ACC_DIR, f"{ckpt}__{task}.json"), "w"), indent=2)
    json.dump(details, open(os.path.join(PARSE_DIR, f"{ckpt}__{task}.json"), "w"), indent=2)
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
    print(f"처리할 json: {len(paths)}개")
    rows = []
    for p in paths:
        s = eval_one(p)
        if s:
            rows.append(s)

    # 전체 요약 테이블 저장
    os.makedirs(ACC_DIR, exist_ok=True)
    json.dump(rows, open(os.path.join(ACC_DIR, "_summary_all.json"), "w"), indent=2)
    print(f"\n전체 요약 → {os.path.join(ACC_DIR, '_summary_all.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--task", type=str, default=None)
    ap.add_argument("--model", type=str, default=MODEL)
    a = ap.parse_args()
    set_model(a.model)
    run(ckpt=a.ckpt, task=a.task)
