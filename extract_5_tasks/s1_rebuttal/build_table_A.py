# build_table_A.py
# 표 A: 회색 이미지 control, rebuttal 표 형식. 기존 result_grayctrl/ 결과만 사용 (재추론 없음 —
# 모든 셀이 이미 존재함, 재확인 결과 "빠진 칸"은 없었다).
#
# 행=모델(5개), 열=stage(S0,S2,S3,S4)x{원본,회색}. Base/Final_fwd(=fwd 최종, S4 stage end) 두 표.
# S1은 제외(표 1에서 별도 처리). Final_rev(=rev 최종)는 별도 파일로만 저장, rebuttal 표엔 미포함.
# S4는 Base-회색 vs Final_fwd-회색 비교 표를 추가로 만든다(언어 prior가 학습으로 생겼는지 원래
#있었는지 판단용).

import os
import sys
import json

EXTRACT5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EXTRACT5)
from acc_parsing import parse_letters, parse_digit, parse_color, parse_mcq  # noqa: E402

RESULT_ROOT = os.path.join(EXTRACT5, "result_grayctrl")
OUT_DIR = os.path.join(EXTRACT5, "s1_rebuttal", "tables")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["llava", "mipha", "vila_3b", "vila_8b", "openflamingo"]
MODEL_LABEL = {"llava": "LLaVA", "mipha": "Mipha", "vila_3b": "VILA-3B",
               "vila_8b": "VILA-8B", "openflamingo": "OpenFlamingo"}
STAGES = ["S0", "S2", "S3", "S4"]


def s1_chance_unused():
    pass


def s0_chance():
    d = json.load(open("/home/hail/HDD/vlm_inference/Task_5_stage/S0_Logmar/dataset.json"))
    # 글자단위 accuracy의 chance = 1/26 (문자셋 26개 균등 가정, score_grayctrl.py 와 동일 정의)
    return 1 / 26


CHANCE = {"S0": s0_chance(), "S2": 0.10, "S3": 1 / 15, "S4": 0.25}


def score_stage(stage, records):
    if stage == "S0":
        cl = tl = 0
        for r in records:
            pred = parse_letters(r["output"])
            gt = r["expected_output"].strip().upper()
            for i in range(len(gt)):
                p = pred[i] if i < len(pred) else "<none>"
                tl += 1
                cl += (p == gt[i])
        return cl / tl if tl else float("nan")
    if stage == "S2":
        correct = total = 0
        for r in records:
            p = parse_digit(r["output"])
            g = str(r["expected_output"]).strip()
            correct += (p == g); total += 1
        return correct / total if total else float("nan")
    if stage == "S3":
        correct = total = 0
        for r in records:
            p = parse_color(r["output"])
            g = str(r["expected_output"]).strip().lower()
            correct += (p == g); total += 1
        return correct / total if total else float("nan")
    if stage == "S4":
        correct = total = 0
        for r in records:
            p = parse_mcq(r["output"])
            g = str(r["expected_output"]).strip().upper()
            correct += (p == g); total += 1
        return correct / total if total else float("nan")
    raise ValueError(stage)


def load_score(model, ckpt, stage, cond):
    p = os.path.join(RESULT_ROOT, model, ckpt, f"{stage}_{cond}.json")
    if not os.path.exists(p):
        return None
    return score_stage(stage, json.load(open(p)))


def build_table(ckpt_key, ckpt_title):
    header = "| model | " + " | ".join(
        f"{s} (orig/gray, chance={CHANCE[s]:.3f})" for s in STAGES) + " |"
    sep = "|" + "---|" * (len(STAGES) + 1)
    lines = [f"### 표 A — 회색 이미지 control, {ckpt_title}", "", header, sep]

    for m in MODELS:
        row = [MODEL_LABEL[m]]
        for s in STAGES:
            o = load_score(m, ckpt_key, s, "orig")
            g = load_score(m, ckpt_key, s, "gray")
            if o is None or g is None:
                row.append("-")
            else:
                row.append(f"{o:.3f} / {g:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    text = "\n".join(lines) + "\n"
    print(text)
    return text


def build_s4_gray_delta_table():
    header = "| model | Base 회색 | Final_fwd 회색 | Δ(학습후−Base) |"
    sep = "|---|---|---|---|"
    lines = ["### S4 회색 조건: Base vs 학습 후(fwd 최종) — 언어 prior가 원래 있었는지/학습으로 생겼는지",
             "", header, sep]
    for m in MODELS:
        b = load_score(m, "Base", "S4", "gray")
        f = load_score(m, "Final_fwd", "S4", "gray")
        if b is None or f is None:
            lines.append(f"| {MODEL_LABEL[m]} | - | - | - |")
            continue
        delta = f - b
        lines.append(f"| {MODEL_LABEL[m]} | {b:.3f} | {f:.3f} | {delta:+.3f} |")
    lines.append(f"\n(chance = {CHANCE['S4']:.3f})")
    text = "\n".join(lines) + "\n"
    print(text)
    return text


def build_rev_table():
    """rebuttal 표에는 안 넣지만 보관용으로 별도 저장."""
    return build_table("Final_rev", "rev 최종 (S0_LogMAR end) — 보관용, rebuttal 미포함")


if __name__ == "__main__":
    t_base = build_table("Base", "Base 체크포인트")
    t_fwd = build_table("Final_fwd", "fwd 최종 (S4_Depth end)")
    t_s4delta = build_s4_gray_delta_table()
    t_rev = build_rev_table()

    open(os.path.join(OUT_DIR, "tableA_base.md"), "w").write(t_base)
    open(os.path.join(OUT_DIR, "tableA_fwd_final.md"), "w").write(t_fwd)
    open(os.path.join(OUT_DIR, "tableA_s4_gray_delta.md"), "w").write(t_s4delta)
    open(os.path.join(OUT_DIR, "tableA_rev_final_ARCHIVE.md"), "w").write(t_rev)
    print("[saved] tableA_base.md, tableA_fwd_final.md, tableA_s4_gray_delta.md, tableA_rev_final_ARCHIVE.md")
