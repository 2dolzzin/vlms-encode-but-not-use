#!/usr/bin/env python3
"""results_<model>/<variant>__<tag>.json → eval_loss_table.csv (model,metric,baseline,s0_mid..s4_end; 열은 스테이지 이름 기준).
variant: fwd/rev → model 키 '<model>' / '<model>_reverse', 무작위 런 → run 키 그대로. make_tables.py 가 열 순서를 학습 순서로 재배열한다."""
import os, glob, json, csv
HERE = os.path.dirname(os.path.abspath(__file__))
RD = {"llava": "results_llava", "mipha": "results_mipha", "vila_3b": "results_vila_3b", "vila_8b": "results_vila_8b",
      "openflamingo": "results_openflamingo", "qwen25vl_7b": "results_qwen25vl_7b"}
COLS = ["baseline"] + [f"s{i}_{m}" for i in range(5) for m in ("mid", "end")]
rows = []
for model, d in RD.items():
    by_key = {}
    for f in sorted(glob.glob(os.path.join(HERE, d, "*__*.json"))):
        variant, tag = os.path.basename(f)[:-5].split("__", 1)
        key = {"fwd": model, "rev": model + "_reverse"}.get(variant, variant)
        by_key.setdefault(key, {})[tag] = json.load(open(f))["losses"]
    for key, tags in by_key.items():
        for i in range(5):
            r = {"model": key, "metric": f"S{i}_loss"}
            for c in COLS:
                tag = "start" if c == "baseline" else f"S{c[1]}_{c[3:]}"
                v = tags.get(tag, {}).get(f"S{i}")
                r[c] = f"{v:.4f}" if v is not None else ""
            rows.append(r)
out = os.path.join(HERE, "eval_loss_table.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model", "metric"] + COLS); w.writeheader(); w.writerows(rows)
keys = sorted({r["model"] for r in rows}); print(f"{out}: {len(rows)} rows, {len(keys)} run keys: {keys}")
