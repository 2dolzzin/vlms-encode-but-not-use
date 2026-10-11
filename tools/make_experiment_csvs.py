#!/usr/bin/env python3
"""
실험별 결과 CSV (행 = 모델[×variant(×layer)], 열 = 학습 시점 전부: baseline + 5단계×{mid,end}).

입력  : csv_out/inf_*_<variant>.csv, csv_out/probing_*_<variant>.csv (export_csv.py 산출물),
        train/eval_loss/eval_loss_table.csv
출력  : csv_out/experiments/<exp>_<...>.csv           열 = 학습 '순서' 기준 (t1_mid … t5_end), order 열에 실제 스테이지 순서
        csv_out/experiments/by_stage/<exp>_<...>.csv  열 = 스테이지 이름 기준 (S0_mid … S4_end)
  exp1_evalloss_<Sk>.csv            Exp1 망각: 검증셋 Sk 의 eval loss (낮을수록↓)
  exp1_inference_<task>_<metric>.csv Exp1 5-task 추론 (S1 은 mae↓, 나머지 accuracy↑)
  exp2_probe_{mlp,linear,logreg}_<task>.csv  Exp2 5-task 프로빙 acc (layer 행)
  exp2_probe_ridge_S1_similarity.csv         Exp2 S1 ridge 회귀 mae↓
  exp3_inference_<task>[_pair_accuracy].csv  Exp3 blind 8 + line_plot + mmvp(question/pair)
  exp4_probe_{mlp,linear,logreg}_<task>.csv  Exp4 벤치마크 프로빙 acc (layer 행)
사용: python tools/make_experiment_csvs.py [--no-refresh]   (--no-refresh: export_csv.py 재실행 생략)
"""
import os, sys, csv, re, glob, subprocess
from collections import OrderedDict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (tools/ 의 상위)
CSV_OUT = os.path.join(BASE, "csv_out")
OUT = os.path.join(CSV_OUT, "experiments")
OUT_STAGE = os.path.join(OUT, "by_stage")

MODEL_ORDER = ["llava", "mipha", "vila_3b", "vila_8b", "openflamingo", "qwen25vl_7b"]
STAGE_COLS = [f"S{k}_{p}" for k in range(5) for p in ("mid", "end")]
POS_COLS = [f"t{k}_{p}" for k in range(1, 6) for p in ("mid", "end")]


def layer_key(l):
    return (1, 0) if l == "proj" else (0, int(l.split("_")[1]))


def variant_label(suffix):
    return suffix if suffix in ("forward", "reverse") else "random_" + suffix.split("_", 1)[1]


def variant_sort(label):
    return {"forward": 0, "reverse": 1}.get(label, 2), label


def read_variant_csv(path):
    """-> (id_cols, train_order [stage names in 학습 순서], rows=[(id_dict, {stage_col: val})])"""
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        n_id = header.index("Base")
        id_cols, ck_cols = header[:n_id], header[n_id:]
        order = [c.split("_")[0] for c in ck_cols[1::2]]        # Base, S1_mid, S1_end, ... -> [S1, ...]
        rows = []
        for line in r:
            ids = dict(zip(id_cols, line[:n_id]))
            vals = dict(zip(ck_cols, line[n_id:]))
            rows.append((ids, vals))
    return id_cols, order, rows


def stage_to_pos(vals, order):
    """{Base, S*_mid/end} -> {baseline, t*_mid/end} (학습 순서 위치로 재배열)"""
    out = {"baseline": vals.get("Base", "")}
    for i, s in enumerate(order, 1):
        for p in ("mid", "end"):
            out[f"t{i}_{p}"] = vals.get(f"{s}_{p}", "")
    return out


class Table:
    """한 실험×지표 파일. row key = (model, variant[, layer])"""
    def __init__(self, name, with_layer=False):
        self.name, self.with_layer, self.rows = name, with_layer, []

    def add(self, model, variant, order, vals, layer=None):
        if not any(v != "" for v in vals.values()):
            return
        self.rows.append((model, variant, order, layer, vals))

    def write(self):
        if not self.rows:
            return 0
        self.rows.sort(key=lambda r: (MODEL_ORDER.index(r[0]) if r[0] in MODEL_ORDER else 99,
                                      variant_sort(r[1]), layer_key(r[3]) if r[3] else (0, 0)))
        idc = ["model", "variant", "order"] + (["layer"] if self.with_layer else [])
        for out_dir, cols, conv in ((OUT, POS_COLS, True), (OUT_STAGE, STAGE_COLS, False)):
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, self.name + ".csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(idc + ["baseline"] + cols)
                for model, variant, order, layer, vals in self.rows:
                    v = stage_to_pos(vals, order) if conv else {"baseline": vals.get("Base", ""), **vals}
                    ids = [model, variant, "→".join(order)] + ([layer] if self.with_layer else [])
                    w.writerow(ids + [v.get("baseline", "")] + [v.get(c, "") for c in cols])
        return len(self.rows)


def variants_present(prefix):
    """csv_out/<prefix>_<variant>.csv 들 -> [(variant_suffix, path)]"""
    out = []
    for p in sorted(glob.glob(os.path.join(CSV_OUT, prefix + "_*.csv"))):
        suf = os.path.basename(p)[len(prefix) + 1:-4]
        if suf in ("forward", "reverse") or suf.startswith("order_"):
            out.append((suf, p))
    return out


def build_inference(prefix, exp, tables, extra_metric_in_name=False):
    for suf, path in variants_present(prefix):
        _, order, rows = read_variant_csv(path)
        for ids, vals in rows:
            task, metric = ids["task"], ids["metric"]
            name = f"{exp}_inference_{task}"
            if extra_metric_in_name or metric != "accuracy":
                name += f"_{metric}"
            tables.setdefault(name, Table(name)).add(ids["model"], variant_label(suf), order, vals)


def build_probing(prefix, exp, probe, tables):
    for suf, path in variants_present(prefix):
        _, order, rows = read_variant_csv(path)
        for ids, vals in rows:
            task, metric, layer = ids["task"], ids["metric"], ids["layer"]
            if metric == "acc":
                name = f"{exp}_probe_{probe}_{task}"
            else:                                   # ridge 회귀(S1, mae)
                name = f"{exp}_probe_ridge_{task}"
            tables.setdefault(name, Table(name, with_layer=True)).add(
                ids["model"], variant_label(suf), order, vals, layer=layer)


def build_evalloss(tables, orders):
    path = os.path.join(BASE, "train/eval_loss/eval_loss_table.csv")
    if not os.path.exists(path):
        print("[warn] eval_loss_table.csv 없음"); return
    with open(path) as f:
        for r in csv.DictReader(f):
            m = re.match(r"^(.*?)(?:_(reverse)|_order_(\d+))?$", r["model"])
            model, rev, tag = m.group(1), m.group(2), m.group(3)
            variant = "reverse" if rev else ("random_" + tag if tag else "forward")
            order = orders.get(variant)
            if order is None:
                print(f"[warn] eval loss {r['model']}: 학습 순서 정보 없음, 생략"); continue
            vals = {"Base": r["baseline"]}
            for c in STAGE_COLS:
                vals[c] = r[c.lower()]
            name = f"exp1_evalloss_{r['metric'].replace('_loss', '')}"
            tables.setdefault(name, Table(name)).add(model, variant, order, vals)


def main():
    if "--no-refresh" not in sys.argv:
        print("[refresh] export_csv.py 재실행 …")
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_csv.py")],
                       check=True, stdout=open(os.path.join(CSV_OUT, "_export.log"), "w"), stderr=subprocess.STDOUT)
    tables = OrderedDict()
    orders = {variant_label(suf): read_variant_csv(p)[1] for suf, p in variants_present("inf_curriculum")}

    build_evalloss(tables, orders)
    build_inference("inf_curriculum", "exp1", tables)
    build_probing("probing_curriculum", "exp2", "mlp", tables)
    build_probing("probing_curriculum_linear", "exp2", "linear", tables)
    build_probing("probing_curriculum_logreg", "exp2", "logreg", tables)
    build_inference("inf_benchmark", "exp3", tables)
    build_probing("probing_benchmark", "exp4", "mlp", tables)
    build_probing("probing_benchmark_linear", "exp4", "linear", tables)
    build_probing("probing_benchmark_logreg", "exp4", "logreg", tables)

    index = ["# csv_out/experiments 색인", "",
             "- 열 `baseline,t1_mid,…,t5_end` = 그 런의 **학습 순서** 기준 시점. `order` 열이 실제 스테이지 순서(예 `S1→S4→S2→S0→S3`).",
             "- `by_stage/` 는 같은 내용을 **스테이지 이름** 열(`S0_mid…S4_end`)로 배치한 것.",
             "- variant: forward / reverse / random_14203 / random_20431 / random_31024. 빈 칸 = 결과 없음(파싱 전멸 NaN 포함).",
             "- S1_similarity 추론·ridge 는 mae(낮을수록 좋음), 그 외는 accuracy/acc(높을수록 좋음).",
             "- fwd/rev 의 S1 추론은 hex 프롬프트, random/Qwen 은 nohex → S1 열 직접 비교 금지.", "",
             "| 파일 | 행 수 |", "|---|---|"]
    total = 0
    for name, t in tables.items():
        n = t.write()
        if n:
            index.append(f"| {name}.csv | {n} |"); total += n
    with open(os.path.join(OUT, "_index.md"), "w") as f:
        f.write("\n".join(index) + "\n")
    print(f"DONE -> {os.path.relpath(OUT, BASE)}  ({sum(1 for t in tables.values() if t.rows)} files, {total} rows)")


if __name__ == "__main__":
    main()
