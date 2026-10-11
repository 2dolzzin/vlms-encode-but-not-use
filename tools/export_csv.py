#!/usr/bin/env python3
"""
inf/probing × curriculum/benchmark(blind) 결과를 CSV 로 덤프.
- forward / reverse / 무작위 순서(order_<tag>) 를 각각 별도 파일로
- 가로(열) = 시점(체크포인트, 학습 진행 순), 세로(행) = test(model×task[×layer])
- 값은 RAW metric: accuracy(높을수록↑) / mae(낮을수록↓, S1_similarity 만 해당)
출력: csv_out/
  inf_{curriculum,benchmark}_{forward,reverse,order_<tag>}.csv
  probing_{curriculum,benchmark}_{forward,reverse,order_<tag>}.csv

시점 키는 항상 스테이지 이름(S2_mid/S2_end …)이고, 열 순서만 variant 마다 다르다:
  forward  = S0→S4,  reverse = S4→S0,  order_<tag> = checkpoint/<Model>_order_<tag>_nohex/order.txt 의 순서.
무작위 순서 런의 결과 폴더 이름은 평가 스크립트의 run 키와 같다 (예: result/llava_order_14203, probing/llava_order_14203).
"""
import os, sys, glob, json, math, csv
from collections import defaultdict
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (tools/ 의 상위)
OUT = os.path.join(BASE, "csv_out")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, BASE)
from order_runs import discover_order_runs, read_order, ckpt_order, order_tag  # noqa: E402

MODELS = ["llava", "mipha", "vila_3b", "vila_8b", "openflamingo", "qwen25vl_7b"]     # forward/reverse 표
# 무작위 순서 표: 체크포인트 폴더 접두사 → 결과/프로빙 폴더의 모델 키
ORDER_MODELS = {
    "llava": "llava", "mipha": "Mipha", "vila_3b": "VILA_3B", "vila_8b": "VILA_8B",
    "openflamingo": "OpenFlamingo", "qwen25vl_7b": "Qwen25VL_7B",
}
N_RUNS = 3
# 레이어는 모델마다 다르다(7B계열=0/8/16/24/32, OpenFlamingo=0/6/12/18/24). 고정하지 않고
# 데이터에서 실제 존재하는 레이어를 수집해서 이 순서로 정렬한다.
def layer_sort_key(l):
    if l == "proj":
        return (1, 0)          # proj 는 맨 뒤
    return (0, int(l.split("_")[1]))

# 체크포인트: 저장키는 항상 stage 이름. forward/reverse 는 '학습 진행 순'으로 열 정렬.
FWD_ORDER = ["S0_start", "S0_mid", "S0_end", "S1_mid", "S1_end",
             "S2_mid", "S2_end", "S3_mid", "S3_end", "S4_mid", "S4_end"]
REV_ORDER = ["S0_start", "S4_mid", "S4_end", "S3_mid", "S3_end",
             "S2_mid", "S2_end", "S1_mid", "S1_end", "S0_mid", "S0_end"]


def variants():
    """[(파일접미사, 열 순서, {model: run_name})]. forward/reverse 뒤에 무작위 순서 tag 들을 붙인다.
    무작위 순서는 체크포인트 폴더에서 자동 수집하므로 새 런(예: VILA-8B)이 생기면 그대로 포함된다."""
    out = [("forward", FWD_ORDER, {m: m for m in MODELS}),
           ("reverse", REV_ORDER, {m: m + "_reverse" for m in MODELS})]
    by_tag = {}
    for model, prefix in ORDER_MODELS.items():
        for key, cfg in discover_order_runs(os.path.join(BASE, "checkpoint"), prefix, model).items():
            tag, order = order_tag(cfg["root"]), read_order(cfg["root"])
            if order is None:
                continue
            ent = by_tag.setdefault(tag, {"order": order, "runs": {}})
            if ent["order"] != order:
                print(f"[warn] tag {tag}: {model} 의 order.txt 가 다른 모델과 불일치 {order} vs {ent['order']}")
            ent["runs"][model] = key
    for tag in sorted(by_tag):
        out.append((f"order_{tag}", ckpt_order(by_tag[tag]["order"]), by_tag[tag]["runs"]))
    return out


def col_label(ck):
    return "Base" if ck == "S0_start" else ck


def fmt(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.4f}"


def write_csv(path, id_cols, order, rows):
    """rows: list of (id_tuple, {ckpt: value})."""
    header = list(id_cols) + [col_label(c) for c in order]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for id_tuple, series in rows:
            w.writerow(list(id_tuple) + [fmt(series.get(c)) for c in order])
    print(f"  -> {os.path.relpath(path, BASE)}  ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════
# 1) INFERENCE  (summary_all.json 기반)
# ══════════════════════════════════════════════════════════════
def load_inf_summary(path):
    """{ckpt: {task: row}}"""
    if not os.path.exists(path):
        return {}
    D = {}
    for r in json.load(open(path)):
        D.setdefault(r["ckpt"], {})[r["task"]] = r
    return D


# task -> (raw metric key). S1 만 mae(낮을수록↓), 나머지 accuracy.
CURR_INF_TASKS = [
    ("S0_logamr", "accuracy"), ("S1_similarity", "mae"), ("S2_ishihara", "accuracy"),
    ("S3_naming", "accuracy"), ("S4_depth", "accuracy"),
]
BEN_INF_TASKS = [  # blind : 전부 accuracy
    ("circled_letter", "accuracy"), ("touching_circles", "accuracy"),
    ("nested_squares", "accuracy"), ("olympic_circles", "accuracy"),
    ("olympic_pentagons", "accuracy"), ("subway_connections", "accuracy"),
    ("counting_grid_blank", "accuracy"), ("counting_grid_word", "accuracy"),
]


def dump_inference(name, tasks):
    """tasks: [(task, metric, path_tmpl | [path_tmpl, ...])] — path_tmpl 은 {run} 자리에 결과 폴더 이름.
    리스트면 앞에서부터 그 task 행이 있는 첫 요약을 쓴다 (예: blind 결과에 line_plot/mmvp 가 있으면 그것, 없으면 result_origin)."""
    print(f"[inf:{name}]")
    for suffix, order, runs in variants():
        rows = []
        for model, run in runs.items():
            cache = {}
            for task, metric, path_tmpls in tasks:
                if isinstance(path_tmpls, str):
                    path_tmpls = [path_tmpls]
                summ = {}
                for path_tmpl in path_tmpls:
                    path = path_tmpl.format(run=run)
                    if path not in cache:
                        cache[path] = load_inf_summary(path)
                    if any(task in d for d in cache[path].values()):
                        summ = cache[path]
                        break
                series = {}
                for ck in order:
                    row = summ.get(ck, {}).get(task)
                    series[ck] = None if row is None else row.get(metric)
                rows.append(((model, task, metric), series))
        write_csv(os.path.join(OUT, f"inf_{name}_{suffix}.csv"), ["model", "task", "metric"], order, rows)


# ══════════════════════════════════════════════════════════════
# 2) PROBING  (run 1..N 평균, layer 별)
# ══════════════════════════════════════════════════════════════
def _load_probe(pattern, value_key):
    """pattern.format(i=) -> dir of *.json. {task: {ckpt: {layer: mean(value_key)}}} (N_RUNS 평균)"""
    runs = []
    for i in range(1, N_RUNS + 1):
        d = defaultdict(lambda: defaultdict(dict))
        for f in glob.glob(pattern.format(i=i) + "/*.json"):
            try:
                data = json.load(open(f))
            except Exception:
                continue
            for ckpt, td in data.items():
                for task, ld in td.items():
                    for layer, m in ld.items():
                        if layer.startswith("_") or not isinstance(m, dict):
                            continue
                        d[task][ckpt][layer] = m.get(value_key, np.nan)
        runs.append(d)
    merged = defaultdict(lambda: defaultdict(dict))
    tasks = set().union(*[set(r.keys()) for r in runs]) if runs else set()
    for task in tasks:
        ckpts = set().union(*[set(r[task].keys()) for r in runs])
        for ck in ckpts:
            layers = set().union(*[set(r[task][ck].keys()) for r in runs])
            for layer in layers:
                vals = [r[task][ck].get(layer, np.nan) for r in runs]
                vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
                merged[task][ck][layer] = float(np.mean(vals)) if vals else np.nan
    return merged


def load_probe_clf(pattern):
    return _load_probe(pattern, "acc")


def load_probe_reg(pattern, metric="mae"):
    return _load_probe(pattern, metric)


def _layers_of(task_data):
    """task_data = {ckpt: {layer: val}} -> 존재하는 레이어를 정렬 순서로."""
    ls = set()
    for ck, ld in task_data.items():
        ls |= set(ld.keys())
    return sorted(ls, key=layer_sort_key)


def dump_probing(name, dir_tmpl, clf_sub, reg_sub, reg_metric, extra_clf=None):
    """dir_tmpl.format(run=) -> probing/<...>. clf_sub/reg_sub = 하위 폴더 prefix.
    extra_clf: [(root_tmpl, clf_sub, {task,...})] — 다른 위치에서 특정 task만 clf 로 이어붙임."""
    print(f"[probing:{name}]")
    for suffix, order, runs in variants():
        rows = []
        for model, run in runs.items():
            root = dir_tmpl.format(run=run)
            clf = load_probe_clf(os.path.join(root, clf_sub + "_{i}"))
            reg = load_probe_reg(os.path.join(root, reg_sub + "_{i}", "ridge"), reg_metric) if reg_sub else {}
            # clf tasks (accuracy)
            for task in sorted(clf.keys()):
                for layer in _layers_of(clf[task]):
                    series = {ck: clf[task].get(ck, {}).get(layer) for ck in order}
                    rows.append(((model, task, layer, "acc"), series))
            # reg tasks (raw metric, 예: mae)
            for task in sorted(reg.keys()):
                for layer in _layers_of(reg[task]):
                    series = {ck: reg[task].get(ck, {}).get(layer) for ck in order}
                    rows.append(((model, task, layer, reg_metric), series))
            # extra clf tasks (다른 소스, 예: 구 non-blind 위치의 line_plot)
            for xroot_tmpl, xsub, xtasks in (extra_clf or []):
                xclf = load_probe_clf(os.path.join(xroot_tmpl.format(run=run), xsub + "_{i}"))
                for task in sorted(t for t in xclf if t in xtasks and t not in clf):   # 본 소스에 이미 있으면 생략
                    for layer in _layers_of(xclf[task]):
                        series = {ck: xclf[task].get(ck, {}).get(layer) for ck in order}
                        rows.append(((model, task, layer, "acc"), series))
        write_csv(os.path.join(OUT, f"probing_{name}_{suffix}.csv"), ["model", "task", "layer", "metric"], order, rows)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    CURR_INF_SRC = BASE + "/extract_5_tasks/result/{run}/inference/accuracy/_summary_all.json"
    BEN_INF_SRC = BASE + "/extract_benchmark/result/{run}_blind/accuracy/_summary_all.json"
    ORIGIN_SRC = BASE + "/extract_benchmark/result_origin/{run}/accuracy/_summary_all.json"

    # ── inference ──
    dump_inference("curriculum", [(t, m, CURR_INF_SRC) for t, m in CURR_INF_TASKS])
    # blind 8 task + line_plot, mmvp. 후자 둘은 blind 결과 폴더에 있으면 그것(랜덤 오더링·Qwen), 없으면 구 result_origin(기존 fwd/rev).
    dump_inference("benchmark",
                   [(t, m, BEN_INF_SRC) for t, m in BEN_INF_TASKS]
                   + [("line_plot", "accuracy", [BEN_INF_SRC, ORIGIN_SRC]),
                      ("mmvp", "accuracy", [BEN_INF_SRC, ORIGIN_SRC]),
                      ("mmvp", "pair_accuracy", [BEN_INF_SRC, ORIGIN_SRC])])

    # ── probing ──
    dump_probing(
        "curriculum",
        BASE + "/probing/{run}",
        clf_sub="inf_classification", reg_sub="inf_regression", reg_metric="mae")
    # blind 8 task + 뒤에 line_plot(구 non-blind 위치 clf) 이어붙임
    dump_probing(
        "benchmark",
        BASE + "/probing/{run}_blind",
        clf_sub="ben_classification", reg_sub=None, reg_metric="mae",
        extra_clf=[(BASE + "/probing/{run}", "ben_classification", {"line_plot", "mmvp"})])

    # ── 선형(GPU torch) 프로브 — MLP 와 같은 프로토콜에서 은닉층만 뺀 용량 대조군 ──
    dump_probing(
        "curriculum_linear",
        BASE + "/probing/{run}",
        clf_sub="inf_classification_linear", reg_sub=None, reg_metric="mae")
    dump_probing(
        "benchmark_linear",
        BASE + "/probing/{run}_blind",
        clf_sub="ben_classification_linear", reg_sub=None, reg_metric="mae")

    # ── 선형(logreg) 프로브 — MLP 용량 대조군. inf 는 전 레이어, blind 는 proj 만(기존 fwd/rev 설정) 있을 수 있음 ──
    dump_probing(
        "curriculum_logreg",
        BASE + "/probing/{run}",
        clf_sub="inf_classification_logreg", reg_sub=None, reg_metric="mae")
    dump_probing(
        "benchmark_logreg",
        BASE + "/probing/{run}_blind",
        clf_sub="ben_classification_logreg", reg_sub=None, reg_metric="mae",
        extra_clf=[(BASE + "/probing/{run}", "ben_classification_logreg", {"line_plot", "mmvp"})])

    print("\nDONE ->", OUT)
