#!/usr/bin/env python3
"""모델별 × variant(forward/reverse/order_*) 별로 11시점 전체 표 4종(eval loss / 5-task / blind / MMVP) 을 markdown 으로.
열 = baseline + 학습 진행 순 (mid/end). S1 은 1−MAE(↑) 로 변환. 실행: python tools/make_tables.py
출력: csv_out/tables/<model>.md (모델별) + csv_out/tables_by_model.md (전체)"""
import csv, glob, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (tools/ 의 상위)
OUT = os.path.join(ROOT, 'csv_out'); TD = f'{OUT}/tables'; os.makedirs(TD, exist_ok=True)
MODELS = ['llava', 'vila_3b', 'vila_8b', 'mipha', 'openflamingo', 'qwen25vl_7b']
NICE = {'llava': 'LLaVA-1.5-7B', 'vila_3b': 'VILA-1.5-3B', 'vila_8b': 'VILA-1.5-8B', 'mipha': 'Mipha-3B',
        'openflamingo': 'OpenFlamingo-3B', 'qwen25vl_7b': 'Qwen2.5-VL-7B'}
VNAME = {'forward': 'forward (S0→S4)', 'reverse': 'reverse (S4→S0)',
         'order_14203': 'random 1→4→2→0→3', 'order_20431': 'random 2→0→4→3→1', 'order_31024': 'random 3→1→0→2→4'}
def f3(v, transform=None):
    if v in ('', 'nan', None): return '—'
    x = float(v); x = transform(x) if transform else x
    return f"{x:.3f}"

def read_variant(prefix, var):
    p = f'{OUT}/{prefix}_{var}.csv'
    if not os.path.exists(p): return None, {}
    rows = list(csv.reader(open(p))); hdr = rows[0]
    cols = ['baseline' if h == 'Base' else h for h in hdr[3:]]
    return cols, {tuple(r[:3]): r[3:] for r in rows[1:]}

def md_table(cols, rows):
    out = ["| task | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, vals in rows: out.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(out)

import sys as _sys; _sys.path.insert(0, ROOT)
from order_runs import discover_order_runs, read_order, order_tag   # noqa: E402
_ORDER_SEQ = {}   # tag → [sid,...]  (checkpoint/*_order_<tag>_nohex/order.txt)
for _m, _p in {'llava': 'llava', 'mipha': 'Mipha', 'vila_3b': 'VILA_3B', 'vila_8b': 'VILA_8B', 'openflamingo': 'OpenFlamingo', 'qwen25vl_7b': 'Qwen25VL_7B'}.items():
    for _k, _c in discover_order_runs(os.path.join(ROOT, 'checkpoint'), _p, _m).items():
        _ORDER_SEQ.setdefault(order_tag(_c['root']), read_order(_c['root']))

def eval_loss_rows(model, var):
    if var in ('forward', 'reverse'):
        key = model if var == 'forward' else model + '_reverse'
        seq = [0, 1, 2, 3, 4] if var == 'forward' else [4, 3, 2, 1, 0]
    else:
        tag = var.split('_')[-1]; key = f"{model}_order_{tag}"
        if tag not in _ORDER_SEQ: return None
        seq = [int(sid[1]) for sid in _ORDER_SEQ[tag]]
    rows = [r for r in csv.DictReader(open(os.path.join(ROOT, 'train/eval_loss/eval_loss_table.csv'))) if r['model'] == key]
    if not rows: return None
    order = ['baseline'] + [f"s{i}_{m}" for i in seq for m in ('mid', 'end')]
    cols = ['baseline'] + [f"S{c[1]}_{c[3:]}" for c in order[1:]]
    by = {r['metric']: r for r in rows}
    return cols, [(f"S{i} loss (↓)", [f3(by[f'S{i}_loss'][c]) for c in order]) for i in range(5)]

TASK5 = [('S0_logamr', 'S0_logmar (acc)', None), ('S1_similarity', 'S1_sim (1−MAE↑)', lambda x: 1 - x),
         ('S2_ishihara', 'S2_ishihara (acc)', None), ('S3_naming', 'S3_naming (acc)', None), ('S4_depth', 'S4_depth (acc)', None)]
BLIND = [('circled_letter', 'circled_letter'), ('touching_circles', 'touching_circles'), ('nested_squares', 'nested_squares'),
         ('olympic_circles', 'olympic_circles'), ('olympic_pentagons', 'olympic_pentagons'), ('subway_connections', 'subway_connections'),
         ('counting_grid_blank', 'counting_grid_blank'), ('counting_grid_word', 'counting_grid_word'), ('line_plot', 'line_plot')]

ALL = []
for model in MODELS:
    sec = [f"# {NICE[model]} (`{model}`)\n"]
    for var in ['forward', 'reverse', 'order_14203', 'order_20431', 'order_31024']:
        c5, d5 = read_variant('inf_curriculum', var)
        cb, db = read_variant('inf_benchmark', var)
        nonblank = lambda d: any(k[0] == model and any(v not in ('', 'nan') for v in vals) for k, vals in d.items())
        has5 = nonblank(d5); hasb = nonblank(db)   # 값이 하나도 없는 variant(예: 학습 중인 VILA-8B 랜덤)는 생략
        el = eval_loss_rows(model, var)
        if not (has5 or hasb or el): continue
        sec.append(f"## {VNAME[var]}\n")
        if el:
            sec.append("### 1. eval loss\n"); sec.append(md_table(*el)); sec.append("")
        else:
            sec.append("### 1. eval loss\n\n(미계산)\n")
        if has5:
            sec.append("### 2. 5-task inference\n")
            sec.append(md_table(c5, [(nm, [f3(v, tr) for v in d5.get((model, t, 'mae' if t == 'S1_similarity' else 'accuracy'), [''] * len(c5))]) for t, nm, tr in TASK5])); sec.append("")
        if hasb:
            sec.append("### 3. blind (acc)\n")
            sec.append(md_table(cb, [(nm, [f3(v) for v in db.get((model, t, 'accuracy'), [''] * len(cb))]) for t, nm in BLIND])); sec.append("")
            sec.append("### 4. MMVP\n")
            sec.append(md_table(cb, [("MMVP question acc (chance 0.5)", [f3(v) for v in db.get((model, 'mmvp', 'accuracy'), [''] * len(cb))]),
                                     ("MMVP pair acc (chance 0.25)", [f3(v) for v in db.get((model, 'mmvp', 'pair_accuracy'), [''] * len(cb))])])); sec.append("")
    text = "\n".join(sec); open(f'{TD}/{model}.md', 'w').write(text); ALL.append(text)
open(f'{OUT}/tables_by_model.md', 'w').write("\n\n---\n\n".join(ALL))
print("saved:", f'{OUT}/tables_by_model.md', "+", ", ".join(f'{TD}/{m}.md' for m in MODELS))
