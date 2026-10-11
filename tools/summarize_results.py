#!/usr/bin/env python3
"""csv_out/*.csv + train/eval_loss/eval_loss_table.csv 를 'Base → 최종' 한 줄 요약표(markdown)로. 실행: python tools/summarize_results.py
전체 11시점 시계열은 csv_out/ 의 CSV 를 볼 것. 출력: csv_out/summary_base_final.md (stdout 에도 출력)"""
import csv, glob, re, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 저장소 루트 (tools/ 의 상위)
OUT = os.path.join(ROOT, 'csv_out')
MODELS = ['llava', 'vila_3b', 'vila_8b', 'mipha', 'openflamingo', 'qwen25vl_7b']
def fmt(v): return '—' if v in ('', 'nan', None) else f"{float(v):.3f}"
def cell(r): return '—' if not r or (r[3] == '' and r[-1] == '') else f"{fmt(r[3])}→{fmt(r[-1])}"
def variant_files(prefix):
    fs = glob.glob(f'{OUT}/{prefix}_*.csv')
    return sorted(fs, key=lambda p: (not p.endswith('forward.csv'), not p.endswith('reverse.csv'), p))
L = []
def table(title, prefix, cols, key_fn, short):
    L.append(f"### {title}  (값: Base → 최종 시점)\n")
    L.append("| model | variant | " + " | ".join(short.get(c, c) for c in cols) + " |")
    L.append("|---|---|" + "---|" * len(cols))
    for f in variant_files(prefix):
        var = re.search(rf'{prefix}_(.*)\.csv', f).group(1)
        rows = list(csv.reader(open(f)))[1:]
        by = {key_fn(r): r for r in rows}
        for m in MODELS:
            cells = [cell(by.get((m,) + (c if isinstance(c, tuple) else (c,)))) for c in cols]
            if any(c != '—' for c in cells):
                L.append(f"| {m} | {var} | " + " | ".join(cells) + " |")
    L.append("")
# 1) eval loss (fwd/rev 만 존재; 랜덤 오더링은 아직 미계산)
L.append("### eval loss (각 스테이지 검증셋 5000, 낮을수록↓)  (값: Base → 최종 시점)\n")
rows = list(csv.DictReader(open(os.path.join(ROOT, 'train/eval_loss/eval_loss_table.csv'))))
L.append("| model | variant | S0 | S1 | S2 | S3 | S4 |"); L.append("|---|---|---|---|---|---|---|")
for m in ['llava', 'vila_3b', 'vila_8b', 'mipha', 'openflamingo']:
    for var, key in [('forward', m), ('reverse', m + '_reverse')]:
        rr = {r['metric']: r for r in rows if r['model'] == key}
        if not rr: continue
        L.append(f"| {m} | {var} | " + " | ".join(f"{fmt(rr[f'S{i}_loss']['baseline'])}→{fmt(rr[f'S{i}_loss']['s4_end' if var=='forward' else 's0_end'])}" for i in range(5)) + " |")
L.append("| (qwen, 랜덤 오더링) | — | 미계산 | | | | |\n")
# 2) 5-task
table("5-task inference (S1 은 MAE↓, 나머지 accuracy↑)", "inf_curriculum",
      ['S0_logamr', 'S1_similarity', 'S2_ishihara', 'S3_naming', 'S4_depth'], lambda r: (r[0], r[1]),
      {'S0_logamr': 'S0 logMAR', 'S1_similarity': 'S1 sim(MAE)', 'S2_ishihara': 'S2 ishihara', 'S3_naming': 'S3 naming', 'S4_depth': 'S4 depth'})
# 3) blind 8 + line_plot + mmvp
bt = ['circled_letter', 'touching_circles', 'nested_squares', 'olympic_circles', 'olympic_pentagons', 'subway_connections',
      'counting_grid_blank', 'counting_grid_word', 'line_plot']
cols = [(t, 'accuracy') for t in bt] + [('mmvp', 'accuracy'), ('mmvp', 'pair_accuracy')]
short = {('circled_letter','accuracy'):'circ', ('touching_circles','accuracy'):'touch', ('nested_squares','accuracy'):'nest',
         ('olympic_circles','accuracy'):'oly-c', ('olympic_pentagons','accuracy'):'oly-p', ('subway_connections','accuracy'):'subway',
         ('counting_grid_blank','accuracy'):'grid-b', ('counting_grid_word','accuracy'):'grid-w', ('line_plot','accuracy'):'line',
         ('mmvp','accuracy'):'MMVP q-acc', ('mmvp','pair_accuracy'):'MMVP pair'}
table("blind 8 + line_plot + MMVP inference accuracy", "inf_benchmark", cols, lambda r: (r[0], r[1], r[2]), short)
L.append("— : 결과 없음 또는 파싱 전멸(NaN). 랜덤 오더링의 '최종'은 그 순서의 마지막 스테이지 end.")
md = "\n".join(L); open(f'{OUT}/summary_base_final.md', 'w').write(md); print(md)
