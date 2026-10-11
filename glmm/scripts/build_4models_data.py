# build_4models_data.py
# 1onr Minor 2 대응: GLMM_analysis_executed.ipynb 의 cell 0-28 데이터 파이프라인을 그대로
# 재현해 4개 모형(배터리 inference=d, 배터리 probe=dp, held-out inference=db, held-out probe=dbp)의
# 입력 데이터프레임을 만들고 CSV로 저장 (R에서 읽어 원본 스펙 그대로 재적합하기 위함).
#
# 원본 노트북이 쓰던 ./csvs/{inf,probing}_{curriculum,benchmark}_{forward,reverse}.csv 는
# /home/hail/vlm3/csv_out/ 에 이름만 살짝 다르게(probing_* vs prob_*) 그대로 존재함을 확인 —
# 세션 초반에 "소실됐다"고 판단했던 inf_curriculum_forward.csv 도 여기 있었다(정정).

import os
import numpy as np
import pandas as pd

DATA_DIR = "/home/hail/vlm3/csv_out"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csv_4models")
os.makedirs(OUT_DIR, exist_ok=True)


def to_long(df, direction, extra=()):
    stages = [4, 3, 2, 1, 0] if direction == 'reverse' else [0, 1, 2, 3, 4]
    rec = []
    for _, r in df.iterrows():
        flip = (r['metric'] == 'mae'); base = 1 - r['Base'] if flip else r['Base']
        for pos, k in enumerate(stages):
            v = r.get(f'S{k}_end')
            if pd.isna(v):
                continue
            row = dict(model=r['model'], task=r['task'], direction=direction,
                       time=pos + 1, base=base, acc=(1 - v) if flip else v)
            row.update({c: r[c] for c in extra}); rec.append(row)
    return pd.DataFrame(rec)


def add_base(d):
    d = d.join(d.groupby('model')['base'].mean().rename('base_bw'), on='model')
    d['base_wi'] = d['base'] - d['base_bw']
    d['delta'] = d['acc'] - d['base']; d['time_c'] = d['time'] - 3.0
    return d


def add_history(d):
    stage = d['task'].str[1].astype(int)
    own_end = np.where(d['direction'] == 'forward', stage + 1, (4 - stage) + 1)
    d['trained_yet'] = (d['time'] >= own_end).astype(int)
    d['recency'] = np.clip(d['time'] - own_end, 0, None)
    return d


def depth_label(df):
    n = df['layer'].str.extract(r'layer_(\d+)')[0].astype(float)
    L = n.groupby(df['model']).transform('max')
    return np.where(df['layer'] == 'proj', 'proj', (n / L).round(2).astype(str))


def add_depth_num(d):
    d['is_proj'] = (d['depth'] == 'proj').astype(int)
    d['depth_num'] = np.where(d['is_proj'] == 1, 0.0,
                              pd.to_numeric(d['depth'].where(d['depth'] != 'proj'), errors='coerce'))
    return d


DIR_CONTRAST = {'forward': 0.5, 'reverse': -0.5}


def dir_code(s):
    return s if pd.api.types.is_numeric_dtype(s) else s.map(DIR_CONTRAST).astype(float)


def load(name):
    return pd.read_csv(os.path.join(DATA_DIR, name))


inf_curr_fwd = load('inf_curriculum_forward.csv')
inf_curr_rvs = load('inf_curriculum_reverse.csv')
inf_bench_fwd = load('inf_benchmark_forward.csv')
inf_bench_rvs = load('inf_benchmark_reverse.csv')
prob_curr_fwd = load('probing_curriculum_forward.csv')
prob_curr_rvs = load('probing_curriculum_reverse.csv')
prob_bench_fwd = load('probing_benchmark_forward.csv')
prob_bench_rvs = load('probing_benchmark_reverse.csv')

# d — 배터리 inference
d = pd.concat([to_long(inf_curr_fwd, 'forward'), to_long(inf_curr_rvs, 'reverse')], ignore_index=True)
d = add_history(add_base(d))
d['direction'] = dir_code(d['direction'])
print('d(배터리 inference)  rows=%d models=%d tasks=%d' % (len(d), d.model.nunique(), d.task.nunique()))

# dp — 배터리 probe
dp = pd.concat([to_long(prob_curr_fwd, 'forward', extra=('layer',)),
                to_long(prob_curr_rvs, 'reverse', extra=('layer',))], ignore_index=True)
dp['depth'] = depth_label(dp)
dp = add_history(add_base(dp))
dp = add_depth_num(dp)
dp['cell'] = dp.model + ':' + dp.task + ':' + dp.direction + ':' + dp.time.astype(str)
dp['direction'] = dir_code(dp['direction'])
print('dp(배터리 probe)     rows=%d depths=%d cells=%d' % (len(dp), dp.depth.nunique(), dp.cell.nunique()))

# db — held-out inference
db = add_base(pd.concat([to_long(inf_bench_fwd, 'forward'), to_long(inf_bench_rvs, 'reverse')],
                        ignore_index=True))
db['direction'] = dir_code(db['direction'])
print('db(held-out inference) rows=%d' % len(db))

# dbp — held-out probe
dbp = pd.concat([to_long(prob_bench_fwd, 'forward', extra=('layer',)),
                 to_long(prob_bench_rvs, 'reverse', extra=('layer',))], ignore_index=True)
dbp['depth'] = depth_label(dbp)
dbp = add_base(dbp)
dbp = add_depth_num(dbp)
dbp['cell'] = dbp.model + ':' + dbp.task + ':' + dbp.direction + ':' + dbp.time.astype(str)
dbp['direction'] = dir_code(dbp['direction'])
print('dbp(held-out probe)  rows=%d' % len(dbp))

for name, df in [('d', d), ('dp', dp), ('db', db), ('dbp', dbp)]:
    out = os.path.join(OUT_DIR, f'{name}.csv')
    df.to_csv(out, index=False)
    print(f'[saved] {out}')

print('\nmodel 컬럼 값(전부):', sorted(d['model'].unique()))
