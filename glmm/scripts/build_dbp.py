import os
import numpy as np, pandas as pd

D = './csvs/'

def to_long(df, direction, extra=()):
    stages = [4,3,2,1,0] if direction=='reverse' else [0,1,2,3,4]
    rec=[]
    for _, r in df.iterrows():
        flip = (r['metric']=='mae'); base = 1-r['Base'] if flip else r['Base']
        for pos,k in enumerate(stages):
            v = r.get(f'S{k}_end')
            if pd.isna(v): continue
            row = dict(model=r['model'], task=r['task'], direction=direction,
                       time=pos+1, base=base, acc=(1-v) if flip else v)
            row.update({c: r[c] for c in extra}); rec.append(row)
    return pd.DataFrame(rec)

def add_base(d):
    d = d.join(d.groupby('model')['base'].mean().rename('base_bw'), on='model')
    d['base_wi'] = d['base']-d['base_bw']
    d['delta'] = d['acc']-d['base']; d['time_c'] = d['time']-3.0
    return d

def depth_label(df):
    n = df['layer'].str.extract(r'layer_(\d+)')[0].astype(float)
    L = n.groupby(df['model']).transform('max')
    return np.where(df['layer']=='proj','proj',(n/L).round(2).astype(str))

def add_depth_num(d):
    d['is_proj'] = (d['depth']=='proj').astype(int)
    d['depth_num'] = np.where(d['is_proj']==1, 0.0,
                              pd.to_numeric(d['depth'].where(d['depth']!='proj'), errors='coerce'))
    return d

dbp = pd.concat([to_long(pd.read_csv(D+'probing_benchmark_forward.csv'),'forward',extra=('layer',)),
                 to_long(pd.read_csv(D+'probing_benchmark_reverse.csv'),'reverse',extra=('layer',))],
                ignore_index=True)
dbp['depth'] = depth_label(dbp); dbp = add_depth_num(add_base(dbp))
dbp['cell'] = dbp.model+':'+dbp.task+':'+dbp.direction+':'+dbp.time.astype(str)
dbp['type'] = np.where(dbp.task=='mmvp','MMVP','BlindTest')
dbp['direction_c'] = dbp['direction'].map({'forward':0.5,'reverse':-0.5})

pd.set_option('display.width', 200)
print('=== base / delta by depth (proj 별도) ===')
g = dbp.groupby(['type', dbp.depth])[['base','acc','delta']].agg(['mean'])
g.columns = ['base','acc','delta']
print((g*1).round(4).to_string())

print()
print('=== headroom vs delta by depth (BlindTest only) ===')
b = dbp[dbp.type=='BlindTest'].groupby('depth').agg(
    base=('base','mean'), delta=('delta','mean'), headroom=('base', lambda s: (1-s).mean()))
b['delta_per_headroom'] = b.delta/b.headroom
print(b.round(4).to_string())

print()
print('=== delta by time (BlindTest / MMVP) ===')
print(dbp.groupby(['type','time'])['delta'].mean().unstack().round(4).to_string())

print()
print('=== delta by direction ===')
print(dbp.groupby(['type','direction'])['delta'].mean().round(4).to_string())

print()
print('=== base_wi 가 MMVP 를 어떻게 잡는가 ===')
print(dbp.groupby('type')[['base','base_wi','delta']].mean().round(4).to_string())

dbp.to_csv(os.environ['SP']+'/dbp.csv', index=False)
print('\nrows=%d saved' % len(dbp))
