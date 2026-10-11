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

db = add_base(pd.concat([to_long(pd.read_csv(D+'inf_benchmark_forward.csv'),'forward'),
                         to_long(pd.read_csv(D+'inf_benchmark_reverse.csv'),'reverse')],
                        ignore_index=True))
db['type'] = np.where(db.task=='mmvp','MMVP','BlindTest')
db['direction_c'] = db['direction'].map({'forward':0.5,'reverse':-0.5})

pd.set_option('display.width', 200)
print('=== 관측 수 / 결측 ===')
full = db.groupby('type').size()
expect = pd.Series({'BlindTest': 5*9*2*5, 'MMVP': 5*1*2*5})
print(pd.DataFrame({'obs': full, 'expected': expect, 'missing': expect-full}).to_string())
print()
print('결측이 몰린 곳 (model x type):')
have = db.groupby(['type','model']).size().unstack(0)
exp2 = pd.DataFrame({'BlindTest':[90]*5,'MMVP':[10]*5}, index=have.index)
print((exp2-have.fillna(0)).astype(int).to_string())

print()
print('=== base / acc / delta by type ===')
print(db.groupby('type')[['base','acc','delta']].mean().round(4).to_string())
print()
print('=== delta by type x time ===')
print(db.groupby(['type','time'])['delta'].mean().unstack().round(4).to_string())
print()
print('=== delta by type x direction ===')
print(db.groupby(['type','direction'])['delta'].agg(['mean','std','count']).round(4).to_string())
print()
print('=== BlindTest task 별 base / delta ===')
print(db[db.type=='BlindTest'].groupby('task')[['base','delta']].mean().round(4).to_string())

db.to_csv(os.environ['OUT'], index=False)
print('\nrows=%d saved' % len(db))
