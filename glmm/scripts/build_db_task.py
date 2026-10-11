import os
import numpy as np, pandas as pd
D='./csvs/'

def to_long(df, direction):
    stages=[4,3,2,1,0] if direction=='reverse' else [0,1,2,3,4]
    rec=[]
    for _,r in df.iterrows():
        flip=(r['metric']=='mae'); base=1-r['Base'] if flip else r['Base']
        for pos,k in enumerate(stages):
            v=r.get(f'S{k}_end')
            if pd.isna(v): continue
            rec.append(dict(model=r['model'],task=r['task'],direction=direction,
                            time=pos+1,base=base,acc=(1-v) if flip else v))
    return pd.DataFrame(rec)

db=pd.concat([to_long(pd.read_csv(D+'inf_benchmark_forward.csv'),'forward'),
              to_long(pd.read_csv(D+'inf_benchmark_reverse.csv'),'reverse')],ignore_index=True)
db=db.join(db.groupby('model')['base'].mean().rename('base_bw'),on='model')
db['base_wi']=db['base']-db['base_bw']; db['delta']=db['acc']-db['base']; db['time_c']=db['time']-3.0
db['direction_c']=db['direction'].map({'forward':0.5,'reverse':-0.5})

# ── a priori mapping: held-out task -> battery stage that trains the needed competence
#    S0 acuity/contrast = reading capital letters   S1/S2/S3 = color   S4 = depth+search
SOURCE = {'circled_letter': 0}                    # strict: letter identification <- S0
SOURCE_BROAD = {'circled_letter': 0, 'subway_connections': 2, 'olympic_circles': 2}  # + color-traced lines

def add_source(d, mapping, suffix=''):
    st = d['task'].map(mapping)
    # stage k 는 forward 에서 k+1 번째, reverse 에서 (4-k)+1 번째 세션에 학습된다
    own_end = np.where(d['direction']=='forward', st+1, (4-st)+1)
    trained = np.where(st.isna(), 0, (d['time'] >= own_end).astype(float))
    rec = np.where(st.isna(), 0, np.clip(d['time']-own_end, 0, None))
    d['source_trained'+suffix] = trained
    d['source_recency'+suffix] = rec
    return d

db = add_source(db, SOURCE)
db = add_source(db, SOURCE_BROAD, '_broad')
db['type']=np.where(db.task=='mmvp','MMVP','BlindTest')

pd.set_option('display.width',220)
print('=== task 별 delta (mean ± sd, n) ===')
g=db[db.type=='BlindTest'].groupby('task')['delta'].agg(['mean','std','count']).round(4)
g['se']=(g['std']/np.sqrt(g['count'])).round(4)
print(g.to_string())
print()
print('=== circled_letter: source(S0) 학습 전/후 ===')
cl=db[db.task=='circled_letter']
print(cl.groupby(['direction','time'])[['acc','delta','source_trained']].mean().round(4).to_string())
print()
print(cl.groupby('source_trained')['delta'].agg(['mean','std','count']).round(4).to_string())
db.to_csv(os.environ['OUT'],index=False); print('\nsaved rows=%d' % len(db))
