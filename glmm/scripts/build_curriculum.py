import os
import numpy as np, pandas as pd
D='./csvs/'
def to_long(df,direction):
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
d=pd.concat([to_long(pd.read_csv(D+'inf_curriculum_forward.csv'),'forward'),
             to_long(pd.read_csv(D+'inf_curriculum_reverse.csv'),'reverse')],ignore_index=True)
d=d.join(d.groupby('model')['base'].mean().rename('base_bw'),on='model')
d['base_wi']=d['base']-d['base_bw']; d['delta']=d['acc']-d['base']; d['time_c']=d['time']-3.0
stage=d['task'].str[1].astype(int)
own=np.where(d['direction']=='forward',stage+1,(4-stage)+1)
d['trained_yet']=(d['time']>=own).astype(int); d['recency']=np.clip(d['time']-own,0,None)
d['direction_c']=d['direction'].map({'forward':0.5,'reverse':-0.5})
d.to_csv(os.environ['OUT'],index=False); print('battery rows=%d'%len(d))
