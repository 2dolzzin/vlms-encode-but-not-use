"""projector 출력(vision encoder) 이 이미 답을 담고 있나, LM 안에서 만들어지나."""
import os
import numpy as np, pandas as pd
D='./csvs/'
CKPT=['Base']+[f'S{k}_{w}' for k in range(5) for w in ('mid','end')]

def long_probe(f, direction):
    d=pd.read_csv(D+f); rec=[]
    for _,r in d.iterrows():
        flip=(r['metric']=='mae')
        for c in CKPT:
            v=r.get(c)
            if pd.isna(v): continue
            rec.append(dict(model=r['model'],task=r['task'],layer=r['layer'],
                            direction=direction,ckpt=c,probe=(1-v) if flip else v))
    return pd.DataFrame(rec)

def long_inf(f, direction):
    d=pd.read_csv(D+f); rec=[]
    for _,r in d.iterrows():
        flip=(r['metric']=='mae')
        for c in CKPT:
            v=r.get(c)
            if pd.isna(v): continue
            rec.append(dict(model=r['model'],task=r['task'],direction=direction,
                            ckpt=c,inf=(1-v) if flip else v))
    return pd.DataFrame(rec)

def depth(df):
    n=df['layer'].str.extract(r'layer_(\d+)')[0].astype(float)
    L=n.groupby(df['model']).transform('max')
    return np.where(df['layer']=='proj', np.nan, (n/L).round(2))

def build(pf, pr, inf_f, inf_r, kind):
    p=pd.concat([long_probe(pf,'forward'), long_probe(pr,'reverse')],ignore_index=True)
    i=pd.concat([long_inf(inf_f,'forward'), long_inf(inf_r,'reverse')],ignore_index=True)
    p['depth']=depth(p)
    key=['model','task','direction','ckpt']
    proj = p[p.layer=='proj'].set_index(key)['probe'].rename('proj_probe')
    lm0  = p[p.depth==0.0].set_index(key)['probe'].rename('lm0_probe')
    last = p[p.depth==1.0].set_index(key)['probe'].rename('final_probe')
    mx   = p[p.depth>0].groupby(key)['probe'].max().rename('maxlm_probe')
    w = i.set_index(key).join([proj,lm0,last,mx]).reset_index()
    w['kind']=kind
    return w

bat = build('probing_curriculum_forward.csv','probing_curriculum_reverse.csv',
            'inf_curriculum_forward.csv','inf_curriculum_reverse.csv','battery')
ben = build('probing_benchmark_forward.csv','probing_benchmark_reverse.csv',
            'inf_benchmark_forward.csv','inf_benchmark_reverse.csv','benchmark')
w = pd.concat([bat,ben],ignore_index=True).dropna(subset=['inf','proj_probe','final_probe'])

pd.set_option('display.width',250)
print('=== task 별 평균 (모든 체크포인트) ===')
g=w.groupby(['kind','task'])[['proj_probe','lm0_probe','final_probe','maxlm_probe','inf']].mean()
g['gap_final_minus_inf']=g.final_probe-g['inf']
g['built_in_LM']=g.maxlm_probe-g.proj_probe
print(g.round(3).to_string())
print()
print('=== layer 0 은 정말 구조적 dip 인가 (proj / lm0 / final) ===')
print(w.groupby('kind')[['proj_probe','lm0_probe','final_probe']].mean().round(3).to_string())
w.to_csv(os.environ['OUT'],index=False); print('\nrows=%d saved'%len(w))
