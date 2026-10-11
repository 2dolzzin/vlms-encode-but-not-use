"""layer 0 을 '이미지 없이 읽었을 때의 바닥값'으로 삼아 정보의 출처를 분류한다."""
import os
import pandas as pd

w = pd.read_csv(os.environ['W'])
g = w.groupby(['kind','task']).agg(
        prior=('lm0_probe','mean'), prior_sd=('lm0_probe','std'),
        proj=('proj_probe','mean'), maxlm=('maxlm_probe','mean'), inf=('inf','mean')).reset_index()
g['img_signal_lm'] = g.maxlm - g.prior     # LM 최고층이 no-image floor 위로 얼마나
g['gained_in_lm']  = g.maxlm - g.proj      # LM 안에서 추가된 몫
g['beyond_probe']  = g['inf'] - g.maxlm    # 행동이 probe 를 넘는 몫

def regime(r):
    if r.beyond_probe > 0.05:   return 'generated at decoding'
    if r.img_signal_lm < 0.10:  return 'absent (at no-image floor)'
    if r.gained_in_lm > 0.10:   return 'readout - assembled inside LM'
    return 'readout - already linear at projector'

g['regime'] = g.apply(regime, axis=1)
g.round(4).to_csv(os.environ['OUT'], index=False)
cols=['kind','task','prior','proj','maxlm','inf','img_signal_lm','gained_in_lm','beyond_probe','regime']
print(g[cols].round(3).to_string(index=False))
print('\nno-image floor 의 모델 간 sd 최대: %.5f (5개 모델이 사실상 동일)' % g.prior_sd.max())
