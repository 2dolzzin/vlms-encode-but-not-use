# 모형 수식 정리 (LaTeX + lme4)

각 모형은 **수식 → 변수 설명 표 → lme4 코드 → 검정** 순으로 제시한다. README, `glmm_vlm_curriculum.R`과 짝을 이룬다.

## 공통 사항

모든 모형은 이항 가능도와 logit 링크를 쓴다: 셀의 정답 수 $k \sim \mathrm{Binomial}(n, p)$. lme4에서는 y 자리에 요약 `acc`를 두고 `family = binomial(link = "logit")`, `weights = n`으로 지정한다. 예외는 MF 하나로, 행이 시행 1회(Bernoulli)라 weights가 없다. 모든 random effect는 평균 0 정규분포를 따른다. 계수 기호는 각 모형 안에서만 유효하다.

### baseline 분해 (여러 모형이 공유하는 공변량)

$$B_{mj}=\log\frac{k^{(0)}_{mj}+0.5}{\,n^{(0)}_{mj}-k^{(0)}_{mj}+0.5\,},\qquad B^{(b)}_{m}=\frac{1}{J}\sum_{j=1}^{J} B_{mj},\qquad B^{(w)}_{mj}=B_{mj}-B^{(b)}_{m}$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $k^{(0)}_{mj},\ n^{(0)}_{mj}$ | — | 학습 전(t0)에 모델 $m$이 태스크 $j$에서 맞힌 개수와 분모 |
| $B_{mj}$ | — | t0 성능의 empirical logit (+0.5는 0%·100% 셀 보정) |
| $J$ | — | 태스크 수 |
| $B^{(b)}_{m}$ | `base_bw` | 모델 평균 — 전반적으로 좋은 모델인가 (between) |
| $B^{(w)}_{mj}$ | `base_wi` | 평균 대비 편차 — 그 태스크를 유난히 잘하는가 (within) |

## M0 — Moravec 확인 (t0, 셀 $(m,j)$)

$$\operatorname{logit}(p_{mj})=\beta_0+\gamma_S\,\tilde s_j+b_{0m}+b_{Sm}\tilde s_j+a_j+\varepsilon_{mj}$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $p_{mj}$ | y = `acc`, `weights = n` | t0에서 모델 $m$이 태스크 $j$를 맞힐 확률. 관측 $k_{mj}\sim\mathrm{Binomial}(n_{mj},p_{mj})$ |
| $\beta_0$ | `(Intercept)` | 절편 — 평균 stage 태스크의 logit 정확도 |
| $\tilde s_j$ | `stage_c` | developmental stage의 중심화 값 |
| $\gamma_S$ | `stage_c` 계수 | stage 기울기 — Moravec 검정 자체 (예측: 양수 = 늦게 발달하는 능력일수록 base 성능이 높음) |
| $b_{0m},\ b_{Sm}$ | `(1 + stage_c \| model)` | 모델별 수준·stage 기울기 편차, 공분산 $\Sigma_M$ |
| $a_j$ | `(1 \| task)` | stage로 설명되지 않는 태스크 고유성 |
| $\varepsilon_{mj}$ | `(1 \| obs)` | 행 단위 과산포(OLRE) |

```r
m0 <- glmer(acc ~ stage_c + (1 + stage_c | model) + (1 | task) + (1 | obs),
            weights = n, family = binomial, data = d0, control = ctrl)
```

검정: $H_0\!:\gamma_S=0$, 예측 $\gamma_S>0$.

## M1 — 전이 (benchmark 8개, 셀 $i=(m,d,j,t)$)

$$
\begin{aligned}
\operatorname{logit}(p_i)={}&\beta_0+\beta_wB^{(w)}_{mj}+\beta_bB^{(b)}_m
+\gamma_DD_d+\gamma_T\tilde t+\gamma_S\tilde s_j\\
&+\gamma_{DT}D_d\tilde t+\gamma_{ST}\tilde s_j\tilde t+\gamma_{DS}D_d\tilde s_j+\gamma_{DST}D_d\tilde s_j\tilde t\\
&+b_{0m}+b_{Dm}D_d+b_{Tm}\tilde t+a_{0j}+a_{Dj}D_d+w_{mj}+\varepsilon_i
\end{aligned}
$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $p_i$ | y = `acc`, `weights = n` | 셀 $i=(m,d,j,t)$ — 모델 $m$이 방향 $d$로 학습해 시점 $t$에 태스크 $j$를 맞힐 확률 |
| $D_d$ | `direction` | reverse 지시변수 (forward = 0). 계수는 forward 대비 reverse의 차이 |
| $\tilde t$ | `time_c` | 누적 학습 세션 수(중간 체크포인트 0.5 단위)의 중심화 값 |
| $\tilde s_j$ | `stage_c` | developmental stage의 중심화 값 |
| $B^{(w)}_{mj},\ B^{(b)}_m$ | `base_wi`, `base_bw` | t0 사전역량의 within/between 성분 (공통 사항 참조) |
| $\beta_w,\ \beta_b$ | `base_wi`, `base_bw` 계수 | 사전역량의 효과 — 태스크 수준 귀인 / 모델 수준 귀인 |
| $\gamma_D$ | `directionreverse` | 방향 주효과 (평균적 차이) |
| $\gamma_T$ | `time_c` | 학습에 따른 평균 상승 |
| $\gamma_S$ | `stage_c` | stage 주효과 |
| $\gamma_{DT}$ | `directionreverse:time_c` | 방향×시간 — reverse의 학습(전이) 속도가 다른가 |
| $\gamma_{ST}$ | `stage_c:time_c` | stage×시간 — 학습이 Moravec 경사를 바꾸는가 (음수 = 완화) |
| $\gamma_{DS}$ | `directionreverse:stage_c` | 방향×stage — 방향별 stage 프로파일 수준 차 |
| $\gamma_{DST}$ | `directionreverse:stage_c:time_c` | 방향×stage×시간 — 방향이 저-stage 태스크를 차등 회복시키는가 (**primary**) |
| $b_{0m},b_{Dm},b_{Tm}$ | `(1 + direction + time_c \| model)` | 모델별 수준·방향·시간 반응 편차, 공분산 $\Sigma_M$ — base 투입 후 남는 이 분산이 "모델 종류" 귀인의 증거 |
| $a_{0j},a_{Dj}$ | `(1 + direction \| task)` | 태스크 고유성과 태스크별 방향 반응, 공분산 $\Sigma_J$ |
| $w_{mj}$ | `(1 \| model:task)` | 모델×태스크 궁합 — 반복측정 의존성 흡수 |
| $\varepsilon_i$ | `(1 \| obs)` | 행 단위 과산포(OLRE) |

```r
m1 <- glmer(acc ~ base_wi + base_bw + direction * stage_c * time_c
              + (1 + direction + time_c | model) + (1 + direction | task)
              + (1 | model:task) + (1 | obs),
            weights = n, family = binomial, data = db, control = ctrl)
```

검정: $\gamma_{ST}<0$(완화), $\gamma_{DT}$(방향의 전이), $\gamma_{DST}$(primary), $\beta_w,\beta_b$(귀인), base 투입 후 남는 $\Sigma_M$(모델 종류).

## M2 — 세션 태스크, 순서 대 망각 (세션 5개, 셀 $i$)

$$\operatorname{logit}(p_i)=\beta_0+\beta_wB^{(w)}_{mj}+\beta_bB^{(b)}_m+\gamma_S\tilde s_j+\gamma_AA_i+\gamma_{SA}\tilde s_jA_i+\gamma_RR_i+\gamma_{SR}\tilde s_jR_i+\gamma_DD_d+b_{0m}+w_{mj}+\varepsilon_i$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $p_i,\ \tilde s_j,\ D_d,\ B^{(w)}_{mj},\ B^{(b)}_m,\ \beta_w,\ \beta_b$ | M1과 동일 | M1과 동일한 역할 |
| $A_i$ | `trained_yet` | 시점 $t$에 자기 태스크의 세션 학습이 끝났는가 (0/1) |
| $R_i$ | `recency` | 자기 세션 종료 후 경과 세션 수 (미학습 행은 0, 구분은 $A_i$가 담당) |
| $\gamma_A$ | `trained_yet` 계수 | 자기 세션 학습의 즉각 이득 (직후 시점 기준) |
| $\gamma_{SA}$ | `stage_c:trained_yet` | 그 이득의 stage 의존 — 쉬운 태스크가 더 많이 배우는가 |
| $\gamma_R$ | `recency` 계수 | 망각 기울기 — 자기 세션 이후 세션이 지날수록 떨어지는 속도 |
| $\gamma_{SR}$ | `stage_c:recency` | 망각의 stage 의존 — 쉬운 태스크가 더 빨리 잊히는가 |
| $\gamma_D$ | `directionreverse` | 같은 $A_i, R_i$ 조건에서 남는 순서/이력 효과 |
| $b_{0m},\ w_{mj},\ \varepsilon_i$ | `(1 \| model)`, `(1 \| model:task)`, `(1 \| obs)` | M1과 동일한 역할 |

```r
m2 <- glmer(acc ~ base_wi + base_bw + stage_c * trained_yet + stage_c * recency
              + direction + (1 | model) + (1 | model:task) + (1 | obs),
            weights = n, family = binomial, data = ds, control = ctrl)
```

검정: $\gamma_A,\gamma_{SA}$(학습 이득과 stage 의존), $\gamma_R$(망각), $\gamma_{SR}$(stage별 망각), $A_i=1,R_i=0$ 고정 시의 방향 대비.

## M3 — 해리 (probing vs inference, time = 1..5, 셀 $(i,P)$)

$$
\begin{aligned}
\operatorname{logit}(p_{iP})={}&\beta_0+\beta_wB^{(w)}_{mj}+\beta_bB^{(b)}_m
+\mu P+\gamma_DD_d+\gamma_T\tilde t+\gamma_S\tilde s_j+\gamma_{DT}D_d\tilde t\\
&+\mu_DPD_d+\mu_TP\tilde t+\mu_{DT}PD_d\tilde t+\mu_SP\tilde s_j
+b_{0m}+b_{Pm}P+w_{mj}+\varepsilon_{iP}
\end{aligned}
$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $P$ | `measure` | probing 지시변수 (inference = 0). probing 행의 $n$은 probe held-out item 수 |
| $p_{iP}$ | y = `acc`, `weights = n` | 셀 $i$를 측정 방식 $P$로 평가했을 때의 정답 확률 |
| $\mu$ | `measureprobing` | 정보–표현 격차의 절편 — inference 대비 probing이 평균적으로 얼마나 높은가 |
| $\mu_D$ | `measureprobing:directionreverse` | 격차의 방향 의존 |
| $\mu_T$ | `measureprobing:time_c` | 격차의 학습량 의존 |
| $\mu_{DT}$ | `measureprobing:directionreverse:time_c` | 격차의 방향×시간 의존 — **해리 분석의 핵심 검정** |
| $\mu_S$ | `measureprobing:stage_c` | 격차의 stage 의존 |
| $\gamma_D,\gamma_T,\gamma_S,\gamma_{DT}$ | M1과 동일 | inference 쪽(기준 수준)의 조건 효과 |
| $b_{Pm}$ | `(1 + measure \| model)`의 slope | 격차 크기의 모델 간 이질성 |
| 나머지 | M1과 동일 | $\beta_0,\beta_w,\beta_b,b_{0m},w_{mj},\varepsilon$ |

```r
m3 <- glmer(acc ~ measure * direction * time_c + measure * stage_c
              + base_wi + base_bw
              + (1 + measure | model) + (1 | model:task) + (1 | obs),
            weights = n, family = binomial, data = dm, control = ctrl)
```

검정: 해리 주장은 오직 $\mu$ 계열 상호작용으로만 (Nieuwenhuis et al., 2011). 핵심은 $H_0\!:\mu_{DT}=0$.

## MP1 — layer 프로파일 (전체 layer 스택, 행 $(i,\ell)$)

$$
\begin{aligned}
\operatorname{logit}(p_{i\ell})={}&\beta_0+\beta_wB^{(w)}_{mj}+\beta_bB^{(b)}_m
+\gamma_DD_d+\gamma_T\tilde t+\gamma_{DT}D_d\tilde t+\gamma_S\tilde s_j\\
&+\sum_{q=1}^{2}\Big(\alpha_q+\delta_qD_d+\eta_q\tilde t+\theta_qD_d\tilde t+\kappa_q\tilde s_j\Big)\phi_q(u_{m\ell})\\
&+b_{0m}+\textstyle\sum_{q=1}^{2} b_{qm}\,\phi_q(u_{m\ell})+v_i+w_{mj}+\varepsilon_{i\ell}
\end{aligned}
$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $\ell,\ L_m$ | `layer`, `L` | layer 번호와 그 모델의 총 layer 수 |
| $u_{m\ell}=\frac{\ell-1}{L_m-1}$ | `depth` | 상대 깊이 (0 = 입력 쪽 첫 layer, 1 = 출력 쪽 마지막). $L_m$이 모델마다 달라 절대 번호 대신 사용 |
| $\phi_1(u),\ \phi_2(u)$ | `depth1`, `depth2` | 상대 깊이의 직교 1·2차 다항 — 직선 추세 축 / 굽음 축 (`poly(depth, 2)`) |
| $q$ | — | 다항 차수 인덱스 (1 = 선형, 2 = 이차) |
| $\alpha_q$ | `depth1`, `depth2` 계수 | 깊이 프로파일 자체 — $\alpha_2<0$이면 중간층 피크(역U) |
| $\delta_q$ | `depth1:directionreverse` 등 | 방향별 프로파일 차이 |
| $\eta_q$ | `depth1:time_c` 등 | 학습에 따른 프로파일 변화 — 정보가 어느 깊이에서 자라는가 |
| $\theta_q$ | `depth1:directionreverse:time_c` 등 | 방향×시간에 따른 프로파일 변화 — 방향이 "어느 깊이"를 키우는가 (**확증 후보**) |
| $\kappa_q$ | `depth1:stage_c` 등 | stage별 프로파일 차이 — 저-stage 정보가 사는 깊이 |
| $b_{qm}$ | `(1 + depth1 + depth2 \| model)`의 slope | 프로파일 모양의 모델 간 이질성 (아키텍처 차) |
| $v_i$ | `(1 \| cell)` | 같은 체크포인트($i$)의 layer들이 공유하는 표상 수준 |
| 나머지 | M1과 동일 | $\gamma$ 계열, $\beta_w,\beta_b$, $b_{0m}$, $w_{mj}$; $\varepsilon_{i\ell}$은 layer 행 단위 OLRE |

```r
mp1 <- glmer(acc ~ base_wi + base_bw
               + (depth1 + depth2) * direction * time_c
               + (depth1 + depth2) * stage_c
               + (1 + depth1 + depth2 | model)
               + (1 | cell) + (1 | model:task) + (1 | obs),
             weights = n, family = binomial, data = dp_all, control = ctrl)
anova(update(mp1, . ~ . - (depth1 + depth2):direction:time_c), mp1)  # H0: theta1 = theta2 = 0
```

검정: $\alpha_2$(중간층 피크), $H_0\!:\eta_1=\eta_2=0$($\chi^2_2$), $H_0\!:\theta_1=\theta_2=0$($\chi^2_2$, 확증 후보 9번), $\kappa_q$(stage별 깊이).

## MP2 — 깊이 구간별 개별 GLMM (구간 $h$, 행 $(i,\ell)$)

$$\operatorname{logit}(p^{(h)}_{i\ell})=\beta^{(h)}_0+\beta^{(h)}_wB^{(w)}_{mj}+\beta^{(h)}_bB^{(b)}_m+\gamma^{(h)}_DD_d+\gamma^{(h)}_T\tilde t+\gamma^{(h)}_{DT}D_d\tilde t+\gamma^{(h)}_S\tilde s_j+b^{(h)}_{0m}+w^{(h)}_{mj}+v^{(h)}_i+\varepsilon^{(h)}_{i\ell}$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $h$ | `dbin` | 상대 깊이 10분위 구간. 구간 $h$는 $u\in\big(\tfrac{h-1}{10},\tfrac{h}{10}\big]$의 행만 포함 |
| 위 첨자 $(h)$ | — | 같은 모형을 구간마다 따로 적합해 얻는 구간별 계수라는 표시 |
| $\gamma^{(h)}_{DT}$ | `directionreverse:time_c` (구간별) | 그 깊이 구간의 방향×시간 효과 — 깊이에 대해 플롯하는 목표 프로파일 |
| $v^{(h)}_i$ | `(1 \| cell)` | 한 구간에 같은 셀의 layer가 여러 개 들어갈 때의 공유 성분 (구간당 1개뿐이면 제거) |
| 나머지 | M1·MP1과 동일 | 동일 역할의 구간별 버전 |

```r
mh <- glmer(acc ~ base_wi + base_bw + direction * time_c + stage_c
              + (1 | model) + (1 | model:task) + (1 | cell) + (1 | obs),
            weights = n, family = binomial,
            data = droplevels(subset(dp_all, dbin == b)), control = ctrl)
```

산출: $\{\hat\gamma^{(h)}_{DT}\}_{h=1}^{10}$ 프로파일과 구간 간 BH-FDR 보정 p값. 최적 구간의 사후 선택 보고는 순환논법 — 확증은 MP1, MP2는 국소화 그림.

## MF — 실패 발생 (t0 이후 전체 셀 $i$)

$$M_i\sim\mathrm{Bernoulli}(\pi_i),\qquad \operatorname{logit}(\pi_i)=\beta_0+\gamma_DD_d+\gamma_T\tilde t+\gamma_{DT}D_d\tilde t+\gamma_S\tilde s_j+b_{0m}+a_j$$

| 기호 | 코드 | 의미 |
|---|---|---|
| $M_i$ | `miss` (y) | 그 셀에서 평가 실패(NaN)가 발생했는가 (0/1) |
| $\pi_i$ | — | 실패 확률. 행이 시행 1회라 Bernoulli, `weights` 없음 |
| $\gamma_D$ | `directionreverse` | 방향의 붕괴율 차 — 커리큘럼의 학습 안정성 효과 |
| $\gamma_T$ | `time_c` | 후반부 누적 불안정 |
| $\gamma_{DT}$ | `directionreverse:time_c` | 방향별 불안정 누적 속도 차 |
| $\gamma_S$ | `stage_c` | 평가 대상 태스크 stage의 효과 |
| $b_{0m},\ a_j$ | `(1 \| model)`, `(1 \| task)` | 모델·태스크 고유의 실패 성향 |

```r
mF <- glmer(miss ~ direction * time_c + stage_c + (1 | model) + (1 | task),
            family = binomial, data = dpost, control = ctrl)   # weights 없음 (n = 1)
```

검정: $\gamma_D,\gamma_{DT}$. 계수당 사건 10개(EPV) 미달 시 정확검정(`fisher.test`)으로 강등. 전제: 실패가 모델 유발(출력 붕괴)일 때만 과학적 의미가 있음.
