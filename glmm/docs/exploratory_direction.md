# 탐색적 분석 — `direction` 효과 찾기

`GLMM_analysis.ipynb` 의 본 분석에서 `direction`(forward vs reverse)은 **네 분석 모두에서
널이었다** (R² 기여 0.002~0.005, 주 검정 p > 0.13). 그러나 두 커리큘럼은 결국 같은 5개 세션을
모두 돌기 때문에 최종 평균이 같은 것은 당연하고, 방향이 만드는 차이는 평균이 아니라 **다른 축에
숨어 있을 수 있다.** 이 문서는 그 축을 탐색한 절차와 결과를 남긴다.

> **성격:** 이것은 사전등록된 확증 분석이 아니라 **탐색적 분석**이다. 여러 가설을 훑고
> 사후에 유의한 것을 골랐으므로, 아래 결과는 "이런 방향으로 후속 확증을 해볼 만하다"는 단서이지
> 그 자체로 확증된 효과가 아니다. 노트북 본문에는 이 중 살아남은 하나(`direction × depth`)만
> 기본 모형과 나란히 실었다.

## 방법

본 분석의 최종 모형에 `direction` 과 다른 변수의 **상호작용을 하나씩** 더해, 넣기 전/후를
가능도비 검정(`anova`, ML 적합)으로 비교했다. 총 11개 가설을 세우고, 11개 p값에 대해
**BH-FDR**(`p.adjust(method="BH")`)로 다중비교를 보정했다.

가설 목록:

| # | 데이터 | 추가한 항 | 묻는 것 |
|---|---|---|---|
| H1 | inf 커리큘럼 | `direction : factor(time)` | 방향차가 특정 세션 시점에 몰리는가 |
| H2 | inf 커리큘럼 | `direction : task` | 방향차가 특정 태스크에 몰리는가 |
| H3 | inf 커리큘럼 | `direction : model` | 방향차가 특정 모델에 몰리는가 |
| H4 | inf 커리큘럼 | `direction : trained_yet` | 학습 직후 이득이 방향에 따라 다른가 |
| H5 | inf 커리큘럼 | `direction : recency` | 망각 속도가 방향에 따라 다른가 |
| H6 | inf 커리큘럼 | (잔차분산 F검정) | 한 방향이 더 불안정한가 |
| H7 | prob 커리큘럼 | `direction : depth` | 방향차가 특정 표상 깊이에 몰리는가 |
| H8 | prob 커리큘럼 | `direction : factor(time)` | (표상) 방향차가 특정 시점에 몰리는가 |
| H9 | prob 커리큘럼 | `direction : model` | (표상) 방향차가 특정 모델에 몰리는가 |
| H10 | inf benchmark | `direction : task` | 전이의 방향차가 특정 태스크에 몰리는가 |
| H11 | inf benchmark | `direction : model` | 전이의 방향차가 특정 모델에 몰리는가 |

```r
# 각 가설을 기준 모형에 더해 가능도비 검정, 마지막에 BH-FDR
# (기준 모형 b/bp/bb, FIT/FITP/FITB, r2m 계열은 노트북에서 정의된 것)
res <- list()
add <- function(nm, a, dr) res[[length(res)+1]] <<-
  data.frame(가설=nm, chisq=a$Chisq[2], df=a$Df[2], p=a$`Pr(>Chisq)`[2], dR2m=dr)

b  <- delta ~ (base_wi+base_bw)*trained_yet + recency + direction*time_c   # inf 커리큘럼
bp <- delta ~ base_wi+base_bw+trained_yet+recency+direction*time_c+depth   # prob 커리큘럼
bb <- delta ~ base_wi+base_bw+direction*time_c                            # benchmark

m0 <- FIT(b)                                    # H1~H5
for (x in list(c("H1 dir x factor(time)", ". ~ . + direction:factor(time)"),
               c("H2 dir x task",         ". ~ . + direction:task"),
               c("H3 dir x model",        ". ~ . + direction:model"),
               c("H4 dir x trained_yet",  ". ~ . + direction:trained_yet"),
               c("H5 dir x recency",      ". ~ . + direction:recency"))) {
  m <- FIT(update(b, as.formula(x[2]))); add(x[1], anova(m0, m), r2m(m)-r2m(m0))
}
r <- residuals(m0); v <- var.test(r[d$direction=="forward"], r[d$direction=="reverse"])  # H6
res[[length(res)+1]] <- data.frame(가설="H6 방향별 잔차분산", chisq=NA, df=NA, p=v$p.value, dR2m=NA)

p0 <- FITP(bp)                                  # H7~H9
for (x in list(c("H7 dir x depth",        ". ~ . + direction:depth"),
               c("H8 dir x factor(time)", ". ~ . + direction:factor(time)"),
               c("H9 dir x model",        ". ~ . + direction:model"))) {
  m <- FITP(update(bp, as.formula(x[2]))); add(x[1], anova(p0, m), r2mP(m)-r2mP(p0))
}
b0 <- FITB(bb)                                  # H10~H11
for (x in list(c("H10 dir x task (bench)",  ". ~ . + direction:task"),
               c("H11 dir x model (bench)", ". ~ . + direction:model"))) {
  m <- FITB(update(bb, as.formula(x[2]))); add(x[1], anova(b0, m), r2mB(m)-r2mB(b0))
}
out <- do.call(rbind, res); out$p_FDR <- p.adjust(out$p, "BH")
print(out[order(out$p), ], row.names=FALSE)
```

## 결과

FDR 보정 후 p값 오름차순:

| 가설 | χ² | df | p | ΔR²m | p (FDR) |
|---|---|---|---|---|---|
| **H7 dir × depth** (prob) | 21.55 | 5 | 0.00064 | +0.009 | **0.0054** |
| **H2 dir × task** (inf) | 26.15 | 8 | 0.00099 | +0.113 | **0.0054** |
| **H9 dir × model** (prob) | 20.68 | 7 | 0.00428 | +0.052 | **0.0157** |
| H10 dir × task (bench) | 30.39 | 18 | 0.0338 | +0.101 | 0.093 |
| H3 dir × model (inf) | 13.57 | 7 | 0.0594 | +0.056 | 0.131 |
| H11 dir × model (bench) | 11.94 | 7 | 0.103 | +0.071 | 0.188 |
| H4 dir × trained_yet | 0.77 | 1 | 0.380 | +0.002 | 0.597 |
| H6 방향별 잔차분산 | — | — | 0.478 | — | 0.598 |
| H1 dir × factor(time) | 5.44 | 6 | 0.489 | +0.005 | 0.598 |
| H8 dir × factor(time) | 3.89 | 6 | 0.692 | +0.002 | 0.761 |
| H5 dir × recency | 0.02 | 1 | 0.885 | −0.000 | 0.885 |

세 개가 FDR < 0.05 를 통과했다. 그러나 **통과한 것을 각각 층·태스크·모델 수준으로 쪼개
사후 대비(emmeans, reverse − forward)를 보면 성격이 완전히 다르다.**

### H2 (dir × task) — 탈락: `recency` 의 재매개화

전체 검정은 강하지만(p=0.001, ΔR²=0.113), **개별 태스크 사후 대비 5개 중 유의한 것이 없다**
(최소 p=0.086). 게다가 이 항은 새로운 현상이 아니다 — 태스크별 `recency` 의 방향차가 이미
`(S0 −2.0, S1 −1.0, S2 0, S3 +1.0, S4 +2.0)` 로 결정돼 있어, `direction:task` 는 사실상
**"`recency` 를 선형 하나로 요약한 것이 부족하다"** 는 적합도 신호를 방향 효과로 잘못 읽은 것이다.
방향의 독립적 효과가 아니다.

### H9 (dir × model) — 탈락: 단일 모델, 그것도 가장 부실한

모델별 사후 대비에서 **5개 중 openflamingo 하나만 유의**하다(reverse − forward = +0.094,
p=0.020; 나머지 넷은 |효과| ≤ 0.028, p ≥ 0.45). 그런데 openflamingo 는 결측이 52건으로 몰려
있고 base 가 0.015~0.24 로 바닥인, 데이터가 가장 부실한 모델이다. 5개 대비 중 하나가 p=0.02 면
Bonferroni 기준(0.10)에도 못 미친다. 단일·취약 모델의 효과라 결과로 내세울 수 없다.

### H7 (dir × depth) — 유지: 깊이 기울기가 방향에 따라 다르다

> **부호 정정.** 이 문서의 이전 판은 이 효과의 부호를 반대로 적었다(emmeans 의 `forward−reverse`
> 를 `reverse−forward` 로 잘못 라벨). 실제 방향은 아래와 같다 — **reverse 가 깊은 층을 *더* 키운다.**

본 분석과 노트북은 depth 를 **연속**(`depth_num`, 0~1)으로 쓰므로, 상호작용도 계수 하나로 본다:

**`direction : depth_num` = +0.0172, p = 0.0003** (forward 기준; reverse 의 깊이 기울기가 더 가파름).

깊이별 reverse − forward (연속 모형 emmeans):

| depth_num | reverse − forward | p |
|---|---|---|
| 0.00 | −0.0031 | 0.246 |
| 0.25 | +0.0012 | 0.558 |
| 0.50 | **+0.0055** | **0.004** |
| 0.75 | **+0.0098** | **<0.001** |
| 1.00 | **+0.0141** | **<0.001** |

얕은 층에선 차이가 없다가 깊어질수록 gap 이 단조 증가한다. 범주형 모형도 같은 부호를 준다
(0.75:+0.016, 1.0:+0.015, 둘 다 유의) — 다만 AIC 가 연속(−5639) 쪽이 범주형(−5589)보다 낫다.
`depth` 는 방향과 구조적 관계가 없으므로(H2 와 달리) `recency` 로 환원되지 않는다.

**해석:** 두 방향은 같은 **행동 성능**(inference)에 도달하지만 같은 **표상**에는 도달하지 않는다.
reverse 커리큘럼이 후반부 층 표상을 forward 보다 최심층 기준 약 +1.4%p 더 키운다 — inference
로는 안 보이고 probing 으로만 드러나는 차이다.

## 독립 검증 시도 — benchmark probing 에서는 재현되지 않음

같은 `direction × depth_num` 을 학습 대상이 아닌 benchmark probing 에 적용하면:

**`direction : depth_num` = −0.0022, p = 0.049** — **커리큘럼과 부호가 반대**이고 크기도 1/8,
겨우 경계 유의다. 즉 이 효과는 **직접 학습한 태스크(커리큘럼)에 국한**되고 benchmark 로는 재현되지
않는다. 두 가지로 읽을 수 있다:

1. (긍정적) 효과가 **학습으로 성장한 표상에 대한 것**이라는 방증 — benchmark 는 애초에 표상이
   성장하지 않으니 방향차가 실릴 데도 없다.
2. (부정적) 그렇더라도 **독립 데이터셋에서 재현되지 않았으므로** 확증으로 승격할 수 없다.

## 결론과 한계

- 방향은 **평균 성능**에는 효과가 없다(본 분석). 유일하게 찾은 의미 있는 축은
  **probing 커리큘럼에서 reverse 가 깊은 층 표상을 forward 보다 더 키우는** 것이다(`dir:depth_num`>0).
- 이는 **탐색적 발견**이다: 11개 가설을 훑어 FDR 로 걸렀고, 통과한 셋 중 둘(H2·H9)은 사후 분해에서
  탈락했다. 살아남은 H7 도 benchmark 로 재현되지 않았고, 초기 서술에서 부호를 한 번 틀릴 만큼
  민감하다.
- 최심층 기준 크기가 +1.4%p 로 작고, 모델 5개·커리큘럼 태스크 5개 위의 결과다. 후속으로는 더 촘촘한
  깊이 샘플링, 더 많은 모델, 사전등록된 확증이 필요하다.
- 노트북의 해당 셀은 이 발견 하나를 기본 모형과 나란히 제시할 뿐, 확증으로 서술하지 않는다.
