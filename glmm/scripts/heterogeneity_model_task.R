suppressMessages({library(lmerTest); library(emmeans)})
d <- read.csv(Sys.getenv("DB"), stringsAsFactors=FALSE)
d$model<-factor(d$model); d$task<-factor(d$task); d$direction<-d$direction_c
bt <- subset(d, type=="BlindTest"); bt$task<-droplevels(bt$task)

cat("########## 학습 반응(기울기)의 이질성은 어디에 있나 ##########\n")
cat("주: model 을 고정효과로 넣으면 base_bw(모델 수준 상수)와 (1|model) 은 공선이라 뺀다.\n\n")

m0 <- lmer(delta ~ base_wi + base_bw + direction*time_c
           + (1|model)+(1|task)+(1|model:task), data=bt, REML=FALSE)          # 현재 §4
m1 <- lmer(delta ~ base_wi + base_bw + direction*time_c + task:time_c
           + (1|model)+(1|task)+(1|model:task), data=bt, REML=FALSE)          # + task 별 기울기
m2 <- lmer(delta ~ base_wi + model + task + direction*time_c + model:time_c + task:time_c
           + (1|model:task), data=bt, REML=FALSE)                              # + model 별 기울기
m3 <- lmer(delta ~ base_wi + model + task + direction*time_c + model:time_c + task:time_c
           + (1 + time_c || model:task), data=bt, REML=FALSE)                  # + cell 별 기울기(랜덤)

cat("--- LRT ---\n")
print(anova(m0, m1))
cat("\n(m2 는 고정효과 구조가 달라 m1 과 직접 LRT 불가; AIC 로 비교)\n")
cat(sprintf("AIC  m0 %.1f | m1 %.1f | m2 %.1f | m3 %.1f\n", AIC(m0),AIC(m1),AIC(m2),AIC(m3)))
cat("\n--- m2 안에서 model:time_c 와 task:time_c 의 III종 검정 ---\n")
print(anova(m2, type=3, ddf="Satterthwaite")[c("time_c","model:time_c","task:time_c","direction:time_c"),])
cat("\n--- m3: cell 별 기울기 랜덤효과가 필요한가 (m2 vs m3) ---\n")
print(anova(m2, m3))

cat("\n########## 모델별 학습 기울기 (m2, emtrends) ##########\n")
m2r <- lmer(delta ~ base_wi + model + task + direction*time_c + model:time_c + task:time_c
            + (1|model:task), data=bt)
et <- as.data.frame(summary(emtrends(m2r, ~model, var="time_c"), infer=c(TRUE,TRUE)))
et$p_FDR <- p.adjust(et$p.value,"BH")
print(data.frame(model=et$model, slope=round(et$time_c.trend,4),
                 lo=round(et$lower.CL,4), hi=round(et$upper.CL,4),
                 p=round(et$p.value,4), p_FDR=round(et$p_FDR,4)))

cat("\n########## 모델 x task 셀 기울기 상위/하위 (m3 BLUP) ##########\n")
m3r <- lmer(delta ~ base_wi + model + task + direction*time_c + model:time_c + task:time_c
            + (1 + time_c || model:task), data=bt)
vc <- as.data.frame(VarCorr(m3r)); print(vc[,c("grp","var1","sdcor")])
re <- ranef(m3r)$`model:task`
sl <- re[order(-re$time_c), , drop=FALSE]
cat("\n기울기 편차 상위 5 / 하위 5 (cell):\n")
print(round(head(sl["time_c"],5),4)); print(round(tail(sl["time_c"],5),4))
