suppressMessages({library(lmerTest); library(emmeans)})
d <- read.csv(Sys.getenv("DB"), stringsAsFactors=FALSE)
d$model<-factor(d$model); d$direction_f<-factor(d$direction)
sw <- subset(d, task=="subway_connections")

cat("##### subway_connections: 모델별 time_c 기울기 #####\n")
f <- lm(delta ~ model*time_c + direction_c*time_c, data=sw)
et <- as.data.frame(summary(emtrends(f, ~model, var="time_c"), infer=c(TRUE,TRUE)))
et$p_FDR <- p.adjust(et$p.value,"BH")
print(data.frame(model=et$model, slope=round(et$time_c.trend,4),
                 lo=round(et$lower.CL,4), hi=round(et$upper.CL,4),
                 p=round(et$p.value,4), p_FDR=round(et$p_FDR,4)))

cat("\n##### 'Base 를 넘었는가' (delta > 0) 를 모델별로 #####\n")
for (m in levels(sw$model)) {
  s <- subset(sw, model==m); tt <- t.test(s$delta)
  cat(sprintf("%-13s mean delta %+.3f  CI [%+.3f, %+.3f]  p=%.4f  n=%d\n",
              m, mean(s$delta), tt$conf.int[1], tt$conf.int[2], tt$p.value, nrow(s)))
}

cat("\n##### color stage 학습 여부(step) vs 선형 시간, 어느 쪽이 맞나 (vila_3b) #####\n")
v <- subset(sw, model=="vila_3b")
m_lin  <- lm(delta ~ time_c + direction_c, data=v)
m_step <- lm(delta ~ source_trained_broad + direction_c, data=v)
m_both <- lm(delta ~ time_c + source_trained_broad + direction_c, data=v)
cat("AIC  linear:", round(AIC(m_lin),2), " step(S2 trained):", round(AIC(m_step),2),
    " both:", round(AIC(m_both),2), "\n")
print(round(summary(m_both)$coefficients,4))

cat("\n##### 색 관련 두 task 만, 모델별 source_trained 효과 #####\n")
col <- subset(d, task %in% c("subway_connections","olympic_circles"))
fc <- lmer(delta ~ source_trained_broad*model + direction_c + (1|task), data=col)
print(round(summary(emmeans(fc, ~source_trained_broad|model,
                            at=list(source_trained_broad=c(0,1))))[,c(1,2,3,4)],4))
cat("\n대비 (trained - not trained) by model:\n")
print(summary(contrast(emmeans(fc, ~source_trained_broad|model,
                               at=list(source_trained_broad=c(0,1))), "revpairwise"),
              infer=c(TRUE,TRUE)))
