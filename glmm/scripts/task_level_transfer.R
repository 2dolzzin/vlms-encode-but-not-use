suppressMessages({library(lmerTest); library(emmeans)})
d <- read.csv(Sys.getenv("DB"), stringsAsFactors=FALSE)
d$model<-factor(d$model); d$task<-factor(d$task); d$direction<-d$direction_c
bt <- subset(d, type=="BlindTest"); bt$task <- droplevels(bt$task)

cat("########## Design B: task 별 기울기가 다른가 ##########\n")
f0 <- lmer(delta ~ base_wi + base_bw + direction*time_c + task + (1|model)+(1|model:task),
           data=bt, REML=FALSE)
f1 <- lmer(delta ~ base_wi + base_bw + direction*time_c + task*time_c + (1|model)+(1|model:task),
           data=bt, REML=FALSE)
print(anova(f0,f1))

cat("\n--- task 별 time_c 기울기 (emtrends, BH-FDR) ---\n")
f1r <- lmer(delta ~ base_wi + base_bw + direction*time_c + task*time_c + (1|model)+(1|model:task), data=bt)
et <- as.data.frame(summary(emtrends(f1r, ~task, var="time_c"), infer=c(TRUE,TRUE)))
et$p_FDR <- p.adjust(et$p.value, "BH")
print(round_df <- data.frame(task=et$task, slope=round(et$time_c.trend,5),
                             lo=round(et$lower.CL,5), hi=round(et$upper.CL,5),
                             p=round(et$p.value,4), p_FDR=round(et$p_FDR,4)))

cat("\n########## Design C: source stage 가 학습되면 오르는가 ##########\n")
cat("\n-- strict mapping (circled_letter <- S0) --\n")
c1 <- lmer(delta ~ base_wi + base_bw + source_trained + source_recency + direction*time_c
           + (1|model)+(1|task)+(1|model:task), data=bt)
print(round(summary(c1)$coefficients,5))
cat("\n[확인] circled_letter 안에서 source_trained 와 direction 의 상관: ",
    round(cor(subset(bt, task=="circled_letter")$source_trained,
              subset(bt, task=="circled_letter")$direction),3), "\n")

cat("\n-- broad mapping (+ olympic_circles, subway_connections <- S2) --\n")
c2 <- lmer(delta ~ base_wi + base_bw + source_trained_broad + source_recency_broad + direction*time_c
           + (1|model)+(1|task)+(1|model:task), data=bt)
print(round(summary(c2)$coefficients,5))
mapped <- subset(bt, task %in% c("circled_letter","olympic_circles","subway_connections"))
cat("\n[확인] broad mapped task 안에서 source_trained_broad ~ direction 상관: ",
    round(cor(mapped$source_trained_broad, mapped$direction),3), "\n")

cat("\n-- mapped task 만 (n =", nrow(mapped), ") --\n")
c3 <- lmer(delta ~ base_wi + source_trained_broad + direction*time_c + (1|model)+(1|task), data=mapped)
print(round(summary(c3)$coefficients,5))
