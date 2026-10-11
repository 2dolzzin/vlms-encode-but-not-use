suppressMessages({library(lmerTest); library(emmeans)})
d <- read.csv(Sys.getenv("DBP"), stringsAsFactors=FALSE)
d$model<-factor(d$model); d$task<-factor(d$task); d$cell<-factor(d$cell)
d$direction<-d$direction_c
bt <- subset(d, type=="BlindTest" & (depth_num > 0 | is_proj == 1)); bt$task<-droplevels(bt$task)
cat("n =", nrow(bt), "\n\n")

cat("########## model / task 별 반응 이질성 (probing) ##########\n")
m2 <- lmer(delta ~ base_wi + model + task + direction*time_c + depth_num + is_proj
           + model:time_c + task:time_c + model:depth_num + task:depth_num
           + (1|model:task) + (1|cell), data=bt)
print(anova(m2, type=3)[c("time_c","depth_num","direction:time_c",
                          "model:time_c","task:time_c","model:depth_num","task:depth_num"),])

cat("\n########## cell 별 기울기 랜덤효과가 필요한가 ##########\n")
a <- lmer(delta ~ base_wi + model + task + direction*time_c + depth_num + is_proj
          + model:time_c + task:time_c + model:depth_num + task:depth_num
          + (1|model:task) + (1|cell), data=bt, REML=FALSE)
b <- lmer(delta ~ base_wi + model + task + direction*time_c + depth_num + is_proj
          + model:time_c + task:time_c + model:depth_num + task:depth_num
          + (1 + time_c || model:task) + (1|cell), data=bt, REML=FALSE)
print(anova(a,b))

cat("\n########## 모델별 probe 기울기 (time_c) ##########\n")
et <- as.data.frame(summary(emtrends(m2, ~model, var="time_c"), infer=c(TRUE,TRUE)))
et$p_FDR <- p.adjust(et$p.value,"BH")
print(data.frame(model=et$model, slope=round(et$time_c.trend,5),
                 lo=round(et$lower.CL,5), hi=round(et$upper.CL,5),
                 p=round(et$p.value,4), p_FDR=round(et$p_FDR,4)))

cat("\n########## 모델별 depth 기울기 ##########\n")
ed <- as.data.frame(summary(emtrends(m2, ~model, var="depth_num"), infer=c(TRUE,TRUE)))
ed$p_FDR <- p.adjust(ed$p.value,"BH")
print(data.frame(model=ed$model, depth_slope=round(ed$depth_num.trend,5),
                 lo=round(ed$lower.CL,5), hi=round(ed$upper.CL,5),
                 p=round(ed$p.value,4), p_FDR=round(ed$p_FDR,4)))

cat("\n########## task 별 probe 기울기 (time_c) ##########\n")
et2 <- as.data.frame(summary(emtrends(m2, ~task, var="time_c"), infer=c(TRUE,TRUE)))
et2$p_FDR <- p.adjust(et2$p.value,"BH")
print(data.frame(task=et2$task, slope=round(et2$time_c.trend,5),
                 lo=round(et2$lower.CL,5), hi=round(et2$upper.CL,5),
                 p=round(et2$p.value,4), p_FDR=round(et2$p_FDR,4)))
