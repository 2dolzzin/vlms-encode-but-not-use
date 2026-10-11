suppressMessages({library(lmerTest); library(emmeans)})
db  <- read.csv(Sys.getenv("DB"),  stringsAsFactors=FALSE)   # held-out inference
dbp <- read.csv(Sys.getenv("DBP"), stringsAsFactors=FALSE)   # held-out probing
dcu <- read.csv(Sys.getenv("DCU"), stringsAsFactors=FALSE)   # battery inference
for (x in c("db","dbp","dcu")) { d<-get(x); d$model<-factor(d$model); d$task<-factor(d$task)
  d$direction<-d$direction_c; assign(x,d) }
bt  <- subset(db,  type=="BlindTest")
btp <- subset(dbp, type=="BlindTest" & (depth_num>0 | is_proj==1))

cat("########## (i) Base 가 높으면 더 빨리 오르나 — base × time ##########\n")
cat("\n-- held-out inference --\n")
f1 <- lmer(delta ~ (base_wi + base_bw)*time_c + direction*time_c
           + (1|model)+(1|task)+(1|model:task), data=bt)
print(round(summary(f1)$coefficients[c("base_wi","base_bw","time_c",
                                       "base_wi:time_c","base_bw:time_c"),],5))
cat("\n-- held-out probing --\n")
f2 <- lmer(delta ~ (base_wi + base_bw)*time_c + direction*time_c + depth_num + is_proj
           + (1|model)+(1|task)+(1|model:task)+(1|cell), data=btp)
print(round(summary(f2)$coefficients[c("base_wi","base_bw","time_c",
                                       "base_wi:time_c","base_bw:time_c"),],5))

cat("\n########## (ii) battery 를 잘 배운 모델이 전이도 잘하나 ##########\n")
cat("\n-- 모델별 battery 학습 효과 (trained_yet) --\n")
b <- lmer(delta ~ base_wi + base_bw + model*trained_yet + recency + direction*time_c
          + (1|task)+(1|model:task), data=dcu)
eb <- as.data.frame(summary(emtrends(b, ~model, var="trained_yet"), infer=c(TRUE,TRUE)))
print(data.frame(model=eb$model, battery_gain=round(eb$trained_yet.trend,4),
                 lo=round(eb$lower.CL,4), hi=round(eb$upper.CL,4)))

cat("\n-- 모델별 held-out inference 기울기 --\n")
mi <- lmer(delta ~ base_wi + model + task + direction*time_c + model:time_c + task:time_c
           + (1|model:task), data=bt)
ei <- as.data.frame(summary(emtrends(mi, ~model, var="time_c")))
print(data.frame(model=ei$model, transfer_slope=round(ei$time_c.trend,5)))

cat("\n-- 상관 (n=5 모델) --\n")
m <- merge(data.frame(model=eb$model, gain=eb$trained_yet.trend),
           data.frame(model=ei$model, slope=ei$time_c.trend))
bb <- aggregate(base_bw ~ model, data=bt, FUN=mean)
m <- merge(m, bb)
print(round(m[,c("gain","slope","base_bw")],4)); print(m$model)
cat("\nbattery gain  vs  transfer slope : Pearson r =", round(cor(m$gain,m$slope),3),
    " Spearman =", round(cor(m$gain,m$slope,method="spearman"),3), "\n")
cat("benchmark Base vs transfer slope : Pearson r =", round(cor(m$base_bw,m$slope),3),
    " Spearman =", round(cor(m$base_bw,m$slope,method="spearman"),3), "\n")
ct <- cor.test(m$gain, m$slope); cat("gain~slope p =", round(ct$p.value,3), "\n")
