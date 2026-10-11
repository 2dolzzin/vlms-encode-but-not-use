suppressMessages({library(lmerTest); library(emmeans)})
w <- read.csv(Sys.getenv("W"), stringsAsFactors=FALSE)
w$model<-factor(w$model); w$task<-factor(w$task)
w$built <- w$maxlm_probe - w$proj_probe      # LM 안에서 새로 생긴 몫
w$gap   <- w$final_probe - w$inf             # representation-utilization gap

cat("########## task 별 'LM 안에서 만들어진 몫' (maxLM - projector) ##########\n")
f <- lmer(built ~ 0 + task + (1|model), data=w)
e <- as.data.frame(summary(emmeans(f, ~task), infer=c(TRUE,TRUE)))
e$p_FDR <- p.adjust(e$p.value,"BH")
o <- order(-e$emmean)
print(data.frame(task=e$task, built=round(e$emmean,3), lo=round(e$lower.CL,3),
                 hi=round(e$upper.CL,3), p_FDR=signif(e$p_FDR,3))[o,], row.names=FALSE)

cat("\n########## inference 는 projector 를 따라가나 LM 최종층을 따라가나 ##########\n")
for (k in c("battery","benchmark")) {
  s <- subset(w, kind==k)
  m <- lmer(inf ~ proj_probe + final_probe + (1|model) + (1|task), data=s)
  cat("\n--", k, "(n =", nrow(s), ") --\n")
  print(round(summary(m)$coefficients,4))
}

cat("\n########## utilization gap (final probe - inference) ##########\n")
g <- lmer(gap ~ 0 + kind + (1|model) + (1|task), data=w)
print(round(as.data.frame(summary(emmeans(g, ~kind), infer=c(TRUE,TRUE)))[,c(1,2,5,6,8)],4))
