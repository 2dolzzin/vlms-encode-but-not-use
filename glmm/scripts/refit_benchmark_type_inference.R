library(lmerTest)
d <- read.csv(Sys.getenv("DB"), stringsAsFactors = FALSE)
d$model <- factor(d$model); d$task <- factor(d$task)
d$direction <- d$direction_c
d$type <- factor(d$type, levels = c("BlindTest","MMVP"))

dump <- function(fit, tag, n) {
  s <- as.data.frame(summary(fit)$coefficients); s$term <- rownames(s)
  names(s)[1:5] <- c("est","se","df","t","p"); s$fit <- tag; s$n <- n; s
}
bt <- subset(d, type=="BlindTest"); mv <- subset(d, type=="MMVP")

cat("\n##### BlindTest only (n =", nrow(bt), ")\n")
f1 <- lmer(delta ~ base_wi + base_bw + direction*time_c
           + (1|model)+(1|task)+(1|model:task), data=bt)
print(round(summary(f1)$coefficients,5))

cat("\n##### MMVP only, base_wi 포함 (n =", nrow(mv), ")  [base_wi 는 모델 수준 상수라 (1|model) 과 중복]\n")
f2 <- lmer(delta ~ base_wi + direction*time_c + (1|model), data=mv)
print(round(summary(f2)$coefficients,5))

cat("\n##### MMVP only, base 항 제외\n")
f3 <- lmer(delta ~ direction*time_c + (1|model), data=mv)
print(round(summary(f3)$coefficients,5))

cat("\n##### 전체 + type 상호작용\n")
f4 <- lmer(delta ~ base_wi + base_bw + type*(direction*time_c)
           + (1|model)+(1|task)+(1|model:task), data=d)
print(round(summary(f4)$coefficients,5))

out <- rbind(dump(f1,"blindtest",nrow(bt)), dump(f2,"mmvp_with_base",nrow(mv)),
             dump(f3,"mmvp",nrow(mv)), dump(f4,"type_interaction",nrow(d)))
write.csv(out[,c("fit","n","term","est","se","df","t","p")], Sys.getenv("OUT"), row.names=FALSE)
