library(lmerTest)
d <- read.csv(Sys.getenv("DBP"), stringsAsFactors = FALSE)
d$model <- factor(d$model); d$task <- factor(d$task); d$cell <- factor(d$cell)
d$direction <- d$direction_c

dump <- function(fit, tag, n) {
  s <- as.data.frame(summary(fit)$coefficients)
  s$term <- rownames(s); s$fit <- tag; s$n <- n
  names(s)[1:5] <- c("est","se","df","t","p"); s
}
bt  <- subset(d, type == "BlindTest")
bt2 <- subset(bt, depth_num > 0 | is_proj == 1)
mv  <- subset(d, type == "MMVP")

f1 <- lmer(delta ~ base_wi + base_bw + direction*time_c + depth_num + is_proj
           + (1|model)+(1|task)+(1|model:task)+(1|cell), data=bt)
f2 <- lmer(delta ~ base_wi + base_bw + direction*time_c + depth_num + is_proj
           + (1|model)+(1|task)+(1|model:task)+(1|cell), data=bt2)
f3 <- lmer(delta ~ base_wi + direction*time_c + depth_num + is_proj
           + (1|model)+(1|cell), data=mv)
out <- rbind(dump(f1,"blindtest",nrow(bt)), dump(f2,"blindtest_nodepth0",nrow(bt2)),
             dump(f3,"mmvp",nrow(mv)))
write.csv(out[,c("fit","n","term","est","se","df","t","p")], Sys.getenv("OUT"), row.names=FALSE)
cat("written\n")
