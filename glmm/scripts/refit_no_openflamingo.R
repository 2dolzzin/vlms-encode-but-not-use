# refit_no_openflamingo.R
# 1onr Minor 2: 4개 주 모형(배터리 inference/probe, held-out inference/probe)을 OpenFlamingo
# 행 제외 후 원본과 동일 스펙으로 재적합. 먼저 전체 데이터로 원본 재현(README.md 수치와 대조해
# 파이프라인 검증) 후, OpenFlamingo 제외판을 적합해 D/D×t 를 나란히 비교한다.

suppressMessages(library(lmerTest))

DIR <- "/home/hail/vlm3/glmm/csv_4models"

d   <- read.csv(file.path(DIR, "d.csv"));   d$model <- factor(d$model);   d$task <- factor(d$task)
dp  <- read.csv(file.path(DIR, "dp.csv"));  dp$model <- factor(dp$model); dp$task <- factor(dp$task); dp$cell <- factor(dp$cell)
db  <- read.csv(file.path(DIR, "db.csv"));  db$model <- factor(db$model); db$task <- factor(db$task)
dbp <- read.csv(file.path(DIR, "dbp.csv")); dbp$model <- factor(dbp$model); dbp$task <- factor(dbp$task); dbp$cell <- factor(dbp$cell)

extract_dt <- function(fit, tag) {
  s <- as.data.frame(summary(fit)$coefficients)
  s$term <- rownames(s)
  rows <- s[s$term %in% c("direction", "direction:time_c"), c("term", "Estimate", "Std. Error", "df", "t value", "Pr(>|t|)")]
  names(rows) <- c("term", "est", "se", "df", "t", "p")
  rows$fit <- tag
  rows
}

fit_battery_inf <- function(data) {
  lmer(delta ~ base_wi + base_bw + trained_yet + recency + direction * time_c
       + (1 | model) + (1 | task) + (1 | model:task), data = data)
}
fit_battery_prob <- function(data) {
  lmer(delta ~ base_wi + base_bw + trained_yet + recency + direction * time_c
       + depth_num + is_proj
       + (1 | model) + (1 | task) + (1 | model:task) + (1 | cell), data = data)
}
fit_heldout_inf <- function(data) {
  lmer(delta ~ base_wi + base_bw + direction * time_c
       + (1 | model) + (1 | task) + (1 | model:task), data = data)
}
fit_heldout_prob <- function(data) {
  lmer(delta ~ base_wi + base_bw + direction * time_c + depth_num + is_proj
       + (1 | model) + (1 | task) + (1 | model:task) + (1 | cell), data = data)
}

cat("========== (A) 전체 데이터 재현 (README.md 원본 수치와 대조용 sanity check) ==========\n\n")

cat("-- battery inference (전체) --\n")
m_full <- fit_battery_inf(d)
print(round(summary(m_full)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n-- battery probe (전체) --\n")
mp_full <- fit_battery_prob(dp)
print(round(summary(mp_full)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n-- held-out inference (전체) --\n")
mb_full <- fit_heldout_inf(db)
print(round(summary(mb_full)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n-- held-out probe (전체) --\n")
mq_full <- fit_heldout_prob(dbp)
print(round(summary(mq_full)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n\n========== (B) OpenFlamingo 제외 재적합 ==========\n\n")

d_x   <- droplevels(subset(d,   model != "openflamingo"))
dp_x  <- droplevels(subset(dp,  model != "openflamingo"))
db_x  <- droplevels(subset(db,  model != "openflamingo"))
dbp_x <- droplevels(subset(dbp, model != "openflamingo"))

cat(sprintf("d: %d -> %d행 | dp: %d -> %d행 | db: %d -> %d행 | dbp: %d -> %d행\n\n",
            nrow(d), nrow(d_x), nrow(dp), nrow(dp_x), nrow(db), nrow(db_x), nrow(dbp), nrow(dbp_x)))

cat("-- battery inference (OF 제외) --\n")
m_x <- fit_battery_inf(d_x)
print(round(summary(m_x)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n-- battery probe (OF 제외) --\n")
mp_x <- fit_battery_prob(dp_x)
print(round(summary(mp_x)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n-- held-out inference (OF 제외) --\n")
mb_x <- fit_heldout_inf(db_x)
print(round(summary(mb_x)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n-- held-out probe (OF 제외) --\n")
mq_x <- fit_heldout_prob(dbp_x)
print(round(summary(mq_x)$coefficients[c("direction","direction:time_c"),], 5))

cat("\n\n========== 정리 CSV 저장 ==========\n")
out <- rbind(
  cbind(extract_dt(m_full,  "battery_inf_full"),   openflamingo="included"),
  cbind(extract_dt(m_x,     "battery_inf_noOF"),   openflamingo="excluded"),
  cbind(extract_dt(mp_full, "battery_prob_full"),  openflamingo="included"),
  cbind(extract_dt(mp_x,    "battery_prob_noOF"),  openflamingo="excluded"),
  cbind(extract_dt(mb_full, "heldout_inf_full"),   openflamingo="included"),
  cbind(extract_dt(mb_x,    "heldout_inf_noOF"),   openflamingo="excluded"),
  cbind(extract_dt(mq_full, "heldout_prob_full"),  openflamingo="included"),
  cbind(extract_dt(mq_x,    "heldout_prob_noOF"),  openflamingo="excluded")
)
write.csv(out, "/home/hail/vlm3/glmm/scripts/../csv_4models/refit_no_openflamingo_results.csv", row.names = FALSE)
cat("saved refit_no_openflamingo_results.csv\n")
