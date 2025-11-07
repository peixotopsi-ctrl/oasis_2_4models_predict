# ============================================================
# 🧠 OASIS-2 Longitudinal MRI – Conversion Prediction
# Version: No ASF, No MMSE SD, Four Models (GLM, RF, SVM, kNN)
# Author: Bruno Peixoto, 2025
# ============================================================

start_time <- Sys.time()

# ---- 0️⃣ Load Packages ----
pkgs <- c("readxl", "dplyr", "ggplot2", "caret", "pROC",
          "openxlsx", "tidyr", "tibble", "randomForest", "e1071")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[, "Package"])]
if (length(new_pkgs) > 0) install.packages(new_pkgs)
lapply(pkgs, require, character.only = TRUE)

set.seed(123)

# ---- 1️⃣ Load OASIS-2 Data ----
data_path <- "C:/Users/35196/Desktop/oasis_predict/oasis_longitudinal_demographics.xlsx"
if (!file.exists(data_path)) stop("❌ Data file not found at: ", data_path)
Oasis <- read_excel(data_path)

# ---- 2️⃣ Harmonize Column Names ----
Oasis <- Oasis %>%
  rename(
    id    = `Subject ID`,
    group = Group,
    delay = `MR Delay`,
    sex   = `M/F`,
    idade = Age,
    educ  = EDUC,
    ses   = SES,
    mmse  = MMSE,
    cdr   = CDR,
    tiv   = eTIV,
    wbv   = nWBV,
    asf   = ASF
  )

# ---- 3️⃣ Impute Missing Values (Group Median) ----
Oasis <- Oasis %>%
  group_by(group) %>%
  mutate(
    ses   = ifelse(is.na(ses),  median(ses,  na.rm = TRUE), ses),
    mmse  = ifelse(is.na(mmse), median(mmse, na.rm = TRUE), mmse),
    educ  = ifelse(is.na(educ), median(educ, na.rm = TRUE), educ),
    idade = ifelse(is.na(idade), median(idade, na.rm = TRUE), idade)
  ) %>%
  ungroup()

# ---- 4️⃣ Longitudinal Features ----
Oasis <- Oasis %>%
  arrange(id, delay) %>%
  group_by(id) %>%
  mutate(
    baseline_mmse = mmse[delay == 0][1],
    years_since_baseline = delay / 12
  ) %>%
  ungroup()

# ---- 5️⃣ Aggregate Per Participant ----
features <- Oasis %>%
  group_by(id, group, sex) %>%
  summarise(
    n_visits      = n(),
    baseline_mmse = first(baseline_mmse),
    mean_mmse     = mean(mmse, na.rm = TRUE),
    mmse_slope    = tryCatch(coef(lm(mmse ~ years_since_baseline))[2], error = function(e) 0),
    mean_wbv      = mean(wbv, na.rm = TRUE),
    wbv_slope     = tryCatch(coef(lm(wbv ~ years_since_baseline))[2], error = function(e) 0),
    mean_tiv      = mean(tiv, na.rm = TRUE),
    tiv_slope     = tryCatch(coef(lm(tiv ~ years_since_baseline))[2], error = function(e) 0),
    idade         = first(idade),
    educ          = first(educ),
    ses           = first(ses),
    .groups = "drop"
  )

# ---- 6️⃣ Define Groups and Outcome ----
features <- features %>%
  mutate(
    group_label = case_when(
      group == 1 ~ "Nondemented",
      group == 2 ~ "Demented",
      group == 3 ~ "Converted",
      TRUE ~ NA_character_
    ),
    converted = case_when(
      group == 3 ~ "Yes",
      group == 1 ~ "No",
      TRUE ~ NA_character_
    )
  ) %>%
  drop_na(converted) %>%
  mutate(converted = factor(converted, levels = c("No", "Yes")))

cat("\n✅ Conversion outcome created. Group counts:\n")
print(table(features$group_label, features$converted))

# ---- 7️⃣ Train/Test Split ----
set.seed(123)
train_idx <- createDataPartition(features$converted, p = 0.7, list = FALSE)
train_data <- features[train_idx, ]
test_data  <- features[-train_idx, ]

# ---- 8️⃣ Impute Remaining NAs ----
train_data <- train_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))
test_data <- test_data %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm = TRUE), .)))

# ---- 9️⃣ Balance Training Data ----
set.seed(123)
train_bal <- train_data %>%
  group_by(converted) %>%
  slice_sample(n = max(table(train_data$converted)), replace = TRUE) %>%
  ungroup()

# ---- 🔟 Train Models ----
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

form <- converted ~ idade + educ + ses +
  baseline_mmse + mean_mmse + mmse_slope +
  mean_wbv + wbv_slope + mean_tiv + tiv_slope +
  n_visits

# GLM
set.seed(123)
glm_mod <- train(form, data = train_bal, method = "glm",
                 family = "binomial", metric = "ROC", trControl = ctrl)

# RF
set.seed(123)
rf_mod  <- train(form, data = train_bal, method = "rf",
                 metric = "ROC", trControl = ctrl,
                 tuneLength = 3, ntree = 500, importance = TRUE)

# SVM
set.seed(123)
svm_mod <- train(form, data = train_bal, method = "svmRadial",
                 metric = "ROC", trControl = ctrl, tuneLength = 5)

# kNN
set.seed(123)
knn_mod <- train(form, data = train_bal, method = "knn",
                 metric = "ROC", trControl = ctrl, tuneLength = 10)

cat("\n✅ Models trained: GLM, RF, SVM, kNN\n")

# ---- 1️⃣1️⃣ Evaluate Models ----
pred_prob <- function(mod, data) {
  predict(mod, newdata = data, type = "prob")[, "Yes"]
}
pred_class <- function(mod, data) {
  predict(mod, newdata = data)
}

probs_glm <- pred_prob(glm_mod, test_data)
probs_rf  <- pred_prob(rf_mod,  test_data)
probs_svm <- pred_prob(svm_mod, test_data)
probs_knn <- pred_prob(knn_mod, test_data)

roc_glm <- roc(test_data$converted, probs_glm)
roc_rf  <- roc(test_data$converted, probs_rf)
roc_svm <- roc(test_data$converted, probs_svm)
roc_knn <- roc(test_data$converted, probs_knn)

auc_values <- c(
  GLM = auc(roc_glm),
  RF  = auc(roc_rf),
  SVM = auc(roc_svm),
  kNN = auc(roc_knn)
)

cat("\n✅ AUC values:\n")
print(round(auc_values, 3))

# ---- 1️⃣2️⃣ Metrics ----
glm_pred <- pred_class(glm_mod, test_data)
rf_pred  <- pred_class(rf_mod,  test_data)
svm_pred <- pred_class(svm_mod, test_data)
knn_pred <- pred_class(knn_mod, test_data)

f1 <- function(pred, truth) caret::F_meas(pred, truth, relevant = "Yes")

metrics <- data.frame(
  Model = c("GLM", "RF", "SVM", "kNN"),
  AUC = round(auc_values, 3),
  Accuracy = c(mean(glm_pred == test_data$converted),
               mean(rf_pred  == test_data$converted),
               mean(svm_pred == test_data$converted),
               mean(knn_pred == test_data$converted)),
  Sensitivity = c(sensitivity(glm_pred, test_data$converted, positive = "Yes"),
                  sensitivity(rf_pred,  test_data$converted, positive = "Yes"),
                  sensitivity(svm_pred, test_data$converted, positive = "Yes"),
                  sensitivity(knn_pred, test_data$converted, positive = "Yes")),
  Specificity = c(specificity(glm_pred, test_data$converted, negative = "No"),
                  specificity(rf_pred,  test_data$converted, negative = "No"),
                  specificity(svm_pred, test_data$converted, negative = "No"),
                  specificity(knn_pred, test_data$converted, negative = "No")),
  F1 = c(f1(glm_pred, test_data$converted),
         f1(rf_pred,  test_data$converted),
         f1(svm_pred, test_data$converted),
         f1(knn_pred, test_data$converted))
)

cat("\n✅ Test-set metrics:\n")
print(metrics)

# ---- 1️⃣3️⃣ Save Outputs ----
dir.create("OASIS2_results_noASF_noSD", showWarnings = FALSE)
write.xlsx(metrics, "OASIS2_results_noASF_noSD/model_metrics_noASF_noSD.xlsx")

save(Oasis, features, train_data, test_data, train_bal,
     glm_mod, rf_mod, svm_mod, knn_mod,
     roc_glm, roc_rf, roc_svm, roc_knn,
     metrics,
     file = "OASIS2_results_noASF_noSD/OASIS2_models_noASF_noSD.RData")

end_time <- Sys.time()
cat("\n✅ Pipeline completed successfully!\n")
cat("Start:", as.character(start_time), "\n")
cat("End:  ", as.character(end_time), "\n")
