library(tidyverse)
library(tidymodels)

# We'll be using this library, so install it if you don't have it:
library(themis)

# Set your own script to work in the right local directory:
setwd('C:\\Users\\dwrig\\Documents\\GitHub\\is-555-10-modeling-pipeline-2-dan-wright-1')


### 1. fraud, fraud_split, fraud_training, fraud_testing
fraud_raw <- read_csv('https://www.dropbox.com/scl/fi/c210k3qguq83gsl8uz8ae/fraud.csv?rlkey=u4lx6k53m0a8untxehuy5k000&dl=1')
set.seed(42)

fraud <- fraud_raw %>% mutate(is_fraud = as.factor(is_fraud))
fraud_split <- initial_split(fraud, strata = is_fraud)
fraud_training <- training(fraud_split)
fraud_testing <- testing(fraud_split)


### 2. fraud_rec_1, fraud_rec_2, fraud_rec_3
fraud_rec_1 <- recipe(is_fraud ~ ., data = fraud_training)

fraud_rec_2 <- recipe(is_fraud ~ ., data = fraud_training) %>% 
  step_YeoJohnson(amount) %>% 
  step_normalize(amount) %>% 
  step_downsample(is_fraud)

fraud_rec_3 <- recipe(is_fraud ~ ., data = fraud_training) %>% 
  step_range(amount , min = 0.01, max = 1) %>% 
  step_BoxCox(amount) %>% 
  step_normalize(amount) %>% 
  step_upsample(is_fraud)


### 3. peek_1, peek_2, peek_3, plot_1, plot_2
peek_1 <- fraud_rec_1 %>% 
  prep() %>% juice() %>% mutate(recipe = 'fraud_rec_1')
  
peek_2 <- fraud_rec_2 %>%
  prep() %>% juice() %>% mutate(recipe = 'fraud_rec_2')

peek_3 <- fraud_rec_3 %>%
  prep() %>% juice() %>% mutate(recipe = 'fraud_rec_3')

fraud_rec_combined <- bind_rows(peek_1, peek_2, peek_3)

plot_1 <- fraud_rec_combined %>% ggplot(aes(x = is_fraud, fill = recipe)) +
  geom_bar() +
  facet_wrap(~ recipe, ncol = 1, scales = "free_y") +
  labs(title = "is_fraud counts after Down (recipe_2) and Up Sampling (recipe_3)",
       x = "is_fraud",
       y = "Count") +
  theme_bw()
plot_2 <- fraud_rec_combined %>% ggplot(aes(x = amount, fill = recipe)) +
  geom_histogram() +
  facet_wrap(~ recipe, ncol = 1, scales = "free") +
  labs(title = "Effect of transformations on amount values",
       x = "Transaction Amount",
       y = "Count") +
  theme_bw()


### 4. lr_spec, xgb_spec, fraud_wkfl_1, fraud_wkfl_2, fraud_wkfl_3, 
###    fraud_wkfl_4, fraud_wkfl_5, fraud_wkfl_6
lr_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

xgb_spec <- boost_tree() %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Logistic regression workflows
fraud_wkfl_1 <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(fraud_rec_1)

fraud_wkfl_2 <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(fraud_rec_2)

fraud_wkfl_3 <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(fraud_rec_3)

# XGBoost workflows
fraud_wkfl_4 <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(fraud_rec_1)

fraud_wkfl_5 <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(fraud_rec_2)

fraud_wkfl_6 <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(fraud_rec_3)


### 5. fraud_metrics_set, fraud_fit_1, fraud_fit_2, fraud_fit_3, fraud_fit_4, 
###    fraud_fit_5, fraud_fit_6
set.seed(42)

fraud_metric_set <- metric_set(recall, roc_auc, spec)

# 1. Logistic Regression: fraud_rec_1
fraud_fit_1 <- last_fit(
  fraud_wkfl_1,
  split = fraud_split,
  metrics = fraud_metric_set
)

# 2. Logistic Regression: fraud_rec_2
fraud_fit_2 <- last_fit(
  fraud_wkfl_2,
  split = fraud_split,
  metrics = fraud_metric_set
)

# 3. Logistic Regression: fraud_rec_3
fraud_fit_3 <- last_fit(
  fraud_wkfl_3,
  split = fraud_split,
  metrics = fraud_metric_set
)

# 4. XGBoost: fraud_rec_1
fraud_fit_4 <- last_fit(
  fraud_wkfl_4,
  split = fraud_split,
  metrics = fraud_metric_set
)

# 5. XGBoost: fraud_rec_2
fraud_fit_5 <- last_fit(
  fraud_wkfl_5,
  split = fraud_split,
  metrics = fraud_metric_set
)

# 6. XGBoost: fraud_rec_3
fraud_fit_6 <- last_fit(
  fraud_wkfl_6,
  split = fraud_split,
  metrics = fraud_metric_set
)


### 6. fraud_perf_summary
collect_metrics(fraud_fit_1)
collect_metrics(fraud_fit_2)
collect_metrics(fraud_fit_3)
collect_metrics(fraud_fit_4)
collect_metrics(fraud_fit_5)
collect_metrics(fraud_fit_6)

# Logistic regression models
fraud_metrics_1 <- collect_metrics(fraud_fit_1) %>%
  mutate(recipe = "fraud_rec_1", algorithm = "lr")

fraud_metrics_2 <- collect_metrics(fraud_fit_2) %>%
  mutate(recipe = "fraud_rec_2", algorithm = "lr")

fraud_metrics_3 <- collect_metrics(fraud_fit_3) %>%
  mutate(recipe = "fraud_rec_3", algorithm = "lr")

# XGBoost models
fraud_metrics_4 <- collect_metrics(fraud_fit_4) %>%
  mutate(recipe = "fraud_rec_1", algorithm = "xgb")

fraud_metrics_5 <- collect_metrics(fraud_fit_5) %>%
  mutate(recipe = "fraud_rec_2", algorithm = "xgb")

fraud_metrics_6 <- collect_metrics(fraud_fit_6) %>%
  mutate(recipe = "fraud_rec_3", algorithm = "xgb")

fraud_perf_summary <- bind_rows(
  fraud_metrics_1,
  fraud_metrics_2,
  fraud_metrics_3,
  fraud_metrics_4,
  fraud_metrics_5,
  fraud_metrics_6
) %>% arrange(.metric, desc(.estimate))


### 7. air, air_split, air_training, air_testing 
air_raw <- read_csv('https://www.dropbox.com/s/1sh4b85y52hrvwx/airlines.csv?dl=1')
set.seed(42)

air <- air_raw %>% mutate(satisfied = as.factor(satisfied))

air_split <- initial_split(air, strata = satisfied)
air_training <- training(air_split)
air_testing  <- testing(air_split)


### 8. air_rec_corr, air_rec_pca, peek_4, peek_5
air_rec_base <- recipe(satisfied ~ ., data = air_training) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors())

air_rec_corr <- air_rec_base %>%
  step_corr(all_numeric_predictors())

air_rec_pca <- air_rec_base %>%
  step_pca(all_numeric_predictors())

peek_4 <- air_rec_corr %>%
  prep() %>%
  bake(new_data = air_training) %>%
  mutate(recipe = "air_rec_corr")

peek_5 <- air_rec_pca %>%
  prep() %>%
  bake(new_data = air_training) %>%
  mutate(recipe = "air_rec_pca")


### 9. air_lr_corr_wkfl, air_lr_pca_wkfl, air_xgb_corr_wkfl, air_xgb_pca_wkfl
# Logistic Regression + Correlation Filter
air_lr_corr_wkfl <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(air_rec_corr)

# Logistic Regression + PCA
air_lr_pca_wkfl <- workflow() %>%
  add_model(lr_spec) %>%
  add_recipe(air_rec_pca)

# XGBoost + Correlation Filter
air_xgb_corr_wkfl <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(air_rec_corr)

# XGBoost + PCA
air_xgb_pca_wkfl <- workflow() %>%
  add_model(xgb_spec) %>%
  add_recipe(air_rec_pca)


### 10. air_folds, air_lr_corr_cv_fit, air_lr_pca_cv_fit, air_xgb_corr_cv_fit, air_xgb_pca_cv_fit
set.seed(42)
air_folds <- vfold_cv(air_training, v = 10, strata = satisfied)

# Log Reg + Corr
air_lr_corr_cv_fit <- fit_resamples(
  air_lr_corr_wkfl,
  resamples = air_folds,
  control = control_resamples(save_pred = TRUE)
)

# Log Reg + PCA
air_lr_pca_cv_fit <- fit_resamples(
  air_lr_pca_wkfl,
  resamples = air_folds,
  control = control_resamples(save_pred = TRUE)
)

# XGB  Corr
air_xgb_corr_cv_fit <- fit_resamples(
  air_xgb_corr_wkfl,
  resamples = air_folds,
  control = control_resamples(save_pred = TRUE)
)

# XGB + PCA
air_xgb_pca_cv_fit <- fit_resamples(
  air_xgb_pca_wkfl,
  resamples = air_folds,
  control = control_resamples(save_pred = TRUE)
)


### 11. air_perf_summary, air_final_fit, plot_3
# Collect detailed ROC AUC values from each workflow
lr_corr_metrics <- collect_metrics(air_lr_corr_cv_fit, summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  mutate(workflow = "air_lr_corr_wkfl")

lr_pca_metrics <- collect_metrics(air_lr_pca_cv_fit, summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  mutate(workflow = "air_lr_pca_wkfl")

xgb_corr_metrics <- collect_metrics(air_xgb_corr_cv_fit, summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  mutate(workflow = "air_xgb_corr_wkfl")

xgb_pca_metrics <- collect_metrics(air_xgb_pca_cv_fit, summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  mutate(workflow = "air_xgb_pca_wkfl")

air_perf_summary <- bind_rows(
  lr_corr_metrics,
  lr_pca_metrics,
  xgb_corr_metrics,
  xgb_pca_metrics
) %>%
  group_by(workflow) %>%
  summarise(
    minimum = min(.estimate),
    mean    = mean(.estimate),
    maximum = max(.estimate),
    sd      = sd(.estimate)
  )

air_final_fit <- last_fit(
  air_lr_corr_wkfl,
  split = air_split
)

# Extract predictions
air_final_predictions <- collect_predictions(air_final_fit)

# Create ROC curve plot
plot_3 <- air_final_predictions %>%
  roc_curve(truth = satisfied, .pred_0) %>%
  autoplot() +
  labs(
    title = "ROC Curve, Xgboost + Corr Preproccesor"
  )


###################################################################################
### 12. lc_*
lc <- read_csv('https://www.dropbox.com/s/yqjek9ve4z6lbw5/lc.csv?dl=1') %>% 
  mutate(loan_default = as.factor(loan_default))
set.seed(42)

# Split
lc_split <- initial_split(lc, prop = 0.75, strata = loan_default)
lc_training <- training(lc_split)
lc_testing  <- testing(lc_split)

# Recipe
lc_rec <- recipe(loan_default ~ ., data = lc_training) %>%
  step_dummy(all_nominal_predictors())

# Workflow using default XGBoost spec
xgb_spec_default <- boost_tree() %>%
  set_engine("xgboost") %>%
  set_mode("classification")

lc_xgb_wkfl <- workflow() %>%
  add_model(xgb_spec_default) %>%
  add_recipe(lc_rec)

# Cross Validate
set.seed(42)
lc_folds <- vfold_cv(lc_training, v = 10, strata = loan_default)

# Train Default Model via Resampling
set.seed(42)
lc_default_fit <- fit_resamples(
  lc_xgb_wkfl,
  resamples = lc_folds,
  control = control_resamples(save_pred = TRUE)
)

# Tunable XGBoost Spec (with selected tunable parameters)
xgb_spec_tuning <- boost_tree(
  learn_rate = tune(),
  loss_reduction = tune(),
  min_n = tune(),
  tree_depth = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# Tuning workflow
lc_xgb_tune_wkfl <- workflow() %>%
  add_model(xgb_spec_tuning) %>%
  add_recipe(lc_rec)

# Tuning Grid (10 random combinations)
set.seed(42)
lc_grid <- grid_random(
  parameters(xgb_spec_tuning),
  size = 10
)

# Enable parallel processing
library(doParallel)
registerDoParallel()

# Tune the model using grid search
set.seed(42)
tune_fit <- tune_grid(
  lc_xgb_tune_wkfl,
  resamples = lc_folds,
  grid = lc_grid,
  control = control_grid(save_pred = TRUE)
)

# Extract Best Parameters using roc_auc as the metric
best_parameters <- select_best(tune_fit, metric = "roc_auc")

# === PERFORMANCE SUMMARY SECTION ===
# Collect per-fold ROC AUC metrics from the default workflow
default_metrics <- collect_metrics(lc_default_fit, summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  mutate(workflow = "lc_xgb_wkfl")

# Collect per-fold ROC AUC metrics from the tuning results
tune_metrics <- collect_metrics(tune_fit, summarize = FALSE) %>%
  filter(.metric == "roc_auc") %>%
  mutate(workflow = "lc_xgb_tune_wkfl")

# Bind the metrics and summarize them (min, mean, max, sd) by workflow
lc_perf_summary <- bind_rows(
  default_metrics,
  tune_metrics
) %>%
  group_by(workflow) %>%
  summarise(
    minimum = min(.estimate),
    mean    = mean(.estimate),
    maximum = max(.estimate),
    sd      = sd(.estimate)
  )

# Print performance summary for inspection
print(lc_perf_summary)
# === END PERFORMANCE SUMMARY SECTION ===

# Finalize Workflow using the best parameters from tuning
final_lc_wkfl <- finalize_workflow(
  lc_xgb_tune_wkfl,
  best_parameters
)

# Fit Final Model on full training data and evaluate on the test set
lc_final_fit <- last_fit(
  final_lc_wkfl,
  split = lc_split
)



# Pre-Submission Checks --------------------------------------------------------------------------------

# The checks run by the command below will see whether you have named your objects 
# and columns exactly correct. Any issues it finds will be reported in the console. 
# If it see what it expects to see, you'll instead see a message that "All naming 
# tests passed."

source('submission_checks.R')