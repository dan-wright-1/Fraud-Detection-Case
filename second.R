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