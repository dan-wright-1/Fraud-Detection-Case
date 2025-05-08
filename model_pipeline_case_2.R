library(tidyverse)
library(tidymodels)
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
