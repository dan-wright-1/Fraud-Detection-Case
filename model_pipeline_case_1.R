library(tidyverse)
library(tidymodels)
# test


# Set your own script to work in the right local directory:
setwd('C:\\Users\\dwrig\\Documents\\GitHub\\is-555-09-modeling-pipeline-1-dan-wright-1')


### 1. wine, wine_split, wine_training, wine_testing
wine_raw <- read_csv('https://www.dropbox.com/scl/fi/gkybj48qxyjedenjook54/wine_clean.csv?rlkey=ye05omtdtrvfj5cggwg1ql2tu&dl=1') 
set.seed(42)

wine <- select(wine_raw, -id)

# Create the split object (default 75% training, 25% testing)
wine_split <- initial_split(wine)

# Extract training and testing sets
wine_training <- training(wine_split)
wine_testing  <- testing(wine_split)


### 2. wine_lm, wine_lm_fit
# Define a linear regression model specification using the standard lm engine
wine_lm <- linear_reg() %>% 
  set_engine("lm")

# Fit the model to the training data (using all predictors to predict quality)
wine_lm_fit <- wine_lm %>% 
  fit(quality ~ ., data = wine_training)


### 3. wine_predictions, wine_results, wine_performance, plot_1

# 1. Generate predictions for the testing data
wine_predictions <- predict(wine_lm_fit, new_data = wine_testing)

# 2. Combine predictions with the wine_testing data
wine_results <- bind_cols(wine_testing, wine_predictions)

# 3. Calculate RMSE and R-squared metrics; filter to create a 2x3 tibble
wine_performance <- wine_results %>% 
  metrics(truth = quality, estimate = .pred) %>% 
  filter(.metric %in% c("rmse", "rsq"))

# 4. Create the plot with alpha = 0.5 for points and using theme_bw().
#    Assuming there is a column named 'type' in wine_results indicating wine type.
plot_1 <- ggplot(wine_results, aes(x = quality, y = .pred, fill = type)) +
  geom_point(alpha = 0.5, shape = 21) +
  facet_wrap(~ type) +
  labs(
    title = "Predicted vs. Actual Wine Quality, Linear Regression",
    x = "Actual Quality",
    y = "Predicted Quality"
  ) +
  theme_bw()


### 4. apple, apple_split, apple_training, apple_testing
apple_raw <- read_csv('https://www.dropbox.com/scl/fi/ua0bkccivculaiz7gelmb/apples.csv?rlkey=tak0jhd2ddi0t1tl8bt45947y&dl=1')
set.seed(42)

# Remove the a_id column and convert quality to a factor
apple <- apple_raw %>% 
  select(-a_id) %>% 
  mutate(quality = as.factor(quality))

# Split the data (default: 75% training, 25% testing)
apple_split <- initial_split(apple)
apple_training <- training(apple_split)
apple_testing <- testing(apple_split)


### 5. apple_spec_rpart, apple_spec_c5, apple_fit_rpart, apple_fit_c5

# Decision tree model specification using the "rpart" engine
apple_spec_rpart <- decision_tree() %>% 
  set_mode("classification") %>% 
  set_engine("rpart")

# Fit the rpart model to the training data
apple_fit_rpart <- apple_spec_rpart %>% 
  fit(quality ~ ., data = apple_training)

# Decision tree model specification using the "C5.0" engine
apple_spec_c5 <- decision_tree() %>% 
  set_mode("classification") %>% 
  set_engine("C5.0")

# Fit the C5.0 model to the training data
apple_fit_c5 <- apple_spec_c5 %>% 
  fit(quality ~ ., data = apple_training)


### 6. apple_preds_class_rpart, apple_preds_class_c5, apple_preds_prob_rpart, 
###    apple_preds_prob_c5, apple_results

# 1. Generate class predictions (default output)
apple_preds_class_rpart <- predict(apple_fit_rpart, new_data = apple_testing) %>%
  mutate(.pred_class_rpart = .pred_class) # %>%
  # select(-.pred_class)

apple_preds_class_c5 <- predict(apple_fit_c5, new_data = apple_testing) %>%
  mutate(.pred_class_c5 = .pred_class) # %>%
  # select(-.pred_class)

# 2. Generate probability predictions (with type = "prob")

# For the rpart model:
apple_preds_prob_rpart <- predict(apple_fit_rpart, new_data = apple_testing, type = "prob") %>%
  mutate(.pred_bad_rpart = .pred_bad,
         .pred_good_rpart = .pred_good) # %>%
  # select(-.pred_bad, -.pred_good)

# For the C5.0 model:
apple_preds_prob_c5 <- predict(apple_fit_c5, new_data = apple_testing, type = "prob") %>%
  mutate(.pred_bad_c5 = .pred_bad,
         .pred_good_c5 = .pred_good) # %>%
  # select(-.pred_bad, -.pred_good)



# 3. Combine the quality column from apple_testing with all prediction tibbles
apple_results <- apple_testing %>%
  select(quality) %>%
  bind_cols(apple_preds_class_rpart, apple_preds_class_c5,
            apple_preds_prob_rpart, apple_preds_prob_c5)

# Preview the combined tibble
print(apple_results)


### 7. perf_default_threshold_rpart, perf_default_threshold_c5
# For the rpart model
perf_default_threshold_rpart <- conf_mat(apple_results,
                                         truth = quality,
                                         estimate = .pred_class_rpart) %>%
  summary()

# For the C5.0 model
perf_default_threshold_c5 <- conf_mat(apple_results,
                                      truth = quality,
                                      estimate = .pred_class_c5) %>%
  summary()


### 8. perf_manual_threshold_rpart, perf_manual_threshold_c5

# Create a new class prediction for the rpart model based on a manual threshold
apple_results <- apple_results %>%
  mutate(manual_class_rpart = factor(if_else(.pred_bad_rpart > 0.17, "bad", "good"),
                                     levels = levels(quality)))


# Compute the confusion matrix and summary metrics for the rpart model with the manual threshold
perf_manual_threshold_rpart <- conf_mat(apple_results,
                                        truth = quality,
                                        estimate = manual_class_rpart) %>%
  summary()

# Create a new class prediction for the C5.0 model based on a manual threshold
apple_results <- apple_results %>%
  mutate(manual_class_c5 = factor(if_else(.pred_bad_c5 > 0.12, "bad", "good"),
                                  levels = levels(quality)))


# Compute the confusion matrix and summary metrics for the C5.0 model with the manual threshold
perf_manual_threshold_c5 <- conf_mat(apple_results,
                                     truth = quality,
                                     estimate = manual_class_c5) %>%
  summary()


### 9. plot_2
plot_2 <- apple_results %>%
  select(quality, .pred_bad_rpart, .pred_bad_c5) %>%
  pivot_longer(cols = c(.pred_bad_rpart, .pred_bad_c5),
               names_to = "engine",
               values_to = "pred_bad_prob") %>%
  mutate(engine = case_when(
    engine == ".pred_bad_rpart" ~ "rpart",
    engine == ".pred_bad_c5" ~ "c5"
  )) %>%
  ggplot(aes(x = pred_bad_prob, fill = quality)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~ engine, ncol = 1) +
  labs(title = "'Bad' Probability Distributions by Actual Quality and Algorithm Engine",
       x = "Predicted Probability of 'Bad' Quality",
       y = "Density",
       fill = "Actual Quality") +
  theme_bw()

### 10. apple_finalized, apple_test_predictions, apple_test_metrics
# Finalize the model using the C5.0 specification and the pre-created apple_split object
apple_finalized <- apple_spec_c5 %>%
  last_fit(quality ~ ., split = apple_split)

# Extract test set predictions from the finalized model
apple_test_predictions <- collect_predictions(apple_finalized)

# Extract performance metrics on the test set from the finalized model
apple_test_metrics <- collect_metrics(apple_finalized)


# Pre-Submission Checks --------------------------------------------------------------------------------

# The checks run by the command below will see whether you have named your objects 
# and columns exactly correct. Any issues it finds will be reported in the console. 
# If it see what it expects to see, you'll instead see a message that "All naming 
# tests passed."

source('submission_checks.R')
  