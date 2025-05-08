# Modeling Pipeline Case - Fraud Detection

## Introduction

Welcome to my fraud detection project! In my intro data science course, we used R to clean data and build a machine learning model to predict a response variable. This particular model leverages historical transaction data to identify patterns and anomalies indicative of fraudulent activities.

## Problem Statement

Fraudulent transactions result in significant financial losses and pose a reputational risk to organizations. In this project, I aimed to build a model that can accurately classify transactions as legitimate or fraudulent, thus mitigating potential losses and enhancing fraud prevention strategies.

## Data Overview

* **Dataset:** Historical transaction data with labeled instances of fraudulent and legitimate transactions. The dataset is highly unbalanced, as per usual with transaction data. The positive class (fraud) accounts for 0.172% of all transactions. See the dataset on Kaggle here: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
* **Features:** Transaction amount, timestamp, and 28 other PCA-transformed features (v1-v28) of unknown column names due to privacy restrictions.
* **Label:** Fraud represented by 1 and normal transaction represented by 0. 
* **Size:** 284,807 transactions with 492 labeled as fraudulent.

## Methodology

1. **Data Preprocessing:**

   * Data Cleaning: The data was clean upon download from Kaggle.
   * Normalization & Downsampling: Various preprocessing techniques and model architectures are used to compare performance.
   * Feature Engineering: All available features were used in this model because this was the first iteration. Further investigation may surface features that worsen model performance, which would be candidates for removal. 
   * Data Splitting: 70% training, 30% testing.
  
   * Three different preprocessing recipes were used:
     * Recipe 1: Baseline with no transformations.
     * Recipe 2: Yeo-Johnson transformation, normalization, and downsampling.
     * Recipe 3: Range scaling, Box-Cox transformation, normalization, and upsampling.

2. **Model Selection:**

   * Logistic Regression and XGBoost models were employed using each preprocessing recipe (a total of 6 workflows).

3. **Model Evaluation:**

   * Accuracy, Precision, Recall, F1 Score, AUC-ROC.

## Results and Findings

* The XGBoost model achieved an accuracy of **97%**[verify number], a recall of **92%**[verify number], and an AUC-ROC of **0.98**[verify number].
* Summary here like "High precision and recall indicate the model's effectiveness in minimizing false positives and false negatives."
* Summary here like "The model demonstrated robust performance on unseen data, validating its generalizability."

## Recommendations

* Integrate the model with real-time monitoring systems to detect fraud as transactions occur.
* Implement periodic retraining to maintain model accuracy with evolving fraud patterns.
* Develop a dashboard to visualize key fraud metrics and model performance.

## Next Steps

* Monitor model performance post-deployment.
* Incorporate feedback from fraud analysts to refine model features.
* Explore advanced algorithms such as neural networks for potential performance gains.

## Key Takeaways
* Downsampling and upsampling are two strategies for dealing with extremely imbalanced data.
* Using multiple preprocessing recipes with multiple algorithms allows data scientists to experiment and compare model performance.
* Determining which metric to optimize for is context-dependent.

---

# Modeling Pipeline

Here's how the sausage was made! 🌭

### 1. Training/Testing Split

Converted the `is_fraud` column to a factor, saving the result to `fraud`. Then used that `fraud` tibble to create a `fraud_split` object, followed by the corresponding `fraud_training` and `fraud_testing` tibbles. The function defaults to a standard 70/30 split.

### 2. Recipe Creation

Created three recipes to compare different approaches to handling imbalanced data. The first recipe (`fraud_rec_1`) was a basic recipe with no recipe steps added. The second recipe (`fraud_rec_2`) addressed the right-skewed shape of the `amount` column using a YeoJohnson transformation, followed by scaling using a z-score, and applied the `step_downsample()` step with its default parameters. The third recipe (`fraud_rec_3`) addressed the right-skewed shape of the `amount` column using a BoxCox transformation, followed by scaling using a z-score, and applied the `step_upsample()` step with its default parameters. A `step_mutate()` or `step_range()` step was added to ensure the BoxCox transformation could handle the zero values in the `amount` column.

### 3. Data Transformation and Visualization

Applied the recipes to the training data using the `prep()` and `bake()` functions, saving the results to `peek_1`, `peek_2`, and `peek_3`. A `recipe` column was added to each tibble to indicate the source recipe (`fraud_rec_1`, `fraud_rec_2`, or `fraud_rec_3`). Created two visualizations to assess the impact of the transformations. The first plot (`plot_1`) illustrates the differences in class distributions due to downsampling and upsampling. The second plot (`plot_2`) demonstrates the effects of YeoJohnson and BoxCox transformations on the right-skewed `amount` column.

### 4. Model Specification and Workflow Creation

Created two model specifications: `lr_spec` for logistic regression using the `glm` engine and `xgb_spec` for a boosted tree model using the `xgboost` engine. Constructed six workflows by combining the recipes and model specifications as follows:

* `fraud_wkfl_1`: `fraud_rec_1` and `lr_spec`
* `fraud_wkfl_2`: `fraud_rec_2` and `lr_spec`
* `fraud_wkfl_3`: `fraud_rec_3` and `lr_spec`
* `fraud_wkfl_4`: `fraud_rec_1` and `xgb_spec`
* `fraud_wkfl_5`: `fraud_rec_2` and `xgb_spec`
* `fraud_wkfl_6`: `fraud_rec_3` and `xgb_spec`

### 5. Metric Set Creation and Model Training

Created a custom metric set named `fraud_metric_set` using `metric_set()` to evaluate model performance using `recall`, `roc_auc`, and `specificity`. Applied the `last_fit()` function to each of the six workflows, providing the `fraud_metric_set` for evaluation. Saved the resulting model fit objects as `fraud_fit_1` through `fraud_fit_6`.

### 6. Model Performance Summary

Compiled model performance metrics using `collect_metrics()` and created a consolidated summary tibble named `fraud_perf_summary`. Added `recipe` and `algorithm` columns to indicate the source recipe and algorithm used for each model fit object. Arranged the tibble by metric (ascending) and estimate (descending) to facilitate comparison of model performance across recipes and algorithms.

---
