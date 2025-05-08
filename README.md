[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/DIu1ZM9Q)
# Modeling Pipeline Case Assignment

Well, my friends. After weeks of anticipation, we've arrived at our first actual model-building assignment. Exciting! This first assignment will have you build two models with two different datasets using the powerful modeling workflow tools from the `tidymodels` packages.

> [!TIP]
> A quick note about setting seeds. There are several places where it will help the grader if you use the same seed that the grader uses. To avoid any confusion or uncertainty, I have included each `set.seed()` command where it's required, along with comments to pinpoint the process that each seed is intended to standardize. You'll also notice that I'm using the exact same seed (42, of course), each time it's set. So just take care not to add any _extra_ random processes, and everything should line up nicely.

---

### 1. wine, wine_split, wine_training, wine_testing

The first model you'll be building will use the wine dataset from our last case assignment. You'll be predicting `quality`. Start by removing the `id` column from `wine_raw`, saving the result to `wine`. Then use that `wine` tibble to create a `wine_split` object, followed by the corresponding `wine_training` and `wine_testing` tibbles. (Just make sure your code for this section comes _after_ the `set.seed(42)` I provided.)

> [!IMPORTANT] 
> After setting up your model pipeline, you should now have `wine`, `wine_split`, `wine_training`, and `wine_testing` objects in your environment.

---

### 2. wine_lm, wine_lm_fit

Next, create a model specification for a basic linear regression model using the standard "lm" engine. Save the specification as `wine_lm`, then fit the training data with that model, saving the result as `wine_lm_fit`.

> [!IMPORTANT] 
> After accomplishing the above, you should now have `wine_lm` and `wine_lm_fit` objects in your environment.

---

### 3. wine_predictions, wine_results, wine_performance, plot_1

Now generate predictions for the testing data (saved to `wine_predictions`), and combine those predictions with the `wine_testing` test data (saved to `wine_results`). Use the standard functions from the `tidymodels` package to calculate the RMSE and the R-squared model performance metrics. Ensure that both of these metrics are stored together in a single, 2x3 tibble called `wine_performance`. Then use the data from `wine_results` to generate a plot (saved to `plot_1`) that looks similar the plot below.

> [!TIP]
> I've used an `alpha=0.5` for the points on the plot, and applied the `theme_bw()` theme. And make sure to double-check your text in each of the following labels: title, x-axis, y-axis, legend.

<img src="plots/plot_1.png"  width="80%">

> [!IMPORTANT] 
> After accomplishing the above, you should now have `wine_predictions`, `wine_results`, `wine_performance`, and `plot_1` objects in your environment.

---

### 4. apple, apple_split, apple_training, apple_testing

On to the next dataset! This one is a classification dataset in which you'll be training a model to predict an apple's quality ("good" or "bad") based on various characteristics. After reading in the `apple_raw` data, remove the `a_id` column and ensure that the `quality` outcome column is a factor so that it's ready to be used in a classification model. Save the result to `apple`. Then use that `apple` tibble to create a `apple_split` object, followed by the corresponding `apple_training` and `apple_testing` tibbles. (Again, make sure your code for this section comes _after_ the `set.seed(42)` I provided.)

> [!IMPORTANT] 
> After setting up your model pipeline, you should now have `apple`, `apple_split`, `apple_training`, and `apple_testing` objects in your environment.

---

### 5. apple_spec_rpart, apple_spec_c5, apple_fit_rpart, apple_fit_c5

Next, create **two** model specifications for decision tree models, one using the "rpart" engine and the other using the "C5.0" engine. (Naturally, we _must_ use decision _tree_ algorithms if we're modeling with data about _apples_.) Save the specifications as `apple_spec_rpart` and `apple_spec_c5`, respectively, then fit the training data with each model specification, saving the results as `apple_fit_rpart` and `apple_fit_c5`, respectively.

> [!IMPORTANT] 
> After accomplishing the above, you should now have `apple_spec_rpart`, `apple_spec_c5`, `apple_fit_rpart`, and `apple_fit_c5` objects in your environment.

---

### 6. apple_preds_class_rpart, apple_preds_class_c5, apple_preds_prob_rpart, apple_preds_prob_c5, apple_results

Next, for both of the models you just fit, you'll need to generate _two sets_ of predictions for the `apple_testing` data. (So that's a total of four `predict()` function calls). One pair of predictions should contain the standard class predictions ("good" or "bad") generated by the trained model; these predictions should be saved as `apple_preds_class_rpart` and `apple_preds_class_c5`. The second pair of predictions should each contain two probability columns that refer to the model's predicted probabilities that a given apple with its characteristics belongs to either the "good" class or the "bad" class; these predictions should be saved as `apple_preds_prob_rpart` and `apple_preds_prob_c5`. 

After generating all 4 of these tibbles, you'll need to combine all of the predictions data with the `apple_testing` data so that we can do some performance comparisons. To do this, I'd recommend selecting just the `quality` column from `apple_testing`, and then doing a series of `bind_cols()` operations with the 4 `apple_preds_*` tibbles, ensuring that the columns are renamed in a way that identifies the model specification that produced them. (See the expected naming conventions in the sample tibble below.) The resulting combined tibble should be saved to `apple_results`. 

> [!TIP]
> Because the previous paragraph might feel a bit convoluted, I'm going to provide you a preview of the ending tibble that you should be aiming for. Note that the actual values of the tibble might differ from what you see below, but this will at least give you an idea of what columns you're aiming for, as well as the number of rows you should see:
>
> ```
> # A tibble: 1,000 × 7
>   quality .pred_class_rpart .pred_class_c5 .pred_bad_rpart .pred_good_rpart .pred_bad_c5 .pred_good_c5
>   <fct>   <fct>             <fct>                    <dbl>            <dbl>        <dbl>         <dbl>
> 1 good    good              bad                      0.241           0.759        0.969         0.0313
> 2 bad     bad               bad                      0.949           0.0515       0.986         0.0138
> 3 good    good              good                     0.241           0.759        0.131         0.869 
> ```

> [!IMPORTANT] 
> After accomplishing the above, you should now have `apple_preds_class_rpart`, `apple_preds_class_c5`, `apple_preds_prob_rpart`, `apple_preds_prob_c5`, and `apple_results` objects in your environment.

---

### 7. perf_default_threshold_rpart, perf_default_threshold_c5

Okay. We're now ready to compare the performance of our two models. Let's start by using the `conf_mat()` function in conjunction with `summary()` to generate the standard set of classification performance metrics using the two class prediction columns that are included in your `apple_results` tibble. (Do this using two completely separate `conf_mat() %>% summary()` commands, one for the `rpart` model and the other for the `C5.0` model.) Store the resulting tibbles as `perf_default_threshold_rpart` and `perf_default_threshold_c5`, respectively.

> [!TIP]
> The "class predictions" I'm asking you to evaluate here are stored in the `.pred_class_rpart` and `.pred_class_c5` columns.

> [!IMPORTANT] 
> After accomplishing the above, you should now have `perf_default_threshold_rpart` and `perf_default_threshold_c5` objects in your environment.

---

### 8. perf_manual_threshold_rpart, perf_manual_threshold_c5

Note that the class predictions you used to generate the results in the prior step came from using the `predict()` function to generate the `.pred_class` columns from each model. When you produce class predictions this way, the value of `.pred_class` is assigned based on a _default_ probability threshold value of 0.5, meaning that when a given apple's probability of belonging to either the "good" class or the "bad" class (stored in `.pred_good` and `.pred_bad` columns, respectively) exceeds 0.5, it is assigned to that class. (Note also that the two probability columns for a given model are _complementary probabilities_ that sum to 1 for a two-class classification model like this, so there is only ever one of the two probabilities greater than 0.5.)

To help us understand what's really going on with these class predictions, consider the following hypothetical scenario. Assume that you're building this model for an apple supplier with the goal to identify "bad" apples that need to be manually inspected by the quality assurance team. Assume further that the company's credibility depends on ensuring that very few "bad" apples get shipped to their grocery store customers. Because of this, the company is much more interested in "catching" any apples that fall into that "bad" quality category, even if it means that their quality assurance team has to manually inspect extra apples that end up being fine. 

Given this scenario, the company's Chief Data Scientist has established a guideline that the model we're building needs to prioritize _recall_ rate without sacrificing too much in terms of the model's _precision_. (It might be helpful for you to return to [this slide](https://www.dropbox.com/scl/fi/fu8hrrn42swin6a31wd4d/confusion-matrix-performance.pdf?rlkey=7igh1hjfgbkpcrs1wlgiwviaf&dl=0) from our lecture on classification model evaluation to review what we're talking about here.) Anyway, the chief data scientist is looking for a model that meets the following performance criteria in identifying "bad" apples:
- recall rate > 90%
- precision > 60%

Let's first have a look at the performance metrics we generated using the default threshold of 0.5 (which can be found in the `perf_default_threshold_rpart` and `perf_default_threshold_c5` tibbles from the previous step). If you glance through the various measures, you'll see that the two engines ("rpart" and "C5.0") are fairly similar in their overall performance, with the "C5.0" model probably _slightly_ outperforming the other on most metrics. Note also that, while each model's default threshold classification produced a precision rate that easily exceeds the desired 60%, the recall rates for either model aren't where the chief data scientist wants it to be. (Nothing to save here, but examine the contents of the two default threshold metrics tibbles.)

Your task is to see if you can find a better probability threshold for each of the two models you've trained that produces classification performance that meets the criteria required by the chief data scientist. To accomplish this, you'll need to create your own classification prediction using a probability threshold _other than_ the default of 0.5. This is as simple as adding a new (factor) column to `apple_results`, using `if_else()` to set the value to either "bad" or "good" based on the corresponding `.pred_bad_*` column's value. The tibble can then be passed to `conf_mat()` and then to `summary()` like you did before (though you'll use your newly created column as the `estimate` parameter in `conf_mat()`).

You'll do this in two completely separate operations, one for each trained model, manually searching for the best threshold value to use that achieves the desired balance between precision and recall.

> [!TIP]
> Quick aside here: One of the two models will be significantly easier to optimize in the way I'm describing above. In fact, you may not actually be able to identify a threshold for that model that meets the chief data scientist's criteria (potentially depending on some randomness in the sampling across different laptop computers, etc.). If this is the case, just use the manual threshold that gets the recall and precision values the _closest_ to the desired rates. 

When you've found what you feel is the best threshold for each model, you'll then save the resulting metrics tibble, naming the two results tibbles as `perf_manual_threshold_rpart` and `perf_manual_threshold_c5`, respectively. (Note that I'm not asking you to save the thresholds nor the manually calculated class predictions that you used on your way toward the performance metric tibbles.)

> [!IMPORTANT] 
> After accomplishing the above, you should now have `perf_manual_threshold_rpart` and `perf_manual_threshold_c5` objects in your environment.

---

### 9. plot_2

The last thing we're going to do before finalizing the model is build a chart that will help us visualize the threshold selection exercise we just finished. This should hopefully drive a few points home and also provide an explanation as to why one of the engines was harder to work with in the previous exercise. 

Using the `apple_results` data, your goal is to compare the distributions of the prediction probabilities we were using in the last step (namely, `.pred_bad_rpart` and `.pred_bad_c5`), separated by the two different decision tree engines that produced the probabilities and the "actual" quality label. This will involve some intentional selecting, pivoting, mapping, and labeling, and you can use the example plot below to see what you're aiming for. (And note that there is no need to save out the prepared tibble prior to generating the plot.)

Once you have the plot looking right, take a moment to think through the following questions:
- Why is it informative to compare the prediction probabilities like this (i.e., splitting the probabilities apart by their corresponding "true" quality label)? What does this reveal about the two engines' performance, despite most of the performance metrics from the default threshold classifications being pretty similar?
- Which of the two engines would you be more confident in recommending? Why?
- What does the plot reveal about why one engine gave you more trouble in identifying a threshold that met the chief data scientist's criteria?
- How _cool_ do you feel right now, knowing that you just emerged victorious from your first battle with the inner workings of a classification algorithm?

<img src="plots/plot_2.png"  width="80%">

> [!IMPORTANT] 
> Make sure that you save the plot object as `plot_2` in your environment.

---

### 10. apple_finalized, apple_test_predictions, apple_test_metrics

Having thoroughly explored the two models, the last thing you'll do is select the one that is most promising in terms of performance and in meeting the criteria established by the chief data scientist. 

> [!TIP]
> Despite the work we just did in problems 8 and 9 to manually explore classification thresholds, we are actually going to just use the default thresholds in the model finalization steps (meaning that you can use the standard `tidymodels` functions referenced in the next paragraph like we've been using them all along). Sorry if that's disappointing...I just assume that most of you are ready to be done with this assignment. :)

Using the model specification for the most promising model, you'll use `last_fit()` to wrap up the model training effort, saving the result as `apple_finalized`. Then you can use the `collect_predictions()` and `collect_metrics()` functions to extract the (default) test set predictions and testing data performance metrics from the finalized model object, saving the extracted data to `apple_test_predictions` and `apple_test_metrics`, respectively.

> [!IMPORTANT] 
> After accomplishing the above, you should now have `apple_finalized`, `apple_test_predictions`, and `apple_test_metrics` objects in your environment.

---

# Final Cleanup and Submission

Please do the following to make sure your code runs without errors and passes the submission checks:

1. Run the "Pre-Submission Checks" section to check whether you saved (and spelled) all of the expected variables and columns along the way. Take care of any issues it uncovers.
2. Restart your R session (Session >> Restart R).
3. Run your entire script from the beginning, watching for any errors along the way. Easiest way to do this is to "Select All" and then hit "Run". If you have any errors (including in the Pre-Submission Checks section), you'll want to fix them before submitting.