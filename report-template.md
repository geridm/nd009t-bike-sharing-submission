# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### GERALDIN DIAZ

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
When attempting to submit the initial predictions to Kaggle, I realized that the submission file required a specific format. The `datetime` column needed to be present, and the predictions for `count` had to be non-negative integer values, as bike demand cannot be fractional or negative.

The specific changes required were:
1.  Ensuring that the `datetime` column from the original test dataset was included in the submission file.
2.  Converting the `count` predictions to integer type (`.astype(int)`) and clipping them to ensure non-negative values (`.clip(0)`). This step is crucial because bike demand cannot logically be a decimal or negative number.

### What was the top ranked model that performed?
In the initial training run, utilizing `presets="best_quality"` without explicit feature engineering or hyperparameter tuning, the best-performing model identified by AutoGluon was `WeightedEnsemble_L2`. This model represents an ensemble of various base models (such as LightGBM, XGBoost, CatBoost, etc.) that AutoGluon automatically selects and combines to achieve optimal performance.

The Root Mean Squared Error (RMSE) score obtained on Kaggle for this initial model was **1.80094**.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Exploratory data analysis (EDA) of the initial dataset highlighted the critical importance of the `datetime` column. This column contained rich temporal information that, when decomposed, could effectively capture cyclic and seasonal patterns influencing bike demand.

Key findings from the EDA included:
* **Hourly patterns:** Bike demand significantly varies throughout the day, often showing distinct peaks during typical commuting hours.
* **Weekly patterns:** Demand differs between weekdays and weekends.
* **Monthly/Yearly patterns:** Seasonality (months, seasons) and overall yearly trends play a substantial role in bike rental variations.

Based on these insights, an `add_features` function was created to extract new, relevant features from the `datetime` column for both the training and test datasets:
* `year`: The year of the observation.
* `month`: The month of the year.
* `dayofweek`: The day of the week (0-6).
* `hour`: The hour of the day.
After these features were extracted, the original `datetime` column was dropped, as its information had been successfully disaggregated. Additionally, the `season` and `weather` columns were explicitly converted to a `category` data type. This conversion is vital for the model to correctly interpret these as distinct categories rather than continuous numerical values, ensuring better feature utilization.

### How much better did your model perform after adding additional features and why do you think that is?
Following the addition of features derived from `datetime` and the explicit conversion of `season` and `weather` to categorical types, the AutoGluon model (again utilizing `presets="best_quality"`) showed a dramatic improvement in performance.

The RMSE score on Kaggle for this model was **0.61383**.

This represents a substantial improvement of **more than 1.1 RMSE** compared to the initial model (from 1.80094 to 0.61383).

I believe this significant improvement is primarily due to the new features (`year`, `month`, `dayofweek`, `hour`) explicitly capturing the **seasonality and cyclical patterns** inherently present in bike demand data. AutoGluon was able to directly leverage this disaggregated information, allowing it to better identify and model demand variations at different times of the day, week, month, and year. Without these features, the model would either have to infer these patterns indirectly or would fail to capture them as effectively. The conversion of `season` and `weather` to categorical types further aided the model's interpretation, preventing it from assuming linear relationships for these distinct categories.

## Hyper parameter tuning
### How much better did your model perform after trying different hyper parameters?
After applying hyperparameter optimization (HPO) to the model with additional features, a further improvement in performance was observed.

The RMSE score on Kaggle for this HPO-tuned model was **0.54847**.

This constitutes an improvement of approximately **0.065 RMSE** compared to the previous model (from 0.61383 to 0.54847). While this improvement is smaller than the impact of feature engineering, it is still a valuable optimization that pushed the model's performance into the "Exceeds Expectations" range.

### If you were given more time with this dataset, where do you think you would spend more time?
If granted more time with this dataset, I would focus on the following areas to further enhance model performance:

1.  **More Advanced Feature Engineering:** I would explore creating additional, more sophisticated features, such as:
    * **Interaction variables:** For example, interactions between `hour` and `dayofweek` (e.g., demand at 8 AM on a Monday vs. a Sunday).
    * **Specific holidays/events:** Researching and incorporating specific national holidays or major city events that might not be fully captured by the generic `holiday` column.
    * **Lag features:** If the dataset allowed, integrating bike demand from previous hours or days (though this dataset doesn't directly provide such time-series granularity).
    * **External data integration:** Considering external data like unusual weather events, public transport strikes, or large-scale sporting/cultural events in the city.

2.  **Deeper and Targeted Hyperparameter Optimization:**
    * While AutoGluon performs an effective random search, with more time, I could:
        * **Increase `num_trials`:** Allow more trials for AutoGluon to explore a larger hyperparameter space.
        * **Define custom search spaces:** If a specific base model (e.g., LightGBM) is consistently performing well, I could configure a more targeted hyperparameter search space specifically for that model, rather than relying solely on AutoGluon's general search.
        * **Experiment with different `searcher` and `scheduler` algorithms:** Explore more advanced search algorithms (e.g., Bayesian Optimization) if computational resources permit.

3.  **Error Analysis and Individual Model Examination:**
    * I would thoroughly analyze the mispredictions of my best model to understand when and why it fails. This could reveal the need for specific features for certain scenarios (e.g., extreme weather conditions not adequately represented).
    * I would examine the performance of individual models within AutoGluon's ensemble (`predictor.leaderboard()`) to understand which ones contribute most and if any could be fine-tuned to outperform the ensemble on specific data subsets.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.

|model|hpo1|hpo2|hpo3|score|
|---|---|---|---|---|
|initial|- |- |- |1.80094|
|add_features|- |- |- |0.61383|
|hpo|num_trials=10|scheduler=local|searcher=random|0.54847|

*Note on hpo1, hpo2, hpo3*: For the "initial" and "add_features" models, specific hyperparameters were not explicitly modified by the user; AutoGluon utilized its default values and internal optimization strategies (`presets="best_quality"`). For the "hpo" model, AutoGluon's hyperparameter optimization process was activated, with the specified number of trials, scheduler, and search strategy used for automatic hyperparameter exploration.

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_train_score.png](model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](model_test_score.png)

## Summary
This project demonstrated a typical machine learning workflow, from data loading to model optimization. The results obtained highlight the critical importance of **feature engineering** in model performance. Creating time-based features (hour, day of week, month, year) drastically improved the initial model by allowing it to capture the inherent seasonal and cyclical patterns in the bike demand data.

Subsequently, **hyperparameter optimization (HPO)** using AutoGluon's automated capabilities provided further, albeit more modest, improvement, pushing the model's performance to an outstanding level. The ease with which AutoGluon handles model ensemble and HPO makes it a powerful tool for rapidly achieving high-performing models.

Overall, the strategy of sound exploratory data analysis followed by intelligent feature engineering and then incremental model optimization (like HPO) proved highly effective for this dataset, achieving a final RMSE of 0.54847 on Kaggle.
