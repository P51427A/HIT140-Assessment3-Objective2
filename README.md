# HIT140 Assessment 3 – Objective 2

Darwin Group 8

- Pranjal Awasthi (S405821)
- Milan Bhattarai (S403198)
- Nischal Malla (S401733)
- Rahul Neupane (S404801)

## Project overview

Objective 2 contains two regression tasks using FIFA World Cup 2026 data and pre-match explanatory variables.

- **Task 2.1:** predict **absolute goal difference** using 104 match-level observations.
- **Task 2.2:** predict **goals scored by a team in a specific match** using 208 team-match observations.

All final explanatory variables are based on information available before kick-off. The final Task 2.2 workflow uses grouped validation by `match_id` so both team rows from the same match remain in the same split.

---

## Final Task 2.1 files

### Final dataset
`task2_1_final_abs_dataset.csv`

### Final validation
`31_task2_1_final_validation.py`

### Competing regression algorithms
`29_task2_1_algorithm_comparison.py`

### Random Forest tuning experiment
`30_task2_1_random_forest_tuning.py`

### Main outputs
- `task2_1_final_validation_results.csv`
- `task2_1_final_cv_folds.csv`
- `task2_1_holdout_predictions.csv`
- `task2_1_algorithm_comparison.csv`
- `task2_1_random_forest_feature_importance.csv`
- `task2_1_random_forest_tuning_results.csv`
- `task2_1_random_forest_final_summary.csv`

---

## Final Task 2.2 files

### Final dataset
`task2_2_final_model_dataset.csv`

### Final validation
`33_task2_2_final_validation.py`

### Competing regression algorithms
`32_task2_2_algorithm_comparison.py`

### Main outputs
- `task2_2_final_validation_results.csv`
- `task2_2_final_cv_folds.csv`
- `task2_2_final_coefficients.csv`
- `task2_2_final_diagnostics.csv`
- `task2_2_holdout_predictions.csv`
- `task2_2_algorithm_comparison.csv`
- `task2_2_algorithm_cv_folds.csv`
- `task2_2_feature_audit.csv`

---

## Final figures

### Main report figures
`34_create_final_report_figures.py`

Produces the main MLR figures, including:
- Task 2.1 actual vs predicted
- Task 2.1 standardised coefficients
- Task 2.2 actual vs predicted
- Task 2.2 standardised coefficients

### MLR diagnostic figures
`35_create_mlr_diagnostic_figures.py`

Produces:
- Task 2.1 residuals vs fitted
- Task 2.1 Q-Q plot
- Task 2.2 residuals vs fitted
- Task 2.2 Q-Q plot

### Hold-out prediction exports
- `36_export_task2_1_holdout_predictions.py`
- `37_export_task2_2_holdout_predictions.py`

### Combined algorithm-comparison figure
`38_create_combined_algorithm_comparison.py`

Produces:
`figure_3_combined_algorithm_comparison.png`

### Task 2.2 standalone algorithm-comparison figure
`39_create_task2_2_algorithm_comparison_png.py`

Produces:
`figure_task2_2_algorithm_comparison.png`

---

## Final modelling summary

### Task 2.1
The final MLR model had weak predictive performance for absolute goal difference.

Key final results:
- Hold-out RMSE: **1.547**
- Hold-out R²: **0.008**
- 5-fold CV RMSE: **1.368**
- 5-fold CV R²: **-0.063**

Random Forest was the strongest competing regression model under cross-validation:
- CV RMSE: **1.291**
- CV R²: **0.069**

### Task 2.2
The final MLR model performed better for predicting team goals scored.

Key final results:
- Grouped hold-out RMSE: **1.289**
- Grouped hold-out R²: **0.233**
- 5-fold grouped CV RMSE: **1.224**
- 5-fold grouped CV R²: **0.198**

MLR performed better than the competing regression algorithms under grouped cross-validation.

---

## Data integrity notes

- Task 2.1 final dataset: **104 rows**
- Task 2.2 final dataset: **208 rows = 104 matches × 2 teams**
- Exactly **8 explanatory variables** are used in each task.
- Only **3 predictor concepts** are shared across the two tasks.
- Final modelling datasets contain **no missing values**.
- Task 2.2 match dates are parsed explicitly as `YYYY-MM-DD`.
- Task 2.2 recent-history features use only information available before the target fixture.
- Task 2.2 uses grouped splitting by `match_id` to prevent paired-match leakage.

---

## Suggested final run order

```bash
python 29_task2_1_algorithm_comparison.py
python 30_task2_1_random_forest_tuning.py
python 31_task2_1_final_validation.py
python 32_task2_2_algorithm_comparison.py
python 33_task2_2_final_validation.py
python 34_create_final_report_figures.py
python 35_create_mlr_diagnostic_figures.py
python 36_export_task2_1_holdout_predictions.py
python 37_export_task2_2_holdout_predictions.py
python 38_create_combined_algorithm_comparison.py
python 39_create_task2_2_algorithm_comparison_png.py
```