"""
Final validation and export for Task 2.1.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


INPUT = "task2_1_margin_feature_pool.csv"

FINAL_DATASET = "task2_1_final_abs_dataset.csv"
SUMMARY_OUT = "task2_1_final_validation_results.csv"
FOLDS_OUT = "task2_1_final_cv_folds.csv"
IMPORTANCE_OUT = "task2_1_random_forest_feature_importance.csv"

TARGET = "goal_difference_abs"

FEATURES = [
    "fifa_rank_diff_abs",
    "recent_net_gd_gap_abs",
    "recent_scoring_consistency_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "rest_days_diff_abs",
    "recent_gd_variability_diff_abs",
    "host_advantage",
    "knockout",
]


def make_linear_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "rest",
                PowerTransformer(method="yeo-johnson", standardize=True),
                ["rest_days_diff_abs"],
            )
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", LinearRegression()),
        ]
    )


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=4,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def metric_row(name, y_true, pred):
    mse = mean_squared_error(y_true, pred)

    return {
        "model": name,
        "mae": mean_absolute_error(y_true, pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, pred),
    }


def main():
    df = pd.read_csv(INPUT)

    required = ["match_id", "date", "stage", "team_a", "team_b", TARGET, *FEATURES]
    missing_columns = [c for c in required if c not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    final_df = df[required].copy()

    if len(final_df) != 104:
        raise ValueError(f"Expected 104 rows, found {len(final_df)}")

    if final_df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found in final modelling columns")

    if len(FEATURES) != 8:
        raise ValueError("Task 2.1 must use exactly 8 explanatory variables")

    final_df.to_csv(FINAL_DATASET, index=False)

    X = final_df[FEATURES]
    y = final_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    models = {
        "Multiple Linear Regression": make_linear_model(),
        "Random Forest": make_random_forest(),
    }

    holdout_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        row = metric_row(name, y_test, pred)
        row["evaluation"] = "holdout"
        holdout_rows.append(row)

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    baseline_row = metric_row("Mean Baseline", y_test, baseline_pred)
    baseline_row["evaluation"] = "holdout"
    holdout_rows.append(baseline_row)

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    fold_rows = []
    cv_summary_rows = []

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            n_jobs=-1,
        )

        fold_mae = -scores["test_mae"]
        fold_rmse = -scores["test_rmse"]
        fold_r2 = scores["test_r2"]

        for fold, (mae, rmse, r2) in enumerate(
            zip(fold_mae, fold_rmse, fold_r2),
            start=1,
        ):
            fold_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                }
            )

        cv_summary_rows.append(
            {
                "model": name,
                "evaluation": "5-fold CV mean",
                "mae": fold_mae.mean(),
                "mse": np.nan,
                "rmse": fold_rmse.mean(),
                "r2": fold_r2.mean(),
                "r2_sd": fold_r2.std(),
            }
        )

    # Mean-baseline cross-validation
    baseline_fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]
        pred = np.repeat(y_train_fold.mean(), len(y_test_fold))

        row = metric_row("Mean Baseline", y_test_fold, pred)
        row["fold"] = fold
        baseline_fold_rows.append(row)

        fold_rows.append(
            {
                "model": "Mean Baseline",
                "fold": fold,
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
            }
        )

    baseline_cv = pd.DataFrame(baseline_fold_rows)
    cv_summary_rows.append(
        {
            "model": "Mean Baseline",
            "evaluation": "5-fold CV mean",
            "mae": baseline_cv["mae"].mean(),
            "mse": np.nan,
            "rmse": baseline_cv["rmse"].mean(),
            "r2": baseline_cv["r2"].mean(),
            "r2_sd": baseline_cv["r2"].std(),
        }
    )

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df["r2_sd"] = np.nan

    cv_summary_df = pd.DataFrame(cv_summary_rows)

    summary = pd.concat(
        [
            holdout_df[
                ["model", "evaluation", "mae", "mse", "rmse", "r2", "r2_sd"]
            ],
            cv_summary_df[
                ["model", "evaluation", "mae", "mse", "rmse", "r2", "r2_sd"]
            ],
        ],
        ignore_index=True,
    )

    summary.to_csv(SUMMARY_OUT, index=False)
    pd.DataFrame(fold_rows).to_csv(FOLDS_OUT, index=False)

    # Fit the selected predictive model on all rows for feature importance.
    rf_final = make_random_forest()
    rf_final.fit(X, y)

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance": rf_final.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(IMPORTANCE_OUT, index=False)

    print("\nTASK 2.1 - FINAL VALIDATION")
    print(f"Rows: {len(final_df)}")
    print(f"Explanatory variables: {len(FEATURES)}")
    print(f"Missing modelling values: {int(final_df[FEATURES + [TARGET]].isna().sum().sum())}")
    print(f"Target: {TARGET}")

    print("\nFINAL EXPLANATORY VARIABLES")
    for i, feature in enumerate(FEATURES, start=1):
        print(f"{i}. {feature}")

    print("\nFINAL PERFORMANCE SUMMARY")
    print(
        summary.round(4).to_string(index=False)
    )

    print("\nRANDOM FOREST FEATURE IMPORTANCE")
    print(
        importance.round(4).to_string(index=False)
    )

    print(f"\nSaved: {FINAL_DATASET}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"Saved: {FOLDS_OUT}")
    print(f"Saved: {IMPORTANCE_OUT}")


if __name__ == "__main__":
    main()
