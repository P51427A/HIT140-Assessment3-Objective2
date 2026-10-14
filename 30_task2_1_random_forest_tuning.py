"""
Tunes the Task 2.1 Random Forest model using the final margin-specific features.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split


INPUT = "task2_1_margin_feature_pool.csv"

RESULTS_OUT = "task2_1_random_forest_tuning_results.csv"
FINAL_OUT = "task2_1_random_forest_final_summary.csv"

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


def main():
    df = pd.read_csv(INPUT)

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found in final modelling columns")

    X = df[FEATURES]
    y = df[TARGET]

    # Keep the hold-out set untouched while tuning on training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = RandomForestRegressor(
        random_state=42,
        n_jobs=-1,
    )

    param_grid = {
        "n_estimators": [200, 400, 800],
        "max_depth": [2, 3, 4, None],
        "min_samples_leaf": [2, 3, 5],
        "max_features": [0.5, 0.75, 1.0],
    }

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_)
    results["cv_rmse_mean"] = -results["mean_test_score"]
    results["cv_rmse_sd"] = results["std_test_score"]

    keep = [
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_leaf",
        "param_max_features",
        "cv_rmse_mean",
        "cv_rmse_sd",
        "rank_test_score",
    ]

    results_out = results[keep].sort_values(
        ["rank_test_score", "cv_rmse_mean"]
    )
    results_out.to_csv(RESULTS_OUT, index=False)

    best_model = search.best_estimator_
    pred = best_model.predict(X_test)

    holdout_mae = mean_absolute_error(y_test, pred)
    holdout_mse = mean_squared_error(y_test, pred)
    holdout_rmse = np.sqrt(holdout_mse)
    holdout_r2 = r2_score(y_test, pred)

    mean_pred = np.repeat(y_train.mean(), len(y_test))
    baseline_mae = mean_absolute_error(y_test, mean_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
    baseline_r2 = r2_score(y_test, mean_pred)

    summary = pd.DataFrame(
        [
            {
                "model": "Tuned Random Forest",
                "n_estimators": search.best_params_["n_estimators"],
                "max_depth": search.best_params_["max_depth"],
                "min_samples_leaf": search.best_params_["min_samples_leaf"],
                "max_features": search.best_params_["max_features"],
                "training_cv_rmse": -search.best_score_,
                "holdout_mae": holdout_mae,
                "holdout_mse": holdout_mse,
                "holdout_rmse": holdout_rmse,
                "holdout_r2": holdout_r2,
            },
            {
                "model": "Mean baseline",
                "n_estimators": np.nan,
                "max_depth": np.nan,
                "min_samples_leaf": np.nan,
                "max_features": np.nan,
                "training_cv_rmse": np.nan,
                "holdout_mae": baseline_mae,
                "holdout_mse": mean_squared_error(y_test, mean_pred),
                "holdout_rmse": baseline_rmse,
                "holdout_r2": baseline_r2,
            },
        ]
    )

    summary.to_csv(FINAL_OUT, index=False)

    print("\nTASK 2.1 - RANDOM FOREST TUNING")
    print(f"Rows: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Hold-out rows: {len(X_test)}")

    print("\nBEST PARAMETERS")
    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    print(f"\nBest training-CV RMSE: {-search.best_score_:.4f}")

    print("\nHOLD-OUT PERFORMANCE")
    print(f"MAE:  {holdout_mae:.4f}")
    print(f"MSE:  {holdout_mse:.4f}")
    print(f"RMSE: {holdout_rmse:.4f}")
    print(f"R^2:  {holdout_r2:.4f}")

    print("\nMEAN BASELINE")
    print(f"MAE:  {baseline_mae:.4f}")
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"R^2:  {baseline_r2:.4f}")

    print("\nTOP 10 PARAMETER COMBINATIONS")
    print(
        results_out.head(10).round(4).to_string(index=False)
    )

    print(f"\nSaved: {RESULTS_OUT}")
    print(f"Saved: {FINAL_OUT}")


if __name__ == "__main__":
    main()
