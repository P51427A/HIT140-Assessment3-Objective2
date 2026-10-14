"""
Compares multiple regression algorithms for Task 2.1.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from sklearn.tree import DecisionTreeRegressor


INPUT = "task2_1_margin_feature_pool.csv"
OUTPUT = "task2_1_algorithm_comparison.csv"

TARGET = "goal_difference_abs"

MODEL_D = [
    "fifa_rank_diff_abs",
    "recent_net_gd_gap_abs",
    "recent_scoring_consistency_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "rest_days_diff_abs",
    "recent_gd_variability_diff_abs",
    "host_advantage",
    "knockout",
]

MODEL_E = [
    "fifa_rank_gap_log",
    "recent_net_gd_gap_abs",
    "two_way_goal_profile_gap",
    "recent_form_gap_index",
    "consistency_balance_gap",
    "rest_days_diff_abs",
    "host_advantage",
    "knockout",
]

FEATURE_SETS = {
    "D_margin_specific": MODEL_D,
    "E_margin_extended": MODEL_E,
}


def linear_pipeline(features):
    if "rest_days_diff_abs" in features:
        prep = ColumnTransformer(
            [
                (
                    "rest",
                    PowerTransformer(method="yeo-johnson", standardize=True),
                    ["rest_days_diff_abs"],
                )
            ],
            remainder="passthrough",
        )
        return Pipeline(
            [
                ("preprocess", prep),
                ("model", LinearRegression()),
            ]
        )

    return LinearRegression()


def algorithms(features):
    return {
        "LinearRegression": linear_pipeline(features),
        "DecisionTree": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }


def main():
    df = pd.read_csv(INPUT)

    results = []

    for feature_set_name, features in FEATURE_SETS.items():
        X = df[features]
        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

        baseline = np.repeat(y_train.mean(), len(y_test))
        results.append(
            {
                "feature_set": feature_set_name,
                "algorithm": "MeanBaseline",
                "holdout_mae": mean_absolute_error(y_test, baseline),
                "holdout_rmse": np.sqrt(mean_squared_error(y_test, baseline)),
                "holdout_r2": r2_score(y_test, baseline),
                "cv_mae_mean": np.nan,
                "cv_rmse_mean": np.nan,
                "cv_r2_mean": np.nan,
                "cv_r2_sd": np.nan,
            }
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        for algorithm_name, model in algorithms(features).items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

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

            results.append(
                {
                    "feature_set": feature_set_name,
                    "algorithm": algorithm_name,
                    "holdout_mae": mean_absolute_error(y_test, pred),
                    "holdout_rmse": np.sqrt(mean_squared_error(y_test, pred)),
                    "holdout_r2": r2_score(y_test, pred),
                    "cv_mae_mean": -scores["test_mae"].mean(),
                    "cv_rmse_mean": -scores["test_rmse"].mean(),
                    "cv_r2_mean": scores["test_r2"].mean(),
                    "cv_r2_sd": scores["test_r2"].std(),
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT, index=False)

    print("\nTASK 2.1 - ALGORITHM COMPARISON")
    print(f"Rows used: {len(df)}")

    ranked = results_df[
        results_df["algorithm"] != "MeanBaseline"
    ].sort_values(
        ["cv_rmse_mean", "cv_mae_mean"]
    )

    print("\nMODEL PERFORMANCE")
    print(
        ranked[
            [
                "feature_set",
                "algorithm",
                "holdout_mae",
                "holdout_rmse",
                "holdout_r2",
                "cv_mae_mean",
                "cv_rmse_mean",
                "cv_r2_mean",
                "cv_r2_sd",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nMEAN BASELINE")
    print(
        results_df[
            results_df["algorithm"] == "MeanBaseline"
        ][
            [
                "feature_set",
                "holdout_mae",
                "holdout_rmse",
                "holdout_r2",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
