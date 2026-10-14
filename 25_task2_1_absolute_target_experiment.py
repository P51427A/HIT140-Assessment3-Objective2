"""
Tests signed and absolute difference representations for Task 2.1.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


FINAL_F = "task2_1_final_model_f_dataset.csv"
MODEL_D = "task2_1_match_level_dataset_model_d.csv"
RAW_MATCHES = "matches_104_raw.csv"

RESULTS_FILE = "task2_1_abs_representation_results.csv"
CANDIDATE_FILE = "task2_1_absolute_candidate_dataset.csv"

TARGET = "goal_difference_abs"

FEATURES = [
    "fifa_rank_diff",
    "prior_wc_matches_diff",
    "rest_days_diff",
    "host_advantage",
    "recent_gd_trend_diff",
    "recent_scoring_consistency_diff",
    "recent_defensive_resilience_diff",
    "recent_close_game_rate_diff",
]

DIFFERENCE_FEATURES = [
    "fifa_rank_diff",
    "prior_wc_matches_diff",
    "rest_days_diff",
    "recent_gd_trend_diff",
    "recent_scoring_consistency_diff",
    "recent_defensive_resilience_diff",
    "recent_close_game_rate_diff",
]


def build_dataset():
    model_f = pd.read_csv(FINAL_F)
    model_d = pd.read_csv(MODEL_D)[["match_id", "rest_days_diff"]]
    raw = pd.read_csv(RAW_MATCHES)[
        ["match_id", "team_a_goals", "team_b_goals"]
    ]

    df = model_f.drop(columns=["rest_days_diff_yj", "goal_difference"]).merge(
        model_d, on="match_id", how="left"
    )
    df = df.merge(raw, on="match_id", how="left")

    df[TARGET] = (df["team_a_goals"] - df["team_b_goals"]).abs()

    if len(df) != 104:
        raise ValueError(f"Expected 104 rows, found {len(df)}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing values found in modelling columns")

    return df


def make_features(df, absolute_columns):
    X = df[FEATURES].copy()

    for column in absolute_columns:
        X[column] = X[column].abs()

    return X


def make_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "rest",
                PowerTransformer(method="yeo-johnson", standardize=True),
                ["rest_days_diff"],
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


def evaluate_specification(df, name, absolute_columns):
    X = make_features(df, absolute_columns)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    holdout_mae = mean_absolute_error(y_test, predictions)
    holdout_rmse = np.sqrt(mean_squared_error(y_test, predictions))
    holdout_r2 = r2_score(y_test, predictions)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        make_model(),
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )

    return {
        "specification": name,
        "absolute_features": ", ".join(absolute_columns) if absolute_columns else "none",
        "holdout_mae": holdout_mae,
        "holdout_rmse": holdout_rmse,
        "holdout_r2": holdout_r2,
        "cv_mae_mean": -scores["test_mae"].mean(),
        "cv_rmse_mean": -scores["test_rmse"].mean(),
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_r2_sd": scores["test_r2"].std(),
    }


def main():
    df = build_dataset()

    experiments = [
        ("all_signed", []),
        ("abs_fifa_rank", ["fifa_rank_diff"]),
        ("abs_prior_wc", ["prior_wc_matches_diff"]),
        ("abs_rest_days", ["rest_days_diff"]),
        ("abs_gd_trend", ["recent_gd_trend_diff"]),
        ("abs_scoring_consistency", ["recent_scoring_consistency_diff"]),
        ("abs_defensive_resilience", ["recent_defensive_resilience_diff"]),
        ("abs_close_game_rate", ["recent_close_game_rate_diff"]),
        ("all_differences_absolute", DIFFERENCE_FEATURES),
    ]

    results = [
        evaluate_specification(df, name, columns)
        for name, columns in experiments
    ]

    results_df = pd.DataFrame(results).sort_values(
        ["cv_rmse_mean", "cv_mae_mean"]
    )
    results_df.to_csv(RESULTS_FILE, index=False)

    candidate = df[
        [
            "match_id",
            "date",
            "stage",
            "team_a",
            "team_b",
            *FEATURES,
            TARGET,
        ]
    ].copy()

    for column in DIFFERENCE_FEATURES:
        candidate[column] = candidate[column].abs()

    candidate.to_csv(CANDIDATE_FILE, index=False)

    print("\nTASK 2.1 - ABSOLUTE GOAL DIFFERENCE")
    print(f"Rows: {len(df)}")
    print(f"Target range: {df[TARGET].min()} to {df[TARGET].max()}")

    print("\nREPRESENTATION COMPARISON")
    print(
        results_df[
            [
                "specification",
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

    print(f"\nSaved: {RESULTS_FILE}")
    print(f"Saved: {CANDIDATE_FILE}")


if __name__ == "__main__":
    main()
