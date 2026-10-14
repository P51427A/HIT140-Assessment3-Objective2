"""
Compares the V3 and V4 Task 2.2 models.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


BASE_DIR = Path(__file__).resolve().parent

V2_FILE = BASE_DIR / "task2_2_candidate_dataset_v2.csv"
V3_FILE = BASE_DIR / "task2_2_candidate_dataset_v3.csv"
V4_FILE = BASE_DIR / "task2_2_candidate_dataset_v4.csv"

TARGET = "goals_scored"
GROUP = "match_id"

V3_RAW_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "goal_expectancy_matchup",
    "scoring_consistency_rate",
    "opponent_high_concede_rate",
    "opponent_concede_volatility",
]

V4_RAW_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "weighted_attack_matchup",
    "scoring_reliability_index",
    "opponent_defensive_fragility",
    "opponent_concede_trend",
]


def metrics(y_true, y_pred, full_target):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    target_range = full_target.max() - full_target.min()
    nrmse = rmse / target_range if target_range != 0 else float("nan")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2_score(y_true, y_pred),
    }


def make_pipeline(features):
    others = [
        feature for feature in features
        if feature != "rest_days"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "rest_yj",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=False,
                ),
                ["rest_days"],
            ),
            (
                "other",
                "passthrough",
                others,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def evaluate_model(df, features):
    X = df[features]
    y = df[TARGET]
    groups = df[GROUP]

    holdout = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        holdout.split(
            X,
            y,
            groups=groups,
        )
    )

    model = make_pipeline(features)

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    holdout_pred = model.predict(
        X.iloc[test_idx]
    )

    holdout_scores = metrics(
        y.iloc[test_idx],
        holdout_pred,
        y,
    )

    gkf = GroupKFold(n_splits=5)

    fold_rows = []

    for fold, (train_i, test_i) in enumerate(
        gkf.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        fold_model = make_pipeline(features)

        fold_model.fit(
            X.iloc[train_i],
            y.iloc[train_i],
        )

        pred = fold_model.predict(
            X.iloc[test_i]
        )

        score = metrics(
            y.iloc[test_i],
            pred,
            y,
        )

        fold_rows.append({
            "fold": fold,
            "MAE": score["MAE"],
            "RMSE": score["RMSE"],
            "R2": score["R2"],
        })

    folds = pd.DataFrame(fold_rows)

    summary = {
        "holdout_MAE": holdout_scores["MAE"],
        "holdout_RMSE": holdout_scores["RMSE"],
        "holdout_R2": holdout_scores["R2"],
        "cv_MAE": folds["MAE"].mean(),
        "cv_RMSE": folds["RMSE"].mean(),
        "cv_R2": folds["R2"].mean(),
        "cv_R2_SD": folds["R2"].std(ddof=1),
    }

    return summary, folds


def merge_raw_rest(model_df, v2):
    raw_rest = v2[
        [
            "match_id",
            "team",
            "opponent",
            "rest_days",
        ]
    ].copy()

    # Remove transformed rest column if present.
    model_df = model_df.drop(
        columns=[
            column
            for column in ["rest_days_yj"]
            if column in model_df.columns
        ]
    )

    merged = model_df.merge(
        raw_rest,
        on=["match_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if merged["rest_days"].isna().any():
        raise ValueError(
            "Could not recover raw rest_days"
        )

    return merged


def main():
    v2 = pd.read_csv(V2_FILE)
    v3 = pd.read_csv(V3_FILE)
    v4 = pd.read_csv(V4_FILE)

    v3 = merge_raw_rest(
        v3,
        v2,
    )

    v4 = merge_raw_rest(
        v4,
        v2,
    )

    v3_summary, v3_folds = evaluate_model(
        v3,
        V3_RAW_FEATURES,
    )

    v4_summary, v4_folds = evaluate_model(
        v4,
        V4_RAW_FEATURES,
    )

    comparison = pd.DataFrame([
        {
            "model": "V3",
            **v3_summary,
        },
        {
            "model": "V4",
            **v4_summary,
        },
    ])

    print("TASK 2.2 - V3 VS V4")
    print("====================")

    print("\nMODEL COMPARISON")
    print("----------------")
    print(
        comparison
        .round(4)
        .to_string(index=False)
    )

    print("\nFOLD-BY-FOLD COMPARISON")
    print("-----------------------")

    for i in range(5):
        print(
            f"Fold {i+1}: "
            f"V3 RMSE={v3_folds.loc[i, 'RMSE']:.4f}, "
            f"R2={v3_folds.loc[i, 'R2']:.4f} | "
            f"V4 RMSE={v4_folds.loc[i, 'RMSE']:.4f}, "
            f"R2={v4_folds.loc[i, 'R2']:.4f}"
        )

if __name__ == "__main__":
    main()
