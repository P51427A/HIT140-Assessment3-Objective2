"""
Evaluates Model E and creates the final Task 2.1 dataset.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "task2_1_match_level_dataset_model_d.csv"
OUTPUT_FILE = BASE_DIR / "task2_1_final_dataset.csv"

TARGET = "goal_difference"

RAW_FEATURES = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
    "prior_wc_matches_diff",
    "recent_gd_variability_diff",
]

FINAL_FEATURES = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff_yj",
    "host_advantage",
    "knockout",
    "prior_wc_matches_diff",
    "recent_gd_variability_diff",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def score_model(y_true, y_pred, y_all):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    y_range = y_all.max() - y_all.min()
    nrmse = rmse / y_range if y_range != 0 else float("nan")

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2,
    }


def print_scores(title, scores):
    print(f"\n{title}")
    print("-" * len(title))
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")


def main():
    df = pd.read_csv(INPUT_FILE)

    if len(df) != 104:
        raise ValueError(f"Expected 104 rows; found {len(df)}")

    print("TASK 2.1 - FINAL MODEL E")
    print("========================")
    print("Rows:", len(df))

    # ------------------------------------------------------------
    # Create the final export dataset
    # ------------------------------------------------------------
    export_df = df.copy()

    full_transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )

    export_df["rest_days_diff_yj"] = full_transformer.fit_transform(
        export_df[["rest_days_diff"]]
    ).ravel()

    output_columns = [
        "match_id",
        "date",
        "stage",
        "team_a",
        "team_b",
        *FINAL_FEATURES,
        TARGET,
    ]

    final_df = export_df[output_columns].copy()

    if len(FINAL_FEATURES) != 8:
        raise ValueError("Final model must contain exactly 8 explanatory variables")

    if final_df[FINAL_FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Final dataset contains missing modelling values")

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFINAL DATASET")
    print("-------------")
    print("Rows:", len(final_df))
    print("Unique matches:", final_df["match_id"].nunique())
    print("Explanatory variables:", len(FINAL_FEATURES))
    print(
        "Missing modelling values:",
        int(final_df[FINAL_FEATURES + [TARGET]].isna().sum().sum()),
    )
    print("Saved:", OUTPUT_FILE.name)

    print("\nFINAL 8 EXPLANATORY VARIABLES")
    print("-----------------------------")
    for i, feature in enumerate(FINAL_FEATURES, 1):
        print(f"{i}. {feature}")

    # ------------------------------------------------------------
    # Leakage-safe hold-out evaluation
    # ------------------------------------------------------------
    X = df[RAW_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "rest_yeojohnson",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=False,
                ),
                ["rest_days_diff"],
            ),
            (
                "other_features",
                "passthrough",
                [
                    "fifa_rank_diff",
                    "recent_gf_avg_diff",
                    "recent_ga_avg_diff",
                    "host_advantage",
                    "knockout",
                    "prior_wc_matches_diff",
                    "recent_gd_variability_diff",
                ],
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regression", LinearRegression()),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    model_scores = score_model(
        y_test,
        y_pred,
        y,
    )

    baseline_value = y_train.mean()
    baseline_pred = np.full(
        len(y_test),
        baseline_value,
        dtype=float,
    )

    baseline_scores = score_model(
        y_test,
        baseline_pred,
        y,
    )

    print("\nTRAIN / TEST SPLIT")
    print("------------------")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    print_scores(
        "MODEL E - HOLD-OUT PERFORMANCE",
        model_scores,
    )

    print_scores(
        "MEAN BASELINE",
        baseline_scores,
    )

    print("\nIMPROVEMENT OVER BASELINE")
    print("-------------------------")
    print(
        "MAE improvement:",
        f"{baseline_scores['MAE'] - model_scores['MAE']:.4f}",
    )
    print(
        "RMSE improvement:",
        f"{baseline_scores['RMSE'] - model_scores['RMSE']:.4f}",
    )
    print(
        "R^2 improvement:",
        f"{model_scores['R2'] - baseline_scores['R2']:.4f}",
    )

    # ------------------------------------------------------------
    # Standardised coefficients
    # ------------------------------------------------------------
    X_final = final_df[FINAL_FEATURES].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_final)

    standardised_model = LinearRegression()
    standardised_model.fit(X_scaled, y)

    coefficient_table = pd.DataFrame({
        "feature": FINAL_FEATURES,
        "standardised_coefficient": standardised_model.coef_,
    })

    coefficient_table["absolute_effect"] = (
        coefficient_table["standardised_coefficient"].abs()
    )

    coefficient_table = coefficient_table.sort_values(
        "absolute_effect",
        ascending=False,
    )

    print("\nSTANDARDISED COEFFICIENTS")
    print("-------------------------")
    print(
        coefficient_table[
            ["feature", "standardised_coefficient"]
        ].round(4).to_string(index=False)
    )

if __name__ == "__main__":
    main()
