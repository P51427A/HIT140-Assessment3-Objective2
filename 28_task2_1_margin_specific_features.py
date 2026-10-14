"""
Builds and compares margin-specific Task 2.1 feature sets.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor


INPUT = "task2_1_feature_pool_abs.csv"
OUTPUT_DATA = "task2_1_margin_feature_pool.csv"
OUTPUT_RESULTS = "task2_1_margin_model_comparison.csv"
OUTPUT_VIF = "task2_1_margin_model_vif.csv"

TARGET = "goal_difference_abs"


def build_margin_features(df):
    out = df.copy()

    # Difference in recent average goal difference between the two teams.
    # Using signed GF and GA differences:
    # (GF_A - GF_B) - (GA_A - GA_B)
    out["recent_net_gd_gap_abs"] = (
        out["recent_gf_avg_diff_abs"] * 0
    )

    # Reconstruct from signed columns if they are available in the source pool.
    # If not, they will be added below from the original Task 2.1 dataset.
    return out


def make_model(features):
    transformers = []

    if "rest_days_diff_abs" in features:
        transformers.append(
            (
                "rest",
                PowerTransformer(method="yeo-johnson", standardize=True),
                ["rest_days_diff_abs"],
            )
        )

    if transformers:
        prep = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
        )
        return Pipeline(
            [
                ("preprocess", prep),
                ("model", LinearRegression()),
            ]
        )

    return LinearRegression()


def evaluate(df, name, features):
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = make_model(features)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        make_model(features),
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
        "model": name,
        "features": " | ".join(features),
        "holdout_mae": mean_absolute_error(y_test, pred),
        "holdout_rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "holdout_r2": r2_score(y_test, pred),
        "cv_mae_mean": -scores["test_mae"].mean(),
        "cv_rmse_mean": -scores["test_rmse"].mean(),
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_r2_sd": scores["test_r2"].std(),
    }


def vif_rows(df, model_name, features):
    X = df[features].copy()
    rows = []

    for i, col in enumerate(features):
        rows.append(
            {
                "model": model_name,
                "feature": col,
                "vif": variance_inflation_factor(X.values, i),
            }
        )
    return rows


def main():
    pool = pd.read_csv(INPUT)

    # Bring back the signed recent GF and GA differences so a true recent
    # goal-difference gap can be calculated.
    signed = pd.read_csv("task2_1_match_level_dataset.csv")[
        [
            "match_id",
            "recent_gf_avg_diff",
            "recent_ga_avg_diff",
            "recent_ppg_diff",
            "recent_clean_sheet_rate_diff",
        ]
    ]

    df = pool.merge(signed, on="match_id", how="left")

    # Margin-specific engineered features
    df["recent_net_gd_gap_abs"] = (
        df["recent_gf_avg_diff"] - df["recent_ga_avg_diff"]
    ).abs()

    df["two_way_goal_profile_gap"] = (
        df["recent_gf_avg_diff"].abs()
        + df["recent_ga_avg_diff"].abs()
    )

    # Ranking gaps may have diminishing effect at very large values.
    df["fifa_rank_gap_log"] = np.log1p(df["fifa_rank_diff_abs"])

    # Combined recent form mismatch.
    df["recent_form_gap_index"] = (
        df["recent_ppg_diff"].abs()
        + df["recent_clean_sheet_rate_diff"].abs()
    )

    # Difference in balance between scoring consistency and defensive resilience.
    df["consistency_balance_gap"] = (
        df["recent_scoring_consistency_diff_abs"]
        + df["recent_defensive_resilience_diff_abs"]
    )

    engineered = [
        "recent_net_gd_gap_abs",
        "two_way_goal_profile_gap",
        "fifa_rank_gap_log",
        "recent_form_gap_index",
        "consistency_balance_gap",
    ]

    if df[engineered].isna().any().any():
        print("\nMissing values in engineered features:")
        print(df[engineered].isna().sum())
        df = df.dropna(subset=engineered).copy()

    # Previous strongest candidate from Step 27
    model_b = [
        "fifa_rank_diff_abs",
        "recent_defensive_resilience_diff_abs",
        "recent_scoring_consistency_diff_abs",
        "recent_ppg_diff_abs",
        "host_advantage",
        "rest_days_diff_abs",
        "recent_gd_variability_diff_abs",
        "knockout",
    ]

    # Replace overlapping form variables with margin-specific features
    model_d = [
        "fifa_rank_diff_abs",
        "recent_net_gd_gap_abs",
        "recent_scoring_consistency_diff_abs",
        "recent_defensive_resilience_diff_abs",
        "rest_days_diff_abs",
        "recent_gd_variability_diff_abs",
        "host_advantage",
        "knockout",
    ]

    # Use log rank gap and broader mismatch measures
    model_e = [
        "fifa_rank_gap_log",
        "recent_net_gd_gap_abs",
        "two_way_goal_profile_gap",
        "recent_form_gap_index",
        "consistency_balance_gap",
        "rest_days_diff_abs",
        "host_advantage",
        "knockout",
    ]

    models = {
        "B_previous_best": model_b,
        "D_margin_specific": model_d,
        "E_margin_extended": model_e,
    }

    results = []
    vif = []

    for name, features in models.items():
        results.append(evaluate(df, name, features))
        vif.extend(vif_rows(df, name, features))

    results_df = pd.DataFrame(results).sort_values(
        ["cv_rmse_mean", "cv_mae_mean"]
    )
    vif_df = pd.DataFrame(vif)

    df.to_csv(OUTPUT_DATA, index=False)
    results_df.to_csv(OUTPUT_RESULTS, index=False)
    vif_df.to_csv(OUTPUT_VIF, index=False)

    print("\nTASK 2.1 - MARGIN-SPECIFIC FEATURE EXPERIMENT")
    print(f"Rows used: {len(df)}")

    print("\nENGINEERED FEATURE CORRELATIONS WITH TARGET")
    for col in engineered:
        print(f"{col:30s} {df[col].corr(df[TARGET]): .4f}")

    print("\nMODEL PERFORMANCE")
    print(
        results_df[
            [
                "model",
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

    print("\nMAXIMUM VIF BY MODEL")
    print(
        vif_df.groupby("model")["vif"]
        .max()
        .sort_values()
        .round(3)
        .to_string()
    )

    print(f"\nSaved: {OUTPUT_DATA}")
    print(f"Saved: {OUTPUT_RESULTS}")
    print(f"Saved: {OUTPUT_VIF}")


if __name__ == "__main__":
    main()
