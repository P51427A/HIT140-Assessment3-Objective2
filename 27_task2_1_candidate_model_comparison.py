"""
Compares candidate 8-feature specifications for Task 2.1.
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
RESULTS_OUT = "task2_1_candidate_model_comparison.csv"
VIF_OUT = "task2_1_candidate_model_vif.csv"
CORR_OUT = "task2_1_predictor_pair_correlations.csv"

TARGET = "goal_difference_abs"

MODEL_A = [
    "fifa_rank_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "recent_scoring_consistency_diff_abs",
    "recent_ppg_diff_abs",
    "host_advantage",
    "recent_clean_sheet_rate_diff_abs",
    "recent_ga_avg_diff_abs",
    "recent_win_rate_diff_abs",
]

MODEL_B = [
    "fifa_rank_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "recent_scoring_consistency_diff_abs",
    "recent_ppg_diff_abs",
    "host_advantage",
    "rest_days_diff_abs",
    "recent_gd_variability_diff_abs",
    "knockout",
]

MODEL_C = [
    "fifa_rank_diff_abs",
    "prior_wc_matches_diff_abs",
    "rest_days_diff_abs",
    "host_advantage",
    "recent_scoring_consistency_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "recent_gd_variability_diff_abs",
    "recent_close_game_rate_diff_abs",
]

MODELS = {
    "A_top_univariate": MODEL_A,
    "B_reduced_redundancy": MODEL_B,
    "C_balanced_context": MODEL_C,
}


def make_model(features):
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

    baseline_pred = np.repeat(y_train.mean(), len(y_test))

    return {
        "model": name,
        "features": " | ".join(features),
        "holdout_mae": mean_absolute_error(y_test, pred),
        "holdout_rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "holdout_r2": r2_score(y_test, pred),
        "baseline_holdout_mae": mean_absolute_error(y_test, baseline_pred),
        "baseline_holdout_rmse": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "baseline_holdout_r2": r2_score(y_test, baseline_pred),
        "cv_mae_mean": -scores["test_mae"].mean(),
        "cv_rmse_mean": -scores["test_rmse"].mean(),
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_r2_sd": scores["test_r2"].std(),
    }


def calculate_vif(df, name, features):
    X = df[features].copy()

    # VIF is used only as a redundancy diagnostic here.
    rows = []
    for i, col in enumerate(features):
        rows.append(
            {
                "model": name,
                "feature": col,
                "vif": variance_inflation_factor(X.values, i),
            }
        )
    return rows


def main():
    df = pd.read_csv(INPUT)

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values")

    all_features = sorted(
        set(feature for features in MODELS.values() for feature in features)
    )

    # Pairwise predictor correlations
    corr = df[all_features].corr()
    pair_rows = []
    for i, a in enumerate(all_features):
        for b in all_features[i + 1:]:
            r = corr.loc[a, b]
            pair_rows.append(
                {
                    "feature_1": a,
                    "feature_2": b,
                    "correlation": r,
                    "abs_correlation": abs(r),
                }
            )

    pair_df = pd.DataFrame(pair_rows).sort_values(
        "abs_correlation",
        ascending=False,
    )
    pair_df.to_csv(CORR_OUT, index=False)

    results = []
    vif_rows = []

    for name, features in MODELS.items():
        if df[features].isna().any().any():
            raise ValueError(f"{name} contains missing predictor values")

        results.append(evaluate(df, name, features))
        vif_rows.extend(calculate_vif(df, name, features))

    results_df = pd.DataFrame(results).sort_values(
        ["cv_rmse_mean", "cv_mae_mean"]
    )
    vif_df = pd.DataFrame(vif_rows)

    results_df.to_csv(RESULTS_OUT, index=False)
    vif_df.to_csv(VIF_OUT, index=False)

    print("\nTASK 2.1 - CANDIDATE MODEL COMPARISON")
    print(f"Rows used: {len(df)}")
    print("All three models use exactly 8 explanatory variables.")

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

    print("\nLARGEST PAIRWISE PREDICTOR CORRELATIONS")
    print(pair_df.head(10).round(4).to_string(index=False))

    print("\nMAXIMUM VIF BY MODEL")
    print(
        vif_df.groupby("model")["vif"]
        .max()
        .sort_values()
        .round(3)
        .to_string()
    )

    print(f"\nSaved: {RESULTS_OUT}")
    print(f"Saved: {VIF_OUT}")
    print(f"Saved: {CORR_OUT}")


if __name__ == "__main__":
    main()
