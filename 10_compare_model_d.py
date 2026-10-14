"""
Compares the Task 2.1 Model A, B, C and D specifications.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

ORIGINAL_FILE = BASE_DIR / "task2_1_match_level_dataset.csv"
MODEL_C_FILE = BASE_DIR / "task2_1_match_level_dataset_model_c.csv"
MODEL_D_FILE = BASE_DIR / "task2_1_match_level_dataset_model_d.csv"

TARGET = "goal_difference"

MODEL_A = [
    "fifa_rank_diff",
    "recent_ppg_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "recent_clean_sheet_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]

MODEL_B = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]

MODEL_C = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
    "recent_win_rate_diff",
    "prior_wc_matches_diff",
]

MODEL_D = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
    "prior_wc_matches_diff",
    "recent_gd_variability_diff",
]


def evaluate(name, df, features):
    X = df[features]
    y = df[TARGET]

    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()

    bp_p = het_breuschpagan(
        ols.resid,
        ols.model.exog,
    )[1]

    vif_values = []
    for i, column in enumerate(X_sm.columns):
        if column == "const":
            continue
        vif_values.append(
            variance_inflation_factor(X_sm.values, i)
        )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    scores = cross_validate(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
    )

    mae = -scores["test_mae"]
    rmse = np.sqrt(-scores["test_mse"])
    r2 = scores["test_r2"]

    return {
        "model": name,
        "features": len(features),
        "r2": ols.rsquared,
        "adjusted_r2": ols.rsquared_adj,
        "max_vif": max(vif_values),
        "breusch_pagan_p": bp_p,
        "cv_mae": mae.mean(),
        "cv_rmse": rmse.mean(),
        "cv_r2": r2.mean(),
        "cv_r2_sd": r2.std(ddof=1),
    }, ols, pd.DataFrame({
        "feature": [
            col for col in X_sm.columns if col != "const"
        ],
        "vif": vif_values,
    }).sort_values("vif", ascending=False)


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    model_c = pd.read_csv(MODEL_C_FILE)
    model_d = pd.read_csv(MODEL_D_FILE)

    a, _, _ = evaluate(
        "Model A - original 8",
        original,
        MODEL_A,
    )
    b, _, _ = evaluate(
        "Model B - reduced 6",
        original,
        MODEL_B,
    )
    c, _, _ = evaluate(
        "Model C - revised 8",
        model_c,
        MODEL_C,
    )
    d, d_ols, d_vif = evaluate(
        "Model D - revised 8",
        model_d,
        MODEL_D,
    )

    print("MODEL D TARGET CORRELATIONS")
    print("===========================")

    corr = model_d[MODEL_D + [TARGET]].corr()

    print(
        corr[TARGET]
        .drop(TARGET)
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False,
        )
        .round(3)
        .to_string()
    )

    print("\nMODEL D STRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("==================================================")

    feature_corr = model_d[MODEL_D].corr()
    found = False

    for i, first in enumerate(MODEL_D):
        for second in MODEL_D[i + 1:]:
            r = feature_corr.loc[first, second]

            if abs(r) >= 0.70:
                found = True
                print(f"{first} vs {second}: r={r:.3f}")

    if not found:
        print("None")

    print("\nMODEL D VIF")
    print("===========")
    print(
        d_vif.round(3).to_string(index=False)
    )

    print("\nMODEL COMPARISON")
    print("================")

    comparison = pd.DataFrame(
        [a, b, c, d]
    )

    print(
        comparison.round(4).to_string(index=False)
    )

    print("\nMODEL D P-VALUES")
    print("================")
    print(
        d_ols.pvalues
        .sort_values()
        .round(4)
        .to_string()
    )

if __name__ == "__main__":
    main()
