"""
Compares the Task 2.1 Model A, Model B and Model C specifications.
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


def evaluate(name, df, features):
    X = df[features]
    y = df[TARGET]

    X_sm = sm.add_constant(X)
    ols = sm.OLS(y, X_sm).fit()

    bp_p = het_breuschpagan(ols.resid, ols.model.exog)[1]

    vif_values = []
    for i, col in enumerate(X_sm.columns):
        if col == "const":
            continue
        vif_values.append(
            variance_inflation_factor(X_sm.values, i)
        )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

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
    }, ols


def main():
    original = pd.read_csv(ORIGINAL_FILE)
    model_c = pd.read_csv(MODEL_C_FILE)

    results = []

    a, a_ols = evaluate("Model A - original 8", original, MODEL_A)
    b, b_ols = evaluate("Model B - reduced 6", original, MODEL_B)
    c, c_ols = evaluate("Model C - revised 8", model_c, MODEL_C)

    results.extend([a, b, c])

    print("MODEL C CORRELATIONS")
    print("====================")
    corr = model_c[MODEL_C + [TARGET]].corr()
    print(
        corr[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda x: x.abs(), ascending=False)
        .round(3)
        .to_string()
    )

    print("\nMODEL C STRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("==================================================")
    found = False
    feature_corr = model_c[MODEL_C].corr()
    for i, first in enumerate(MODEL_C):
        for second in MODEL_C[i+1:]:
            r = feature_corr.loc[first, second]
            if abs(r) >= 0.70:
                found = True
                print(f"{first} vs {second}: r={r:.3f}")
    if not found:
        print("None")

    print("\nMODEL COMPARISON")
    print("================")
    table = pd.DataFrame(results)
    print(table.round(4).to_string(index=False))

    print("\nMODEL C P-VALUES")
    print("================")
    print(c_ols.pvalues.round(4).sort_values().to_string())

if __name__ == "__main__":
    main()
