"""
Tests a reduced Task 2.1 model to examine multicollinearity.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn import metrics
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "task2_1_match_level_dataset.csv"
TARGET = "goal_difference"

FULL_FEATURES = [
    "fifa_rank_diff",
    "recent_ppg_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "recent_clean_sheet_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]

REDUCED_FEATURES = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]


def get_cv_scores(X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    scores = cross_validate(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring=scoring,
    )

    mae = -scores["test_mae"]
    mse = -scores["test_mse"]
    rmse = np.sqrt(mse)
    r2 = scores["test_r2"]

    return {
        "fold_mae": mae,
        "fold_rmse": rmse,
        "fold_r2": r2,
        "mean_mae": mae.mean(),
        "sd_mae": mae.std(ddof=1),
        "mean_rmse": rmse.mean(),
        "sd_rmse": rmse.std(ddof=1),
        "mean_r2": r2.mean(),
        "sd_r2": r2.std(ddof=1),
    }


def fit_ols(X, y):
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    residuals = model.resid
    bp = het_breuschpagan(residuals, model.model.exog)

    vif_rows = []
    for i, column in enumerate(X_sm.columns):
        if column == "const":
            continue
        vif_rows.append({
            "feature": column,
            "VIF": variance_inflation_factor(X_sm.values, i)
        })

    return model, pd.DataFrame(vif_rows).sort_values("VIF", ascending=False), bp[1]


def print_model_results(title, features, df):
    X = df[features]
    y = df[TARGET]

    model, vif, bp_p = fit_ols(X, y)
    cv = get_cv_scores(X, y)

    print(f"\n{title}")
    print("=" * len(title))
    print("Predictors:", len(features))
    print("Features:")
    for feature in features:
        print(" -", feature)

    print("\nOLS FIT")
    print("-------")
    print(f"R^2: {model.rsquared:.4f}")
    print(f"Adjusted R^2: {model.rsquared_adj:.4f}")
    print(f"Breusch-Pagan p-value: {bp_p:.4f}")

    print("\nP-VALUES")
    print("--------")
    pvals = model.pvalues.drop("const").sort_values()
    print(pvals.round(4).to_string())

    print("\nVIF")
    print("---")
    print(vif.round(3).to_string(index=False))

    print("\n5-FOLD CV")
    print("---------")
    for i in range(5):
        print(
            f"Fold {i+1}: "
            f"MAE={cv['fold_mae'][i]:.4f}, "
            f"RMSE={cv['fold_rmse'][i]:.4f}, "
            f"R^2={cv['fold_r2'][i]:.4f}"
        )

    print("\nCV MEAN ± SD")
    print("------------")
    print(f"MAE:  {cv['mean_mae']:.4f} ± {cv['sd_mae']:.4f}")
    print(f"RMSE: {cv['mean_rmse']:.4f} ± {cv['sd_rmse']:.4f}")
    print(f"R^2:  {cv['mean_r2']:.4f} ± {cv['sd_r2']:.4f}")

    return {
        "name": title,
        "n_features": len(features),
        "r2": model.rsquared,
        "adj_r2": model.rsquared_adj,
        "bp_p": bp_p,
        "max_vif": vif["VIF"].max(),
        "cv_mae": cv["mean_mae"],
        "cv_rmse": cv["mean_rmse"],
        "cv_r2": cv["mean_r2"],
    }


def main():
    df = pd.read_csv(DATA_FILE)

    print("TASK 2.1 - MULTICOLLINEARITY REDUCTION EXPERIMENT")
    print("=================================================")
    print("Rows:", len(df))

    full = print_model_results(
        "MODEL A - ORIGINAL 8 VARIABLES",
        FULL_FEATURES,
        df,
    )

    reduced = print_model_results(
        "MODEL B - REDUCED 6 VARIABLES",
        REDUCED_FEATURES,
        df,
    )

    comparison = pd.DataFrame([full, reduced])

    print("\nMODEL COMPARISON")
    print("================")
    print(
        comparison[
            [
                "name",
                "n_features",
                "r2",
                "adj_r2",
                "max_vif",
                "bp_p",
                "cv_mae",
                "cv_rmse",
                "cv_r2",
            ]
        ].round(4).to_string(index=False)
    )

if __name__ == "__main__":
    main()
