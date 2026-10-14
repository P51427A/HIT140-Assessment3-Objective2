"""
Runs diagnostics and cross-validation for the Task 2.1 regression model.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "task2_1_match_level_dataset.csv"

FEATURES = [
    "fifa_rank_diff",
    "recent_ppg_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "recent_clean_sheet_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]

TARGET = "goal_difference"


def main():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    print("TASK 2.1 - ORIGINAL 8-VARIABLE MODEL")
    print("====================================")
    print("Rows:", len(df))
    print("Predictors:", len(FEATURES))

    # ------------------------------------------------------------
    # 1. STATSMODELS OLS
    # ------------------------------------------------------------
    X_sm = sm.add_constant(X)
    ols_model = sm.OLS(y, X_sm).fit()

    print("\nOLS MODEL SUMMARY")
    print("-----------------")
    print(ols_model.summary())

    print("\nKEY OLS RESULTS")
    print("---------------")
    print(f"R^2: {ols_model.rsquared:.4f}")
    print(f"Adjusted R^2: {ols_model.rsquared_adj:.4f}")

    coef_table = pd.DataFrame({
        "coefficient": ols_model.params,
        "p_value": ols_model.pvalues,
    })
    print("\nCOEFFICIENTS AND P-VALUES")
    print("-------------------------")
    print(coef_table.round(4).to_string())

    # ------------------------------------------------------------
    # 2. MULTICOLLINEARITY - VIF
    # ------------------------------------------------------------
    vif_rows = []

    X_vif = sm.add_constant(X)

    for i, column in enumerate(X_vif.columns):
        if column == "const":
            continue

        vif_rows.append({
            "feature": column,
            "VIF": variance_inflation_factor(X_vif.values, i)
        })

    vif_table = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)

    print("\nVARIANCE INFLATION FACTORS (VIF)")
    print("--------------------------------")
    print(vif_table.round(3).to_string(index=False))

    # ------------------------------------------------------------
    # 3. RESIDUAL DIAGNOSTICS
    # ------------------------------------------------------------
    residuals = ols_model.resid
    fitted = ols_model.fittedvalues

    shapiro_stat, shapiro_p = shapiro(residuals)

    bp_test = het_breuschpagan(
        residuals,
        ols_model.model.exog,
    )
    bp_stat, bp_p, _, _ = bp_test

    print("\nRESIDUAL DIAGNOSTICS")
    print("--------------------")
    print(f"Shapiro-Wilk statistic: {shapiro_stat:.4f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Breusch-Pagan statistic: {bp_stat:.4f}")
    print(f"Breusch-Pagan p-value: {bp_p:.4f}")

    if shapiro_p >= 0.05:
        print("Residual normality: no strong evidence against normality.")
    else:
        print("Residual normality: evidence of non-normal residuals.")

    if bp_p >= 0.05:
        print("Homoscedasticity: no strong evidence of unequal variance.")
    else:
        print("Homoscedasticity: evidence of unequal residual variance.")

    # ------------------------------------------------------------
    # 4. 5-FOLD CROSS-VALIDATION
    # ------------------------------------------------------------
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    model = LinearRegression()

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
    )

    fold_mae = -scores["test_mae"]
    fold_mse = -scores["test_mse"]
    fold_rmse = np.sqrt(fold_mse)
    fold_r2 = scores["test_r2"]

    print("\n5-FOLD CROSS-VALIDATION")
    print("-----------------------")

    for i in range(5):
        print(
            f"Fold {i+1}: "
            f"MAE={fold_mae[i]:.4f}, "
            f"RMSE={fold_rmse[i]:.4f}, "
            f"R^2={fold_r2[i]:.4f}"
        )

    print("\nCROSS-VALIDATION MEAN ± SD")
    print("--------------------------")
    print(f"MAE:  {fold_mae.mean():.4f} ± {fold_mae.std(ddof=1):.4f}")
    print(f"RMSE: {fold_rmse.mean():.4f} ± {fold_rmse.std(ddof=1):.4f}")
    print(f"R^2:  {fold_r2.mean():.4f} ± {fold_r2.std(ddof=1):.4f}")

    # ------------------------------------------------------------
    # 5. TWO USEFUL DIAGNOSTIC PLOTS
    # ------------------------------------------------------------
    sm.qqplot(residuals, line="45", fit=True)
    plt.title("Q-Q Plot of OLS Residuals")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, residuals)
    plt.axhline(0)
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Goal Difference")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
