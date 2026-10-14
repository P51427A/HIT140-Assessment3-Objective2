"""
Runs the final validation for Task 2.1 Model F.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "task2_1_model_f_creative_dataset.csv"
OUTPUT_FILE = BASE_DIR / "task2_1_final_model_f_dataset.csv"

TARGET = "goal_difference"

FEATURES = [
    "fifa_rank_diff",
    "prior_wc_matches_diff",
    "rest_days_diff_yj",
    "host_advantage",
    "recent_gd_trend_diff",
    "recent_scoring_consistency_diff",
    "recent_defensive_resilience_diff",
    "recent_close_game_rate_diff",
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


def calculate_metrics(y_true, y_pred, y_all):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    target_range = y_all.max() - y_all.min()
    nrmse = rmse / target_range if target_range != 0 else float("nan")

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2,
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("-" * len(title))
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


def main():
    df = pd.read_csv(INPUT_FILE)

    print("TASK 2.1 - FINAL MODEL F VALIDATION")
    print("===================================")

    if len(df) != 104:
        raise ValueError(f"Expected 104 rows; found {len(df)}")

    if len(FEATURES) != 8:
        raise ValueError("Model F must contain exactly 8 explanatory variables")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing modelling values found")

    print("Rows:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Explanatory variables:", len(FEATURES))
    print(
        "Missing modelling values:",
        int(df[FEATURES + [TARGET]].isna().sum().sum()),
    )

    print("\nFINAL MODEL F VARIABLES")
    print("-----------------------")
    for i, feature in enumerate(FEATURES, 1):
        print(f"{i}. {feature}")

    # ------------------------------------------------------------
    # Save the final Model F dataset
    # ------------------------------------------------------------
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {OUTPUT_FILE.name}")

    # ------------------------------------------------------------
    # 80/20 hold-out evaluation
    # ------------------------------------------------------------
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_metrics = calculate_metrics(
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

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_pred,
        y,
    )

    print("\nTRAIN / TEST SPLIT")
    print("------------------")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    print_metrics(
        "MODEL F - HOLD-OUT PERFORMANCE",
        model_metrics,
    )

    print_metrics(
        "MEAN BASELINE",
        baseline_metrics,
    )

    print("\nIMPROVEMENT OVER BASELINE")
    print("-------------------------")
    print(
        "MAE improvement:",
        f"{baseline_metrics['MAE'] - model_metrics['MAE']:.4f}",
    )
    print(
        "RMSE improvement:",
        f"{baseline_metrics['RMSE'] - model_metrics['RMSE']:.4f}",
    )
    print(
        "R^2 improvement:",
        f"{model_metrics['R2'] - baseline_metrics['R2']:.4f}",
    )

    # ------------------------------------------------------------
    # Full-data OLS residual diagnostics
    # ------------------------------------------------------------
    X_ols = sm.add_constant(X)
    ols_model = sm.OLS(y, X_ols).fit()

    residuals = ols_model.resid

    shapiro_stat, shapiro_p = shapiro(residuals)

    bp_stat, bp_p, _, _ = het_breuschpagan(
        residuals,
        ols_model.model.exog,
    )

    print("\nFULL-DATA OLS FIT")
    print("-----------------")
    print(f"R^2: {ols_model.rsquared:.4f}")
    print(f"Adjusted R^2: {ols_model.rsquared_adj:.4f}")

    print("\nRESIDUAL DIAGNOSTICS")
    print("--------------------")
    print(f"Shapiro-Wilk statistic: {shapiro_stat:.4f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"Breusch-Pagan statistic: {bp_stat:.4f}")
    print(f"Breusch-Pagan p-value: {bp_p:.4f}")

    # ------------------------------------------------------------
    # Standardised coefficients
    # ------------------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    std_model = LinearRegression()
    std_model.fit(X_scaled, y)

    coefficients = pd.DataFrame({
        "feature": FEATURES,
        "standardised_coefficient": std_model.coef_,
    })

    coefficients["absolute_effect"] = (
        coefficients["standardised_coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_effect",
        ascending=False,
    )

    print("\nSTANDARDISED COEFFICIENTS")
    print("-------------------------")
    print(
        coefficients[
            ["feature", "standardised_coefficient"]
        ].round(4).to_string(index=False)
    )

    print("\nP-VALUES")
    print("--------")
    print(
        ols_model.pvalues
        .drop("const")
        .sort_values()
        .round(4)
        .to_string()
    )

if __name__ == "__main__":
    main()
