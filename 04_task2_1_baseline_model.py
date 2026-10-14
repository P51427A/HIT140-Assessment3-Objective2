"""
Fits the baseline multiple linear regression model for Task 2.1.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

TEST_SIZE = 0.20
RANDOM_STATE = 42


def regression_metrics(y_true, y_pred, y_full):
    """Return standard regression metrics taught in Weeks 8-9."""
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    y_range = y_full.max() - y_full.min()
    nrmse = rmse / y_range if y_range != 0 else float("nan")

    r2 = metrics.r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2,
    }


def print_metrics(title, scores):
    print(f"\n{title}")
    print("-" * len(title))
    for name, value in scores.items():
        print(f"{name}: {value:.4f}")


def main():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    print("DATASET")
    print("-------")
    print("Rows:", len(df))
    print("Predictors:", len(FEATURES))
    print("Target:", TARGET)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("\nTRAIN/TEST SPLIT")
    print("----------------")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))

    # Baseline multiple linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_scores = regression_metrics(y_test, y_pred, y)

    # Mean-prediction baseline
    y_train_mean = y_train.mean()
    y_pred_baseline = [y_train_mean] * len(y_test)

    baseline_scores = regression_metrics(y_test, y_pred_baseline, y)

    print("\nMODEL COEFFICIENTS")
    print("------------------")
    print(f"Intercept: {model.intercept_:.4f}")

    coef_table = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_,
    }).sort_values(
        "coefficient",
        key=lambda s: s.abs(),
        ascending=False,
    )

    print(coef_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print_metrics("MULTIPLE LINEAR REGRESSION", model_scores)
    print_metrics("MEAN BASELINE", baseline_scores)

    print("\nIMPROVEMENT OVER BASELINE")
    print("-------------------------")
    print(
        "MAE improvement:",
        f"{baseline_scores['MAE'] - model_scores['MAE']:.4f}"
    )
    print(
        "RMSE improvement:",
        f"{baseline_scores['RMSE'] - model_scores['RMSE']:.4f}"
    )
    print(
        "R^2 improvement:",
        f"{model_scores['R2'] - baseline_scores['R2']:.4f}"
    )

    # Actual vs Predicted
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, y_pred)

    min_value = min(y_test.min(), y_pred.min())
    max_value = max(y_test.max(), y_pred.max())
    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
    )

    plt.title("Actual vs Predicted Goal Difference")
    plt.xlabel("Actual Goal Difference")
    plt.ylabel("Predicted Goal Difference")
    plt.tight_layout()
    plt.show()

    # Residual plot
    residuals = y_test - y_pred

    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals)
    plt.axhline(0)
    plt.title("Residuals vs Predicted Values")
    plt.xlabel("Predicted Goal Difference")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
