"""
Creates MLR diagnostic figures for the Objective 2 appendix.

Outputs:
- task2_1_mlr_residuals_vs_fitted.png
- task2_1_mlr_qq_plot.png
- task2_2_mlr_residuals_vs_fitted.png
- task2_2_mlr_qq_plot.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


TASK21_DATA = "task2_1_final_abs_dataset.csv"
TASK22_DATA = "task2_2_final_model_dataset.csv"

TASK21_TARGET = "goal_difference_abs"
TASK22_TARGET = "goals_scored"

TASK21_FEATURES = [
    "fifa_rank_diff_abs",
    "recent_net_gd_gap_abs",
    "recent_scoring_consistency_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "rest_days_diff_abs",
    "recent_gd_variability_diff_abs",
    "host_advantage",
    "knockout",
]

TASK22_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days_yj",
    "host_team",
    "weighted_attack_matchup",
    "scoring_reliability_index",
    "opponent_defensive_fragility",
    "opponent_concede_trend",
]


def save(fig, filename):
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_task21_mlr():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "rest",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=True,
                ),
                ["rest_days_diff_abs"],
            )
        ],
        remainder="passthrough",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("model", LinearRegression()),
        ]
    )


def standardise_residuals(residuals):
    residuals = np.asarray(residuals, dtype=float)
    sd = residuals.std(ddof=1)

    if sd == 0:
        return residuals

    return (residuals - residuals.mean()) / sd


def residuals_vs_fitted(fitted, residuals, title, output):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.scatter(
        fitted,
        residuals,
        s=40,
        alpha=0.80,
    )
    ax.axhline(
        0,
        linewidth=1.0,
    )

    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(title, fontsize=12, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, output)


def qq_plot(standardised_residuals, title, output):
    (theoretical, ordered), (slope, intercept, _) = stats.probplot(
        standardised_residuals,
        dist="norm",
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.4))

    ax.scatter(
        theoretical,
        ordered,
        s=32,
        alpha=0.85,
    )
    ax.plot(
        theoretical,
        slope * theoretical + intercept,
        linewidth=1.0,
    )

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Standardised residuals")
    ax.set_title(title, fontsize=12, pad=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, output)


def task21_diagnostics():
    df = pd.read_csv(TASK21_DATA)

    required = TASK21_FEATURES + [TASK21_TARGET]
    if df[required].isna().any().any():
        raise ValueError("Task 2.1 contains missing modelling values.")

    X = df[TASK21_FEATURES]
    y = df[TASK21_TARGET]

    model = make_task21_mlr()
    model.fit(X, y)

    fitted = model.predict(X)
    residuals = y.to_numpy() - fitted
    standardised = standardise_residuals(residuals)

    residuals_vs_fitted(
        fitted,
        residuals,
        "Task 2.1 MLR: residuals vs fitted",
        "task2_1_mlr_residuals_vs_fitted.png",
    )

    qq_plot(
        standardised,
        "Task 2.1 MLR: normal Q-Q plot",
        "task2_1_mlr_qq_plot.png",
    )


def task22_diagnostics():
    df = pd.read_csv(TASK22_DATA)

    required = TASK22_FEATURES + [TASK22_TARGET]
    if df[required].isna().any().any():
        raise ValueError("Task 2.2 contains missing modelling values.")

    X = df[TASK22_FEATURES]
    y = df[TASK22_TARGET]

    model = LinearRegression()
    model.fit(X, y)

    fitted = model.predict(X)
    residuals = y.to_numpy() - fitted
    standardised = standardise_residuals(residuals)

    residuals_vs_fitted(
        fitted,
        residuals,
        "Task 2.2 MLR: residuals vs fitted",
        "task2_2_mlr_residuals_vs_fitted.png",
    )

    qq_plot(
        standardised,
        "Task 2.2 MLR: normal Q-Q plot",
        "task2_2_mlr_qq_plot.png",
    )


def main():
    task21_diagnostics()
    task22_diagnostics()

    print("Saved:")
    print("task2_1_mlr_residuals_vs_fitted.png")
    print("task2_1_mlr_qq_plot.png")
    print("task2_2_mlr_residuals_vs_fitted.png")
    print("task2_2_mlr_qq_plot.png")


if __name__ == "__main__":
    main()
