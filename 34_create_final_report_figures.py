"""
Creates final MLR-focused report figures for Objective 2.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler


TASK21_DATA = "task2_1_final_abs_dataset.csv"
TASK22_DATA = "task2_2_final_model_dataset.csv"
TASK22_COEF = "task2_2_final_coefficients.csv"

OUT1 = "figure_task2_1_mlr_actual_vs_predicted.png"
OUT2 = "figure_task2_1_mlr_standardised_coefficients.png"
OUT3 = "figure_task2_2_mlr_actual_vs_predicted.png"
OUT4 = "figure_task2_2_mlr_standardised_coefficients.png"

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

LABELS = {
    "fifa_rank_diff_abs": "FIFA ranking gap",
    "recent_net_gd_gap_abs": "Recent net goal-difference gap",
    "recent_scoring_consistency_diff_abs": "Scoring consistency gap",
    "recent_defensive_resilience_diff_abs": "Defensive resilience gap",
    "rest_days_diff_abs": "Rest-days gap",
    "recent_gd_variability_diff_abs": "Goal-difference variability gap",
    "host_advantage": "Host advantage",
    "knockout": "Knockout stage",
    "fifa_rank": "FIFA ranking",
    "prior_wc_matches": "Prior World Cup matches",
    "rest_days_yj": "Rest days",
    "host_team": "Host team",
    "weighted_attack_matchup": "Weighted attack matchup",
    "scoring_reliability_index": "Scoring reliability",
    "opponent_defensive_fragility": "Opponent defensive fragility",
    "opponent_concede_trend": "Opponent defensive trend",
}


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_task21_mlr():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "rest",
                PowerTransformer(method="yeo-johnson", standardize=True),
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


def task21_actual_vs_predicted():
    df = pd.read_csv(TASK21_DATA)

    X = df[TASK21_FEATURES]
    y = df["goal_difference_abs"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    model = make_task21_mlr()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    line_min = min(float(y_test.min()), float(pred.min()))
    line_max = max(float(y_test.max()), float(pred.max()))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y_test, pred, s=45, alpha=0.85)
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel("Actual absolute goal difference")
    ax.set_ylabel("Predicted absolute goal difference")
    ax.set_title(
        "Task 2.1 MLR: actual vs predicted",
        fontsize=12,
        pad=8,
    )

    ax.text(
        0.03,
        0.95,
        f"Hold-out RMSE = {rmse:.3f}\nHold-out $R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    plot_min = min(0.0, line_min) - 0.2
    plot_max = line_max + 0.2
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal", adjustable="box")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, OUT1)


def task21_standardised_coefficients():
    df = pd.read_csv(TASK21_DATA)

    X = df[TASK21_FEATURES].copy()
    y = df["goal_difference_abs"]

    # Apply the same Yeo-Johnson treatment to rest days used in the final MLR.
    transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )
    X["rest_days_diff_abs"] = transformer.fit_transform(
        X[["rest_days_diff_abs"]]
    ).ravel()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    coef_df = pd.DataFrame(
        {
            "feature": TASK21_FEATURES,
            "standardised_coefficient": model.coef_,
        }
    )

    coef_df["label"] = coef_df["feature"].map(LABELS)
    coef_df = coef_df.sort_values(
        "standardised_coefficient",
        ascending=True,
    )

    values = coef_df["standardised_coefficient"]

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.barh(coef_df["label"], values)
    ax.axvline(0, linewidth=0.9)

    ax.set_xlabel("Standardised coefficient")
    ax.set_title(
        "Task 2.1 MLR: standardised coefficients",
        fontsize=12,
        pad=8,
    )

    # Give negative coefficient labels extra room so they do not crowd the y-axis labels.
    ax.set_xlim(min(-0.14, values.min() - 0.08), values.max() + 0.05)

    for i, value in enumerate(values):
        if value >= 0:
            ax.text(
                value + 0.01,
                i,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        else:
            ax.text(
                value - 0.01,
                i,
                f"{value:.3f}",
                va="center",
                ha="right",
                fontsize=9,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, OUT2)


def task22_actual_vs_predicted():
    df = pd.read_csv(TASK22_DATA)

    X = df[TASK22_FEATURES]
    y = df["goals_scored"]
    groups = df["match_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    line_min = min(float(y_test.min()), float(pred.min()))
    line_max = max(float(y_test.max()), float(pred.max()))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(y_test, pred, s=45, alpha=0.85)
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        linestyle="--",
        linewidth=1.2,
    )

    ax.set_xlabel("Actual goals scored")
    ax.set_ylabel("Predicted goals scored")
    ax.set_title(
        "Task 2.2 MLR: actual vs predicted",
        fontsize=12,
        pad=8,
    )

    ax.text(
        0.03,
        0.95,
        f"Grouped hold-out RMSE = {rmse:.3f}\nGrouped hold-out $R^2$ = {r2:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    plot_min = min(0.0, line_min) - 0.2
    plot_max = line_max + 0.2
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal", adjustable="box")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, OUT3)


def task22_standardised_coefficients():
    df = pd.read_csv(TASK22_COEF).copy()
    df["label"] = df["feature"].map(LABELS)
    df = df.sort_values(
        "standardised_coefficient",
        ascending=True,
    )

    values = df["standardised_coefficient"]

    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    ax.barh(df["label"], values)
    ax.axvline(0, linewidth=0.9)

    ax.set_xlabel("Standardised coefficient")
    ax.set_title(
        "Task 2.2 MLR: standardised coefficients",
        fontsize=12,
        pad=8,
    )
    ax.set_xlim(-0.34, 0.40)

    for i, value in enumerate(values):
        if value >= 0:
            ax.text(
                value + 0.008,
                i,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=9,
            )
        else:
            ax.text(
                value - 0.012,
                i,
                f"{value:.3f}",
                va="center",
                ha="right",
                fontsize=9,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save(fig, OUT4)


def main():
    task21_actual_vs_predicted()
    task21_standardised_coefficients()
    task22_actual_vs_predicted()
    task22_standardised_coefficients()

    print("Saved:")
    print(OUT1)
    print(OUT2)
    print(OUT3)
    print(OUT4)


if __name__ == "__main__":
    main()
