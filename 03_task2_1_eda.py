"""
Exploratory analysis for the Task 2.1 dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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


def iqr_outliers(series):
    """Return rows outside the standard 1.5*IQR bounds."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (series < lower) | (series > upper)
    return lower, upper, mask


def main():
    df = pd.read_csv(DATA_FILE)

    print("\nDATASET CHECK")
    print("-------------")
    print("Rows:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Missing modelling values:",
          int(df[FEATURES + [TARGET]].isna().sum().sum()))

    print("\nDESCRIPTIVE STATISTICS")
    print("----------------------")
    print(df[FEATURES + [TARGET]].describe().T.round(3))

    print("\nTARGET CORRELATIONS")
    print("-------------------")
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    target_corr = (
        corr[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda x: x.abs(), ascending=False)
    )
    print(target_corr.round(3))

    print("\nSTRONG PREDICTOR-TO-PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("--------------------------------------------------------")
    found = False
    feature_corr = df[FEATURES].corr(numeric_only=True)
    for i, first in enumerate(FEATURES):
        for second in FEATURES[i + 1:]:
            r = feature_corr.loc[first, second]
            if abs(r) >= 0.70:
                found = True
                print(f"{first} vs {second}: r = {r:.3f}")
    if not found:
        print("None")

    print("\nIQR OUTLIER CHECK")
    print("-----------------")
    for column in FEATURES + [TARGET]:
        lower, upper, mask = iqr_outliers(df[column])
        count = int(mask.sum())
        print(
            f"{column}: {count} outlier(s), "
            f"IQR bounds = [{lower:.3f}, {upper:.3f}]"
        )
        if count:
            print(
                df.loc[
                    mask,
                    ["match_id", "team_a", "team_b", column]
                ].to_string(index=False)
            )

    # 1. Target distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x=TARGET, bins=11, kde=True)
    plt.title("Distribution of Goal Difference")
    plt.xlabel("Goal Difference (Team A - Team B)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # 2. Boxplots for continuous predictors
    continuous = [
        "fifa_rank_diff",
        "recent_ppg_diff",
        "recent_gf_avg_diff",
        "recent_ga_avg_diff",
        "recent_clean_sheet_rate_diff",
        "rest_days_diff",
    ]

    for column in continuous:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot: {column}")
        plt.xlabel(column)
        plt.tight_layout()
        plt.show()

    # 3. Predictor-target scatterplots
    for column in continuous:
        plt.figure(figsize=(7, 5))
        sns.regplot(
            data=df,
            x=column,
            y=TARGET,
            ci=None,
            scatter_kws={"alpha": 0.7},
        )
        plt.title(f"{column} vs Goal Difference")
        plt.xlabel(column)
        plt.ylabel("Goal Difference")
        plt.tight_layout()
        plt.show()

    # 4. Binary predictors vs target
    for column in ["host_advantage", "knockout"]:
        plt.figure(figsize=(7, 5))
        sns.boxplot(data=df, x=column, y=TARGET)
        sns.swarmplot(data=df, x=column, y=TARGET, size=4)
        plt.title(f"{column} vs Goal Difference")
        plt.xlabel(column)
        plt.ylabel("Goal Difference")
        plt.tight_layout()
        plt.show()

    # 5. Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
    )
    plt.title("Task 2.1 Correlation Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
