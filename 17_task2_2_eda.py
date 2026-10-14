"""
Performs exploratory analysis for the Task 2.2 dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "task2_2_candidate_dataset.csv"

TARGET = "goals_scored"

FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "attack_matchup_index",
    "scoring_momentum",
    "multi_goal_rate",
    "scoring_volatility",
]

CONTINUOUS_FOR_OUTLIERS = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "attack_matchup_index",
    "scoring_momentum",
    "multi_goal_rate",
    "scoring_volatility",
    TARGET,
]


def iqr_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (series < lower) | (series > upper)

    return lower, upper, mask


def main():
    df = pd.read_csv(DATA_FILE)

    print("TASK 2.2 - EDA AND DIAGNOSTICS")
    print("==============================")
    print("Rows:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Unique teams:", df["team"].nunique())
    print("Explanatory variables:", len(FEATURES))
    print(
        "Missing modelling values:",
        int(df[FEATURES + [TARGET]].isna().sum().sum()),
    )

    rows_per_match = df.groupby("match_id").size()

    print("\nPAIRED MATCH STRUCTURE")
    print("----------------------")
    print("Minimum rows per match:", int(rows_per_match.min()))
    print("Maximum rows per match:", int(rows_per_match.max()))
    print(
        "All matches have exactly 2 rows:",
        bool((rows_per_match == 2).all()),
    )

    print("\nDESCRIPTIVE STATISTICS")
    print("----------------------")
    print(
        df[FEATURES + [TARGET]]
        .describe()
        .round(3)
        .to_string()
    )

    print("\nTARGET DISTRIBUTION")
    print("-------------------")
    print(df[TARGET].value_counts().sort_index().to_string())

    print("\nTARGET CORRELATIONS")
    print("-------------------")

    correlations = (
        df[FEATURES + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
    )

    print(correlations.round(3).to_string())

    print("\nSTRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("------------------------------------------")

    feature_corr = df[FEATURES].corr()
    found = False

    for i, first in enumerate(FEATURES):
        for second in FEATURES[i + 1:]:
            r = feature_corr.loc[first, second]

            if abs(r) >= 0.70:
                found = True
                print(
                    f"{first} vs {second}: r={r:.3f}"
                )

    if not found:
        print("None")

    # ------------------------------------------------------------
    # VIF
    # ------------------------------------------------------------
    X = sm.add_constant(df[FEATURES])

    vif_rows = []

    for i, column in enumerate(X.columns):
        if column == "const":
            continue

        vif_rows.append({
            "feature": column,
            "VIF": variance_inflation_factor(
                X.values,
                i,
            ),
        })

    vif_df = (
        pd.DataFrame(vif_rows)
        .sort_values("VIF", ascending=False)
    )

    print("\nVIF")
    print("---")
    print(vif_df.round(3).to_string(index=False))

    # ------------------------------------------------------------
    # IQR outliers
    # ------------------------------------------------------------
    print("\nIQR OUTLIER CHECK")
    print("-----------------")

    for column in CONTINUOUS_FOR_OUTLIERS:
        lower, upper, mask = iqr_outliers(df[column])

        print(
            f"{column}: "
            f"{int(mask.sum())} outlier(s); "
            f"bounds [{lower:.3f}, {upper:.3f}]"
        )

        if mask.sum() > 0:
            cols = [
                "match_id",
                "team",
                "opponent",
                column,
            ]

            if TARGET not in cols:
                cols.append(TARGET)

            print(
                df.loc[mask, cols]
                .sort_values(column)
                .to_string(index=False)
            )

    # ------------------------------------------------------------
    # Minimal plots
    # ------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.hist(
        df[TARGET],
        bins=np.arange(
            df[TARGET].min() - 0.5,
            df[TARGET].max() + 1.5,
            1,
        ),
        edgecolor="black",
    )
    plt.xlabel("Goals scored")
    plt.ylabel("Frequency")
    plt.title("Task 2.2 Target Distribution")
    plt.tight_layout()
    plt.show()

    corr_matrix = df[FEATURES + [TARGET]].corr()

    plt.figure(figsize=(9, 7))
    image = plt.imshow(
        corr_matrix,
        aspect="auto",
    )
    plt.colorbar(image)

    plt.xticks(
        range(len(corr_matrix.columns)),
        corr_matrix.columns,
        rotation=90,
    )
    plt.yticks(
        range(len(corr_matrix.index)),
        corr_matrix.index,
    )

    plt.title("Task 2.2 Correlation Heatmap")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
