"""
Builds the V3 feature set for Task 2.2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

V2_FILE = BASE_DIR / "task2_2_candidate_dataset_v2.csv"
AUDIT_FILE = BASE_DIR / "task2_2_feature_audit.csv"
OUTPUT_FILE = BASE_DIR / "task2_2_candidate_dataset_v3.csv"

TARGET = "goals_scored"

V2_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "goal_expectancy_matchup",
    "scoring_consistency_rate",
    "opponent_high_concede_rate",
    "scoring_streak",
]

V3_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days_yj",
    "host_team",
    "goal_expectancy_matchup",
    "scoring_consistency_rate",
    "opponent_high_concede_rate",
    "opponent_concede_volatility",
]


def parse_scores(text):
    scores = []

    for part in str(text).split("|"):
        part = part.strip()

        if not part:
            continue

        pieces = part.split("-")

        if len(pieces) != 2:
            raise ValueError(f"Could not parse score: {part}")

        scores.append((int(pieces[0]), int(pieces[1])))

    if len(scores) != 5:
        raise ValueError(
            f"Expected 5 historical scores; found {len(scores)}"
        )

    return scores


def vif_table(df, features):
    X = sm.add_constant(df[features])

    rows = []

    for i, column in enumerate(X.columns):
        if column == "const":
            continue

        rows.append({
            "feature": column,
            "VIF": variance_inflation_factor(X.values, i),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("VIF", ascending=False)
    )


def target_correlations(df, features):
    return (
        df[features + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )
    )


def print_strong_correlations(df, features, threshold=0.70):
    corr = df[features].corr()

    found = False

    for i, first in enumerate(features):
        for second in features[i + 1:]:
            value = corr.loc[first, second]

            if abs(value) >= threshold:
                found = True
                print(
                    f"{first} vs {second}: r={value:.3f}"
                )

    if not found:
        print("None")


def main():
    v2 = pd.read_csv(V2_FILE)
    audit = pd.read_csv(AUDIT_FILE)

    audit_small = audit[
        [
            "match_id",
            "team",
            "opponent",
            "opponent_history_scores",
        ]
    ].copy()

    df = v2.merge(
        audit_small,
        on=["match_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if df["opponent_history_scores"].isna().any():
        raise ValueError(
            "Missing opponent history after audit merge"
        )

    opponent_volatility = []

    for text in df["opponent_history_scores"]:
        scores = parse_scores(text)

        # In the opponent's own historical score strings:
        # first value = opponent goals scored
        # second value = opponent goals conceded
        conceded = np.array(
            [ga for _, ga in scores],
            dtype=float,
        )

        opponent_volatility.append(
            float(np.std(conceded, ddof=0))
        )

    df["opponent_concede_volatility"] = opponent_volatility

    transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )

    df["rest_days_yj"] = transformer.fit_transform(
        df[["rest_days"]]
    ).ravel()

    output_columns = [
        "match_id",
        "date",
        "stage",
        "team",
        "opponent",
        *V3_FEATURES,
        TARGET,
    ]

    output = df[output_columns].copy()

    if len(output) != 208:
        raise ValueError(
            f"Expected 208 rows; found {len(output)}"
        )

    if output["match_id"].nunique() != 104:
        raise ValueError(
            "Expected 104 unique matches"
        )

    if not (output.groupby("match_id").size() == 2).all():
        raise ValueError(
            "Each match must have exactly 2 rows"
        )

    if len(V3_FEATURES) != 8:
        raise ValueError(
            "V3 must contain exactly 8 explanatory variables"
        )

    if output[V3_FEATURES + [TARGET]].isna().any().any():
        raise ValueError(
            "V3 contains missing modelling values"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("TASK 2.2 - V3 FEATURE REFINEMENT")
    print("================================")
    print("Rows:", len(output))
    print("Unique matches:", output["match_id"].nunique())
    print("Rows per match: 2")
    print("Explanatory variables:", len(V3_FEATURES))
    print(
        "Missing modelling values:",
        int(output[V3_FEATURES + [TARGET]].isna().sum().sum()),
    )

    print("\nREST-DAYS TRANSFORMATION")
    print("------------------------")
    print(
        f"Yeo-Johnson lambda: {transformer.lambdas_[0]:.4f}"
    )
    print(
        f"Raw range: {v2['rest_days'].min():.3f} "
        f"to {v2['rest_days'].max():.3f}"
    )
    print(
        f"Transformed range: {output['rest_days_yj'].min():.3f} "
        f"to {output['rest_days_yj'].max():.3f}"
    )

    print("\nV2 TARGET CORRELATIONS")
    print("----------------------")
    print(
        target_correlations(v2, V2_FEATURES)
        .round(3)
        .to_string()
    )

    print("\nV3 TARGET CORRELATIONS")
    print("----------------------")
    print(
        target_correlations(output, V3_FEATURES)
        .round(3)
        .to_string()
    )

    print("\nV3 STRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("---------------------------------------------")
    print_strong_correlations(
        output,
        V3_FEATURES,
    )

    print("\nV3 VIF")
    print("------")
    print(
        vif_table(output, V3_FEATURES)
        .round(3)
        .to_string(index=False)
    )

    print("\nEXACT 8 V3 VARIABLES")
    print("--------------------")
    for i, feature in enumerate(V3_FEATURES, 1):
        print(f"{i}. {feature}")

    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
