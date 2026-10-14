"""
Builds the V4 feature set for Task 2.2.
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
OUTPUT_FILE = BASE_DIR / "task2_2_candidate_dataset_v4.csv"

TARGET = "goals_scored"

V4_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days_yj",
    "host_team",
    "weighted_attack_matchup",
    "scoring_reliability_index",
    "opponent_defensive_fragility",
    "opponent_concede_trend",
]


def parse_scores(text):
    values = []

    for part in str(text).split("|"):
        part = part.strip()

        if not part:
            continue

        pieces = part.split("-")

        if len(pieces) != 2:
            raise ValueError(f"Could not parse score '{part}'")

        values.append((int(pieces[0]), int(pieces[1])))

    if len(values) != 5:
        raise ValueError(
            f"Expected exactly 5 historical scores; found {len(values)}"
        )

    return values


def weighted_average(values):
    """
    Oldest -> newest weights = 1,2,3,4,5.
    More recent matches therefore contribute more.
    """
    weights = np.arange(1, len(values) + 1, dtype=float)

    return float(
        np.average(
            np.asarray(values, dtype=float),
            weights=weights,
        )
    )


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
                print(f"{first} vs {second}: r={value:.3f}")

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
            "team_history_scores",
            "opponent_history_scores",
        ]
    ].copy()

    df = v2.merge(
        audit_small,
        on=["match_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if df[
        ["team_history_scores", "opponent_history_scores"]
    ].isna().any().any():
        raise ValueError("Missing historical score data after merge")

    weighted_matchup = []
    scoring_reliability = []
    defensive_fragility = []
    concede_trend = []

    for _, row in df.iterrows():
        team_scores = parse_scores(
            row["team_history_scores"]
        )
        opponent_scores = parse_scores(
            row["opponent_history_scores"]
        )

        team_gf = np.array(
            [gf for gf, _ in team_scores],
            dtype=float,
        )

        opponent_ga = np.array(
            [ga for _, ga in opponent_scores],
            dtype=float,
        )

        weighted_team_gf = weighted_average(team_gf)
        weighted_opp_ga = weighted_average(opponent_ga)

        weighted_matchup.append(
            (weighted_team_gf + weighted_opp_ga) / 2
        )

        scoring_reliability.append(
            float(
                team_gf.mean()
                / (1.0 + team_gf.std(ddof=0))
            )
        )

        defensive_fragility.append(
            float(
                opponent_ga.mean()
                / (1.0 + opponent_ga.std(ddof=0))
            )
        )

        # Oldest -> newest; positive slope means the opponent
        # has recently been conceding more goals.
        x = np.arange(5, dtype=float)

        concede_trend.append(
            float(
                np.polyfit(
                    x,
                    opponent_ga,
                    1,
                )[0]
            )
        )

    df["weighted_attack_matchup"] = weighted_matchup
    df["scoring_reliability_index"] = scoring_reliability
    df["opponent_defensive_fragility"] = defensive_fragility
    df["opponent_concede_trend"] = concede_trend

    transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )

    df["rest_days_yj"] = transformer.fit_transform(
        df[["rest_days"]]
    ).ravel()

    output_cols = [
        "match_id",
        "date",
        "stage",
        "team",
        "opponent",
        *V4_FEATURES,
        TARGET,
    ]

    output = df[output_cols].copy()

    if len(output) != 208:
        raise ValueError(
            f"Expected 208 rows; found {len(output)}"
        )

    if output["match_id"].nunique() != 104:
        raise ValueError(
            "Expected 104 unique match IDs"
        )

    if not (output.groupby("match_id").size() == 2).all():
        raise ValueError(
            "Each match must produce exactly 2 rows"
        )

    if len(V4_FEATURES) != 8:
        raise ValueError(
            "V4 must contain exactly 8 explanatory variables"
        )

    if output[V4_FEATURES + [TARGET]].isna().any().any():
        raise ValueError(
            "V4 contains missing modelling values"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("TASK 2.2 - V4 FEATURE ENGINEERING")
    print("===========================================")
    print("Rows:", len(output))
    print("Unique matches:", output["match_id"].nunique())
    print("Rows per match: 2")
    print("Explanatory variables:", len(V4_FEATURES))
    print(
        "Missing modelling values:",
        int(
            output[V4_FEATURES + [TARGET]]
            .isna()
            .sum()
            .sum()
        ),
    )

    print("\nV4 TARGET CORRELATIONS")
    print("----------------------")
    print(
        target_correlations(
            output,
            V4_FEATURES,
        )
        .round(3)
        .to_string()
    )

    print("\nV4 STRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("---------------------------------------------")
    print_strong_correlations(
        output,
        V4_FEATURES,
    )

    print("\nV4 VIF")
    print("------")
    print(
        vif_table(
            output,
            V4_FEATURES,
        )
        .round(3)
        .to_string(index=False)
    )

    print("\nEXACT 8 V4 VARIABLES")
    print("--------------------")
    for i, feature in enumerate(V4_FEATURES, 1):
        print(f"{i}. {feature}")

    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
