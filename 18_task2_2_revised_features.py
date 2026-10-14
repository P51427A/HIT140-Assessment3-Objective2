"""
Builds the revised Task 2.2 feature set.
"""

from pathlib import Path

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_FILE = BASE_DIR / "task2_2_candidate_dataset.csv"
AUDIT_FILE = BASE_DIR / "task2_2_feature_audit.csv"
OUTPUT_FILE = BASE_DIR / "task2_2_candidate_dataset_v2.csv"

TARGET = "goals_scored"

FEATURES_V1 = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "attack_matchup_index",
    "scoring_momentum",
    "multi_goal_rate",
    "scoring_volatility",
]

FEATURES_V2 = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "goal_expectancy_matchup",
    "scoring_consistency_rate",
    "opponent_high_concede_rate",
    "scoring_streak",
]


def parse_scores(text):
    """
    Parse strings such as:
        2-1|1-0|0-0|3-2|1-1
    into [(2,1), (1,0), ...].
    """
    values = []

    for part in str(text).split("|"):
        part = part.strip()

        if not part:
            continue

        pieces = part.split("-")

        if len(pieces) != 2:
            raise ValueError(f"Could not parse score '{part}'")

        gf = int(pieces[0])
        ga = int(pieces[1])

        values.append((gf, ga))

    if len(values) != 5:
        raise ValueError(
            f"Expected exactly 5 historical scores, found {len(values)} in '{text}'"
        )

    return values


def scoring_streak(scores):
    """
    Count consecutive matches with at least one goal,
    starting from the most recent match.
    """
    streak = 0

    for gf, _ in reversed(scores):
        if gf >= 1:
            streak += 1
        else:
            break

    return streak


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


def strong_correlations(df, features, threshold=0.70):
    corr = df[features].corr()
    rows = []

    for i, first in enumerate(features):
        for second in features[i + 1:]:
            value = corr.loc[first, second]

            if abs(value) >= threshold:
                rows.append(
                    (first, second, value)
                )

    return rows


def main():
    candidate = pd.read_csv(CANDIDATE_FILE)
    audit = pd.read_csv(AUDIT_FILE)

    merge_cols = [
        "match_id",
        "team",
        "opponent",
        "team_history_scores",
        "opponent_history_scores",
        "team_recent_gf_avg",
        "opponent_recent_ga_avg",
    ]

    audit_small = audit[merge_cols].copy()

    df = candidate.merge(
        audit_small,
        on=["match_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if df[
        [
            "team_history_scores",
            "opponent_history_scores",
            "team_recent_gf_avg",
            "opponent_recent_ga_avg",
        ]
    ].isna().any().any():
        raise ValueError("Audit merge produced missing historical information")

    goal_expectancy = []
    scoring_consistency = []
    opponent_high_concede = []
    streaks = []

    for _, row in df.iterrows():
        team_scores = parse_scores(row["team_history_scores"])
        opponent_scores = parse_scores(row["opponent_history_scores"])

        # Correct direction: both stronger team attack and weaker opponent
        # defence should increase this variable.
        goal_expectancy.append(
            (
                float(row["team_recent_gf_avg"])
                + float(row["opponent_recent_ga_avg"])
            ) / 2
        )

        scoring_consistency.append(
            sum(gf >= 1 for gf, _ in team_scores) / 5
        )

        opponent_high_concede.append(
            sum(ga >= 2 for _, ga in opponent_scores) / 5
        )

        streaks.append(
            scoring_streak(team_scores)
        )

    df["goal_expectancy_matchup"] = goal_expectancy
    df["scoring_consistency_rate"] = scoring_consistency
    df["opponent_high_concede_rate"] = opponent_high_concede
    df["scoring_streak"] = streaks

    output_cols = [
        "match_id",
        "date",
        "stage",
        "team",
        "opponent",
        *FEATURES_V2,
        TARGET,
    ]

    output = df[output_cols].copy()

    if len(output) != 208:
        raise ValueError(
            f"Expected 208 rows; found {len(output)}"
        )

    if output["match_id"].nunique() != 104:
        raise ValueError(
            f"Expected 104 unique matches; found {output['match_id'].nunique()}"
        )

    if not (output.groupby("match_id").size() == 2).all():
        raise ValueError(
            "Every match must have exactly two team rows"
        )

    if len(FEATURES_V2) != 8:
        raise ValueError(
            "Task 2.2 must contain exactly 8 explanatory variables"
        )

    if output[FEATURES_V2 + [TARGET]].isna().any().any():
        raise ValueError(
            "Revised dataset contains missing modelling values"
        )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("TASK 2.2 - REVISED FEATURES")
    print("====================================")
    print("Rows:", len(output))
    print("Unique matches:", output["match_id"].nunique())
    print("Rows per match: 2")
    print("Explanatory variables:", len(FEATURES_V2))
    print(
        "Missing modelling values:",
        int(output[FEATURES_V2 + [TARGET]].isna().sum().sum()),
    )

    print("\nV1 TARGET CORRELATIONS")
    print("----------------------")
    print(
        target_correlations(candidate, FEATURES_V1)
        .round(3)
        .to_string()
    )

    print("\nV2 TARGET CORRELATIONS")
    print("----------------------")
    print(
        target_correlations(output, FEATURES_V2)
        .round(3)
        .to_string()
    )

    print("\nV2 STRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("---------------------------------------------")

    strong = strong_correlations(
        output,
        FEATURES_V2,
    )

    if not strong:
        print("None")
    else:
        for first, second, value in strong:
            print(
                f"{first} vs {second}: r={value:.3f}"
            )

    print("\nV2 VIF")
    print("------")
    print(
        vif_table(output, FEATURES_V2)
        .round(3)
        .to_string(index=False)
    )

    print("\nEXACT 8 V2 VARIABLES")
    print("--------------------")
    for i, feature in enumerate(FEATURES_V2, 1):
        print(f"{i}. {feature}")

    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
