"""
Builds and audits a wider Task 2.1 feature pool for the absolute goal-margin target.
"""

import numpy as np
import pandas as pd


RAW = "matches_104_raw.csv"
BASE = "task2_1_match_level_dataset.csv"
MODEL_C = "task2_1_match_level_dataset_model_c.csv"
MODEL_D = "task2_1_match_level_dataset_model_d.csv"
MODEL_F = "task2_1_model_f_creative_dataset.csv"

POOL_OUT = "task2_1_feature_pool_abs.csv"
AUDIT_OUT = "task2_1_feature_pool_missingness.csv"
CORR_OUT = "task2_1_feature_pool_correlations.csv"


def load_selected(path, columns):
    df = pd.read_csv(path)
    available = [c for c in columns if c in df.columns]
    return df[["match_id", *available]].copy()


def main():
    raw = pd.read_csv(RAW)[
        [
            "match_id",
            "date",
            "stage",
            "team_a",
            "team_b",
            "team_a_goals",
            "team_b_goals",
        ]
    ].copy()

    raw["goal_difference_abs"] = (
        raw["team_a_goals"] - raw["team_b_goals"]
    ).abs()

    base = load_selected(
        BASE,
        [
            "fifa_rank_diff",
            "recent_ppg_diff",
            "recent_gf_avg_diff",
            "recent_ga_avg_diff",
            "recent_clean_sheet_rate_diff",
            "rest_days_diff",
            "host_advantage",
            "knockout",
        ],
    )

    model_c = load_selected(
        MODEL_C,
        [
            "recent_win_rate_diff",
            "prior_wc_matches_diff",
        ],
    )

    model_d = load_selected(
        MODEL_D,
        [
            "recent_gd_variability_diff",
        ],
    )

    model_f = load_selected(
        MODEL_F,
        [
            "recent_gd_trend_diff",
            "recent_scoring_consistency_diff",
            "recent_defensive_resilience_diff",
            "recent_close_game_rate_diff",
        ],
    )

    df = raw.merge(base, on="match_id", how="left")
    df = df.merge(model_c, on="match_id", how="left")
    df = df.merge(model_d, on="match_id", how="left")
    df = df.merge(model_f, on="match_id", how="left")

    signed_difference_features = [
        "fifa_rank_diff",
        "recent_ppg_diff",
        "recent_gf_avg_diff",
        "recent_ga_avg_diff",
        "recent_clean_sheet_rate_diff",
        "rest_days_diff",
        "recent_win_rate_diff",
        "prior_wc_matches_diff",
        "recent_gd_variability_diff",
        "recent_gd_trend_diff",
        "recent_scoring_consistency_diff",
        "recent_defensive_resilience_diff",
        "recent_close_game_rate_diff",
    ]

    # For an absolute goal-margin target, use the magnitude of between-team gaps.
    for col in signed_difference_features:
        if col in df.columns:
            df[f"{col}_abs"] = df[col].abs()

    final_feature_pool = [
        "fifa_rank_diff_abs",
        "prior_wc_matches_diff_abs",
        "rest_days_diff_abs",
        "recent_ppg_diff_abs",
        "recent_gf_avg_diff_abs",
        "recent_ga_avg_diff_abs",
        "recent_clean_sheet_rate_diff_abs",
        "recent_win_rate_diff_abs",
        "recent_gd_variability_diff_abs",
        "recent_gd_trend_diff_abs",
        "recent_scoring_consistency_diff_abs",
        "recent_defensive_resilience_diff_abs",
        "recent_close_game_rate_diff_abs",
        "host_advantage",
        "knockout",
    ]

    modelling = df[
        [
            "match_id",
            "date",
            "stage",
            "team_a",
            "team_b",
            "goal_difference_abs",
            *final_feature_pool,
        ]
    ].copy()

    audit_rows = []
    for col in final_feature_pool:
        missing = int(modelling[col].isna().sum())
        non_missing = int(modelling[col].notna().sum())
        audit_rows.append(
            {
                "feature": col,
                "rows_available": non_missing,
                "rows_missing": missing,
                "missing_percent": 100 * missing / len(modelling),
            }
        )

    audit = pd.DataFrame(audit_rows).sort_values(
        ["rows_missing", "feature"],
        ascending=[True, True],
    )

    corr_rows = []
    for col in final_feature_pool:
        valid = modelling[[col, "goal_difference_abs"]].dropna()
        corr = (
            valid[col].corr(valid["goal_difference_abs"])
            if len(valid) >= 3 and valid[col].nunique() > 1
            else np.nan
        )
        corr_rows.append(
            {
                "feature": col,
                "rows_used": len(valid),
                "target_correlation": corr,
                "abs_target_correlation": abs(corr) if pd.notna(corr) else np.nan,
            }
        )

    correlations = pd.DataFrame(corr_rows).sort_values(
        "abs_target_correlation",
        ascending=False,
        na_position="last",
    )

    modelling.to_csv(POOL_OUT, index=False)
    audit.to_csv(AUDIT_OUT, index=False)
    correlations.to_csv(CORR_OUT, index=False)

    print("\nTASK 2.1 - FEATURE POOL AUDIT")
    print(f"Rows in raw match dataset: {len(raw)}")
    print(f"Candidate explanatory variables: {len(final_feature_pool)}")

    print("\nMISSINGNESS")
    print(audit.round(2).to_string(index=False))

    print("\nTARGET CORRELATIONS")
    print(correlations.round(4).to_string(index=False))

    complete_all = modelling[final_feature_pool + ["goal_difference_abs"]].dropna()
    print(
        f"\nRows complete across all {len(final_feature_pool)} candidate features: "
        f"{len(complete_all)}"
    )

    print(f"\nSaved: {POOL_OUT}")
    print(f"Saved: {AUDIT_OUT}")
    print(f"Saved: {CORR_OUT}")


if __name__ == "__main__":
    main()
