"""
Runs the final validation for Task 2.2 Model V4.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

V2_FILE = BASE_DIR / "task2_2_candidate_dataset_v2.csv"
V4_FILE = BASE_DIR / "task2_2_candidate_dataset_v4.csv"
OUTPUT_FILE = BASE_DIR / "task2_2_final_dataset.csv"

TARGET = "goals_scored"
GROUP = "match_id"

RAW_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "weighted_attack_matchup",
    "scoring_reliability_index",
    "opponent_defensive_fragility",
    "opponent_concede_trend",
]

FINAL_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days_yj",
    "host_team",
    "weighted_attack_matchup",
    "scoring_reliability_index",
    "opponent_defensive_fragility",
    "opponent_concede_trend",
]


def make_pipeline():
    other_features = [
        f for f in RAW_FEATURES
        if f != "rest_days"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "rest_yj",
                PowerTransformer(
                    method="yeo-johnson",
                    standardize=False,
                ),
                ["rest_days"],
            ),
            (
                "other",
                "passthrough",
                other_features,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def regression_metrics(y_true, y_pred, full_target):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)

    target_range = full_target.max() - full_target.min()
    nrmse = rmse / target_range if target_range != 0 else float("nan")

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(title, values):
    print(f"\n{title}")
    print("-" * len(title))

    for key, value in values.items():
        print(f"{key}: {value:.4f}")


def main():
    v2 = pd.read_csv(V2_FILE)
    v4 = pd.read_csv(V4_FILE)

    # Recover raw rest_days for leakage-safe transformation.
    raw_rest = v2[
        ["match_id", "team", "opponent", "rest_days"]
    ].copy()

    model_df = v4.drop(
        columns=[
            c for c in ["rest_days_yj"]
            if c in v4.columns
        ]
    ).merge(
        raw_rest,
        on=["match_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if model_df["rest_days"].isna().any():
        raise ValueError("Could not recover raw rest_days for all rows")

    if len(v4) != 208:
        raise ValueError(f"Expected 208 rows; found {len(v4)}")

    if v4["match_id"].nunique() != 104:
        raise ValueError("Expected 104 unique matches")

    if not (v4.groupby("match_id").size() == 2).all():
        raise ValueError("Each match must contain exactly 2 team rows")

    if len(FINAL_FEATURES) != 8:
        raise ValueError("Task 2.2 must have exactly 8 explanatory variables")

    if v4[FINAL_FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Missing modelling values in final V4 dataset")

    print("TASK 2.2 - FINAL V4 VALIDATION")
    print("==============================")
    print("Rows:", len(v4))
    print("Unique matches:", v4["match_id"].nunique())
    print("Rows per match: 2")
    print("Explanatory variables:", len(FINAL_FEATURES))
    print(
        "Missing modelling values:",
        int(v4[FINAL_FEATURES + [TARGET]].isna().sum().sum()),
    )

    print("\nFINAL TASK 2.2 VARIABLES")
    print("------------------------")
    for i, feature in enumerate(FINAL_FEATURES, 1):
        print(f"{i}. {feature}")

    # Save final dataset.
    final_cols = [
        "match_id",
        "date",
        "stage",
        "team",
        "opponent",
        *FINAL_FEATURES,
        TARGET,
    ]

    final_df = v4[final_cols].copy()

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {OUTPUT_FILE.name}")

    X = model_df[RAW_FEATURES]
    y = model_df[TARGET]
    groups = model_df[GROUP]

    # ------------------------------------------------------------
    # Grouped 80/20 hold-out
    # ------------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise ValueError(f"Match leakage detected: {overlap}")

    model = make_pipeline()

    model.fit(
        X.iloc[train_idx],
        y.iloc[train_idx],
    )

    predictions = model.predict(
        X.iloc[test_idx]
    )

    holdout_scores = regression_metrics(
        y.iloc[test_idx],
        predictions,
        y,
    )

    baseline_value = y.iloc[train_idx].mean()

    baseline_predictions = np.full(
        len(test_idx),
        baseline_value,
        dtype=float,
    )

    baseline_scores = regression_metrics(
        y.iloc[test_idx],
        baseline_predictions,
        y,
    )

    print("\nGROUPED TRAIN / TEST SPLIT")
    print("--------------------------")
    print("Training rows:", len(train_idx))
    print("Test rows:", len(test_idx))
    print("Training matches:", len(train_groups))
    print("Test matches:", len(test_groups))
    print("Shared match IDs:", len(overlap))

    print_metrics(
        "FINAL V4 - GROUPED HOLD-OUT PERFORMANCE",
        holdout_scores,
    )

    print_metrics(
        "MEAN BASELINE - GROUPED HOLD-OUT",
        baseline_scores,
    )

    print("\nIMPROVEMENT OVER BASELINE")
    print("-------------------------")
    print(
        "MAE improvement:",
        f"{baseline_scores['MAE'] - holdout_scores['MAE']:.4f}",
    )
    print(
        "RMSE improvement:",
        f"{baseline_scores['RMSE'] - holdout_scores['RMSE']:.4f}",
    )
    print(
        "R^2 improvement:",
        f"{holdout_scores['R2'] - baseline_scores['R2']:.4f}",
    )

    # ------------------------------------------------------------
    # Grouped 5-fold CV
    # ------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)

    fold_rows = []

    print("\n5-FOLD GROUPED CROSS-VALIDATION")
    print("-------------------------------")

    for fold, (train_i, test_i) in enumerate(
        gkf.split(
            X,
            y,
            groups=groups,
        ),
        start=1,
    ):
        fold_train_groups = set(groups.iloc[train_i])
        fold_test_groups = set(groups.iloc[test_i])

        if fold_train_groups.intersection(fold_test_groups):
            raise ValueError(f"Match leakage in fold {fold}")

        fold_model = make_pipeline()

        fold_model.fit(
            X.iloc[train_i],
            y.iloc[train_i],
        )

        fold_pred = fold_model.predict(
            X.iloc[test_i]
        )

        fold_scores = regression_metrics(
            y.iloc[test_i],
            fold_pred,
            y,
        )

        fold_baseline_value = y.iloc[train_i].mean()

        fold_baseline_pred = np.full(
            len(test_i),
            fold_baseline_value,
            dtype=float,
        )

        fold_baseline_scores = regression_metrics(
            y.iloc[test_i],
            fold_baseline_pred,
            y,
        )

        fold_rows.append({
            "fold": fold,
            "MAE": fold_scores["MAE"],
            "RMSE": fold_scores["RMSE"],
            "R2": fold_scores["R2"],
            "baseline_MAE": fold_baseline_scores["MAE"],
            "baseline_RMSE": fold_baseline_scores["RMSE"],
            "baseline_R2": fold_baseline_scores["R2"],
        })

        print(
            f"Fold {fold}: "
            f"MAE={fold_scores['MAE']:.4f}, "
            f"RMSE={fold_scores['RMSE']:.4f}, "
            f"R^2={fold_scores['R2']:.4f} | "
            f"Baseline RMSE={fold_baseline_scores['RMSE']:.4f}"
        )

    folds = pd.DataFrame(fold_rows)

    print("\nGROUPED CV MEAN ± SD")
    print("--------------------")
    print(
        f"MAE:  {folds['MAE'].mean():.4f} "
        f"± {folds['MAE'].std(ddof=1):.4f}"
    )
    print(
        f"RMSE: {folds['RMSE'].mean():.4f} "
        f"± {folds['RMSE'].std(ddof=1):.4f}"
    )
    print(
        f"R^2:  {folds['R2'].mean():.4f} "
        f"± {folds['R2'].std(ddof=1):.4f}"
    )

    print("\nGROUPED BASELINE CV MEAN")
    print("------------------------")
    print(
        f"MAE:  {folds['baseline_MAE'].mean():.4f}"
    )
    print(
        f"RMSE: {folds['baseline_RMSE'].mean():.4f}"
    )
    print(
        f"R^2:  {folds['baseline_R2'].mean():.4f}"
    )

    # ------------------------------------------------------------
    # Full-data OLS diagnostics
    # ------------------------------------------------------------
    X_ols = sm.add_constant(
        v4[FINAL_FEATURES]
    )

    ols = sm.OLS(
        v4[TARGET],
        X_ols,
    ).fit()

    residuals = ols.resid

    shapiro_stat, shapiro_p = shapiro(
        residuals
    )

    bp_stat, bp_p, _, _ = het_breuschpagan(
        residuals,
        ols.model.exog,
    )

    print("\nFULL-DATA OLS FIT")
    print("-----------------")
    print(f"R^2: {ols.rsquared:.4f}")
    print(
        f"Adjusted R^2: "
        f"{ols.rsquared_adj:.4f}"
    )

    print("\nRESIDUAL DIAGNOSTICS")
    print("--------------------")
    print(
        f"Shapiro-Wilk statistic: "
        f"{shapiro_stat:.4f}"
    )
    print(
        f"Shapiro-Wilk p-value: "
        f"{shapiro_p:.4f}"
    )
    print(
        f"Breusch-Pagan statistic: "
        f"{bp_stat:.4f}"
    )
    print(
        f"Breusch-Pagan p-value: "
        f"{bp_p:.4f}"
    )

    print("\nP-VALUES")
    print("--------")
    print(
        ols.pvalues
        .drop("const")
        .sort_values()
        .round(4)
        .to_string()
    )

    # ------------------------------------------------------------
    # VIF
    # ------------------------------------------------------------
    vif_rows = []

    for i, column in enumerate(
        X_ols.columns
    ):
        if column == "const":
            continue

        vif_rows.append({
            "feature": column,
            "VIF": variance_inflation_factor(
                X_ols.values,
                i,
            ),
        })

    vif_df = (
        pd.DataFrame(vif_rows)
        .sort_values("VIF", ascending=False)
    )

    print("\nVIF")
    print("---")
    print(
        vif_df
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Standardised coefficients
    # ------------------------------------------------------------
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        v4[FINAL_FEATURES]
    )

    std_model = LinearRegression()

    std_model.fit(
        X_scaled,
        v4[TARGET],
    )

    coef_df = pd.DataFrame({
        "feature": FINAL_FEATURES,
        "standardised_coefficient": std_model.coef_,
    })

    coef_df["absolute_effect"] = (
        coef_df["standardised_coefficient"]
        .abs()
    )

    coef_df = coef_df.sort_values(
        "absolute_effect",
        ascending=False,
    )

    print("\nSTANDARDISED COEFFICIENTS")
    print("-------------------------")
    print(
        coef_df[
            [
                "feature",
                "standardised_coefficient",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()
