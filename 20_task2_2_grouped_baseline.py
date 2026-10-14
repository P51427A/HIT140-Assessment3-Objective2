"""
Fits the grouped baseline regression for Task 2.2.
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
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

V2_FILE = BASE_DIR / "task2_2_candidate_dataset_v2.csv"
V3_FILE = BASE_DIR / "task2_2_candidate_dataset_v3.csv"

TARGET = "goals_scored"
GROUP = "match_id"

RAW_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days",
    "host_team",
    "goal_expectancy_matchup",
    "scoring_consistency_rate",
    "opponent_high_concede_rate",
    "opponent_concede_volatility",
]

DIAG_FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days_yj",
    "host_team",
    "goal_expectancy_matchup",
    "scoring_consistency_rate",
    "opponent_high_concede_rate",
    "opponent_concede_volatility",
]


def make_pipeline():
    other_features = [
        feature for feature in RAW_FEATURES
        if feature != "rest_days"
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

    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2": r2,
    }


def print_metrics(title, values):
    print(f"\n{title}")
    print("-" * len(title))

    for key, value in values.items():
        print(f"{key}: {value:.4f}")


def main():
    v2 = pd.read_csv(V2_FILE)
    v3 = pd.read_csv(V3_FILE)

    # Bring raw rest_days into the V3 dataset so Yeo-Johnson can be fitted
    # separately within each training split/fold.
    raw_rest = v2[
        ["match_id", "team", "opponent", "rest_days"]
    ].copy()

    df = v3.merge(
        raw_rest,
        on=["match_id", "team", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if df["rest_days"].isna().any():
        raise ValueError("Raw rest_days could not be recovered for all rows")

    if len(df) != 208:
        raise ValueError(f"Expected 208 rows; found {len(df)}")

    if not (df.groupby(GROUP).size() == 2).all():
        raise ValueError("Each match must contain exactly two team rows")

    X = df[RAW_FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    print("TASK 2.2 - GROUPED BASELINE REGRESSION")
    print("======================================")
    print("Rows:", len(df))
    print("Unique matches:", groups.nunique())
    print("Rows per match: 2")
    print("Predictors:", len(RAW_FEATURES))

    # ------------------------------------------------------------
    # 80/20 grouped hold-out split
    # ------------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_index, test_index = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    train_groups = set(groups.iloc[train_index])
    test_groups = set(groups.iloc[test_index])

    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise ValueError(
            f"Match leakage detected between train and test: {overlap}"
        )

    model = make_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    model_scores = regression_metrics(
        y_test,
        y_pred,
        y,
    )

    baseline_value = y_train.mean()
    baseline_pred = np.full(
        len(y_test),
        baseline_value,
        dtype=float,
    )

    baseline_scores = regression_metrics(
        y_test,
        baseline_pred,
        y,
    )

    print("\nGROUPED TRAIN / TEST SPLIT")
    print("--------------------------")
    print("Training rows:", len(train_index))
    print("Test rows:", len(test_index))
    print("Training matches:", len(train_groups))
    print("Test matches:", len(test_groups))
    print("Shared match IDs:", len(overlap))

    print_metrics(
        "V3 MODEL - GROUPED HOLD-OUT PERFORMANCE",
        model_scores,
    )

    print_metrics(
        "MEAN BASELINE - GROUPED HOLD-OUT",
        baseline_scores,
    )

    print("\nIMPROVEMENT OVER BASELINE")
    print("-------------------------")
    print(
        "MAE improvement:",
        f"{baseline_scores['MAE'] - model_scores['MAE']:.4f}",
    )
    print(
        "RMSE improvement:",
        f"{baseline_scores['RMSE'] - model_scores['RMSE']:.4f}",
    )
    print(
        "R^2 improvement:",
        f"{model_scores['R2'] - baseline_scores['R2']:.4f}",
    )

    # ------------------------------------------------------------
    # 5-fold grouped cross-validation
    # ------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)

    fold_rows = []

    print("\n5-FOLD GROUPED CROSS-VALIDATION")
    print("-------------------------------")

    for fold_number, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1,
    ):
        fold_model = make_pipeline()

        X_fold_train = X.iloc[train_idx]
        X_fold_test = X.iloc[test_idx]
        y_fold_train = y.iloc[train_idx]
        y_fold_test = y.iloc[test_idx]

        fold_train_groups = set(groups.iloc[train_idx])
        fold_test_groups = set(groups.iloc[test_idx])

        if fold_train_groups.intersection(fold_test_groups):
            raise ValueError(
                f"Match leakage detected in fold {fold_number}"
            )

        fold_model.fit(
            X_fold_train,
            y_fold_train,
        )

        fold_pred = fold_model.predict(
            X_fold_test
        )

        fold_metrics = regression_metrics(
            y_fold_test,
            fold_pred,
            y,
        )

        # Fold-specific training-mean baseline
        fold_baseline_pred = np.full(
            len(y_fold_test),
            y_fold_train.mean(),
            dtype=float,
        )

        fold_baseline = regression_metrics(
            y_fold_test,
            fold_baseline_pred,
            y,
        )

        fold_rows.append({
            "fold": fold_number,
            "MAE": fold_metrics["MAE"],
            "RMSE": fold_metrics["RMSE"],
            "R2": fold_metrics["R2"],
            "baseline_MAE": fold_baseline["MAE"],
            "baseline_RMSE": fold_baseline["RMSE"],
            "baseline_R2": fold_baseline["R2"],
            "test_matches": len(fold_test_groups),
        })

        print(
            f"Fold {fold_number}: "
            f"MAE={fold_metrics['MAE']:.4f}, "
            f"RMSE={fold_metrics['RMSE']:.4f}, "
            f"R^2={fold_metrics['R2']:.4f} | "
            f"Baseline RMSE={fold_baseline['RMSE']:.4f}"
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
    X_diag = sm.add_constant(
        df[DIAG_FEATURES]
    )

    ols = sm.OLS(
        y,
        X_diag,
    ).fit()

    residuals = ols.resid

    shapiro_stat, shapiro_p = shapiro(
        residuals
    )

    bp_stat, bp_p, _, _ = het_breuschpagan(
        residuals,
        ols.model.exog,
    )

    print("\nFULL-DATA OLS DIAGNOSTICS")
    print("-------------------------")
    print(f"R^2: {ols.rsquared:.4f}")
    print(
        f"Adjusted R^2: "
        f"{ols.rsquared_adj:.4f}"
    )
    print(
        f"Shapiro-Wilk p-value: "
        f"{shapiro_p:.4f}"
    )
    print(
        f"Breusch-Pagan p-value: "
        f"{bp_p:.4f}"
    )

    print("\nCOEFFICIENTS AND P-VALUES")
    print("-------------------------")

    result_table = pd.DataFrame({
        "coefficient": ols.params,
        "p_value": ols.pvalues,
    })

    print(
        result_table
        .round(4)
        .to_string()
    )

    # ------------------------------------------------------------
    # VIF
    # ------------------------------------------------------------
    vif_rows = []

    for i, column in enumerate(
        X_diag.columns
    ):
        if column == "const":
            continue

        vif_rows.append({
            "feature": column,
            "VIF": variance_inflation_factor(
                X_diag.values,
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

if __name__ == "__main__":
    main()
