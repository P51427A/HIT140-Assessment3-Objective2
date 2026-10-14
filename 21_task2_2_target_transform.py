"""
Compares raw and transformed target models for Task 2.2.
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
    other_features = [f for f in RAW_FEATURES if f != "rest_days"]

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


def metrics(y_true, y_pred, full_target):
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


def run_holdout(X, y, groups, transformed=False):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    model = make_pipeline()

    if transformed:
        model.fit(X_train, np.log1p(y_train))
        pred_log = model.predict(X_test)
        pred = np.expm1(pred_log)
        pred = np.clip(pred, 0, None)
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    return (
        metrics(y_test, pred, y),
        train_idx,
        test_idx,
    )


def run_grouped_cv(X, y, groups, transformed=False):
    gkf = GroupKFold(n_splits=5)

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(X, y, groups=groups),
        start=1,
    ):
        model = make_pipeline()

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if transformed:
            model.fit(X_train, np.log1p(y_train))
            pred_log = model.predict(X_test)
            pred = np.expm1(pred_log)
            pred = np.clip(pred, 0, None)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

        score = metrics(y_test, pred, y)

        rows.append({
            "fold": fold,
            "MAE": score["MAE"],
            "RMSE": score["RMSE"],
            "R2": score["R2"],
        })

    return pd.DataFrame(rows)


def full_ols_diagnostics(df, transformed=False):
    X = sm.add_constant(df[DIAG_FEATURES])

    if transformed:
        y = np.log1p(df[TARGET])
    else:
        y = df[TARGET]

    model = sm.OLS(y, X).fit()

    shapiro_stat, shapiro_p = shapiro(model.resid)
    bp_stat, bp_p, _, _ = het_breuschpagan(
        model.resid,
        model.model.exog,
    )

    return model, shapiro_p, bp_p


def print_metric_block(title, values):
    print(f"\n{title}")
    print("-" * len(title))
    for key, value in values.items():
        print(f"{key}: {value:.4f}")


def main():
    v2 = pd.read_csv(V2_FILE)
    v3 = pd.read_csv(V3_FILE)

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
        raise ValueError("Missing raw rest_days after merge")

    X = df[RAW_FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    print("TASK 2.2 - TARGET TRANSFORMATION EXPERIMENT")
    print("===========================================")
    print("Rows:", len(df))
    print("Unique matches:", groups.nunique())

    raw_holdout, _, _ = run_holdout(
        X, y, groups, transformed=False
    )

    log_holdout, _, _ = run_holdout(
        X, y, groups, transformed=True
    )

    print_metric_block(
        "MODEL A - RAW TARGET HOLD-OUT",
        raw_holdout,
    )

    print_metric_block(
        "MODEL B - LOG1P TARGET HOLD-OUT",
        log_holdout,
    )

    raw_cv = run_grouped_cv(
        X, y, groups, transformed=False
    )

    log_cv = run_grouped_cv(
        X, y, groups, transformed=True
    )

    print("\n5-FOLD GROUPED CV - RAW TARGET")
    print("------------------------------")
    print(raw_cv.round(4).to_string(index=False))

    print("\n5-FOLD GROUPED CV - LOG1P TARGET")
    print("-------------------------------")
    print(log_cv.round(4).to_string(index=False))

    comparison = pd.DataFrame([
        {
            "model": "Raw target",
            "holdout_MAE": raw_holdout["MAE"],
            "holdout_RMSE": raw_holdout["RMSE"],
            "holdout_R2": raw_holdout["R2"],
            "cv_MAE": raw_cv["MAE"].mean(),
            "cv_RMSE": raw_cv["RMSE"].mean(),
            "cv_R2": raw_cv["R2"].mean(),
            "cv_R2_SD": raw_cv["R2"].std(ddof=1),
        },
        {
            "model": "log1p target",
            "holdout_MAE": log_holdout["MAE"],
            "holdout_RMSE": log_holdout["RMSE"],
            "holdout_R2": log_holdout["R2"],
            "cv_MAE": log_cv["MAE"].mean(),
            "cv_RMSE": log_cv["RMSE"].mean(),
            "cv_R2": log_cv["R2"].mean(),
            "cv_R2_SD": log_cv["R2"].std(ddof=1),
        },
    ])

    print("\nMODEL COMPARISON")
    print("================")
    print(
        comparison
        .round(4)
        .to_string(index=False)
    )

    raw_ols, raw_shapiro, raw_bp = full_ols_diagnostics(
        df,
        transformed=False,
    )

    log_ols, log_shapiro, log_bp = full_ols_diagnostics(
        df,
        transformed=True,
    )

    print("\nOLS DIAGNOSTIC COMPARISON")
    print("=========================")
    print(
        f"Raw target:    R^2={raw_ols.rsquared:.4f}, "
        f"Adj R^2={raw_ols.rsquared_adj:.4f}, "
        f"Shapiro p={raw_shapiro:.4f}, "
        f"Breusch-Pagan p={raw_bp:.4f}"
    )
    print(
        f"log1p target:  R^2={log_ols.rsquared:.4f}, "
        f"Adj R^2={log_ols.rsquared_adj:.4f}, "
        f"Shapiro p={log_shapiro:.4f}, "
        f"Breusch-Pagan p={log_bp:.4f}"
    )

if __name__ == "__main__":
    main()
