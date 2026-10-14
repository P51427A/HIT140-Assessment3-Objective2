"""
Final validation and export for Task 2.2.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


INPUT = "task2_2_final_dataset.csv"

FINAL_DATASET = "task2_2_final_model_dataset.csv"
SUMMARY_OUT = "task2_2_final_validation_results.csv"
FOLDS_OUT = "task2_2_final_cv_folds.csv"
COEFFICIENTS_OUT = "task2_2_final_coefficients.csv"
DIAGNOSTICS_OUT = "task2_2_final_diagnostics.csv"

TARGET = "goals_scored"

FEATURES = [
    "fifa_rank",
    "prior_wc_matches",
    "rest_days_yj",
    "host_team",
    "weighted_attack_matchup",
    "scoring_reliability_index",
    "opponent_defensive_fragility",
    "opponent_concede_trend",
]


def metrics(y_true, pred):
    mse = mean_squared_error(y_true, pred)
    return {
        "mae": mean_absolute_error(y_true, pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, pred),
    }


def main():
    df = pd.read_csv(INPUT)

    id_columns = [
        c for c in ["match_id", "date", "stage", "team", "opponent"]
        if c in df.columns
    ]

    required = ["match_id", TARGET, *FEATURES]
    missing_columns = [c for c in required if c not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    final_df = df[[*id_columns, *FEATURES, TARGET]].copy()

    if len(final_df) != 208:
        raise ValueError(f"Expected 208 rows, found {len(final_df)}")

    if final_df["match_id"].nunique() != 104:
        raise ValueError("Expected 104 unique matches")

    rows_per_match = final_df.groupby("match_id").size()
    if not (rows_per_match == 2).all():
        raise ValueError("Each match must have exactly 2 team rows")

    if len(FEATURES) != 8:
        raise ValueError("Task 2.2 must use exactly 8 explanatory variables")

    missing_values = int(final_df[FEATURES + [TARGET]].isna().sum().sum())
    if missing_values != 0:
        raise ValueError(
            f"Found {missing_values} missing values in modelling columns"
        )

    final_df.to_csv(FINAL_DATASET, index=False)

    X = final_df[FEATURES]
    y = final_df[TARGET]
    groups = final_df["match_id"]

    # Grouped hold-out split
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

    holdout_model = {
        "model": "Multiple Linear Regression",
        "evaluation": "grouped holdout",
        **metrics(y_test, pred),
        "r2_sd": np.nan,
    }

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    holdout_baseline = {
        "model": "Mean Baseline",
        "evaluation": "grouped holdout",
        **metrics(y_test, baseline_pred),
        "r2_sd": np.nan,
    }

    # Grouped 5-fold CV
    gkf = GroupKFold(n_splits=5)
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(
        gkf.split(X, y, groups),
        start=1,
    ):
        X_tr = X.iloc[tr_idx]
        X_te = X.iloc[te_idx]
        y_tr = y.iloc[tr_idx]
        y_te = y.iloc[te_idx]

        fold_model = LinearRegression()
        fold_model.fit(X_tr, y_tr)
        fold_pred = fold_model.predict(X_te)

        model_metrics = metrics(y_te, fold_pred)
        fold_rows.append(
            {
                "model": "Multiple Linear Regression",
                "fold": fold,
                **model_metrics,
            }
        )

        fold_baseline = np.repeat(y_tr.mean(), len(y_te))
        baseline_metrics = metrics(y_te, fold_baseline)
        fold_rows.append(
            {
                "model": "Mean Baseline",
                "fold": fold,
                **baseline_metrics,
            }
        )

    folds_df = pd.DataFrame(fold_rows)

    cv_rows = []
    for name in ["Multiple Linear Regression", "Mean Baseline"]:
        subset = folds_df[folds_df["model"] == name]
        cv_rows.append(
            {
                "model": name,
                "evaluation": "5-fold grouped CV mean",
                "mae": subset["mae"].mean(),
                "mse": subset["mse"].mean(),
                "rmse": subset["rmse"].mean(),
                "r2": subset["r2"].mean(),
                "r2_sd": subset["r2"].std(ddof=0),
            }
        )

    summary = pd.concat(
        [
            pd.DataFrame([holdout_model, holdout_baseline]),
            pd.DataFrame(cv_rows),
        ],
        ignore_index=True,
    )

    summary.to_csv(SUMMARY_OUT, index=False)
    folds_df.to_csv(FOLDS_OUT, index=False)

    # Full-data OLS diagnostics
    X_const = sm.add_constant(X)
    ols = sm.OLS(y, X_const).fit()
    residuals = ols.resid

    shapiro_stat, shapiro_p = shapiro(residuals)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)

    # VIF must include a constant column to avoid inflated values.
    vif_design = sm.add_constant(X)
    vif_rows = []
    for i, column in enumerate(vif_design.columns):
        if column == "const":
            continue
        vif_rows.append(
            {
                "feature": column,
                "vif": variance_inflation_factor(vif_design.values, i),
            }
        )

    diagnostics = pd.DataFrame(
        [
            {"diagnostic": "OLS R2", "value": ols.rsquared},
            {"diagnostic": "Adjusted R2", "value": ols.rsquared_adj},
            {"diagnostic": "Shapiro-Wilk statistic", "value": shapiro_stat},
            {"diagnostic": "Shapiro-Wilk p-value", "value": shapiro_p},
            {"diagnostic": "Breusch-Pagan statistic", "value": bp_stat},
            {"diagnostic": "Breusch-Pagan p-value", "value": bp_p},
            {
                "diagnostic": "Maximum VIF",
                "value": max(row["vif"] for row in vif_rows),
            },
        ]
    )
    diagnostics.to_csv(DIAGNOSTICS_OUT, index=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    standardised_model = LinearRegression()
    standardised_model.fit(X_scaled, y)

    vif_lookup = {row["feature"]: row["vif"] for row in vif_rows}

    coef_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "standardised_coefficient": standardised_model.coef_,
            "ols_coefficient": [ols.params[f] for f in FEATURES],
            "p_value": [ols.pvalues[f] for f in FEATURES],
            "vif": [vif_lookup[f] for f in FEATURES],
        }
    ).sort_values(
        "standardised_coefficient",
        key=lambda s: s.abs(),
        ascending=False,
    )

    coef_df.to_csv(COEFFICIENTS_OUT, index=False)

    print("\nTASK 2.2 - FINAL VALIDATION")
    print(f"Rows: {len(final_df)}")
    print(f"Matches: {final_df['match_id'].nunique()}")
    print(f"Explanatory variables: {len(FEATURES)}")
    print(f"Missing modelling values: {missing_values}")
    print(f"Target: {TARGET}")

    print("\nFINAL EXPLANATORY VARIABLES")
    for i, feature in enumerate(FEATURES, start=1):
        print(f"{i}. {feature}")

    print("\nFINAL PERFORMANCE SUMMARY")
    print(summary.round(4).to_string(index=False))

    print("\nOLS DIAGNOSTICS")
    print(diagnostics.round(4).to_string(index=False))

    print("\nCOEFFICIENTS")
    print(coef_df.round(4).to_string(index=False))

    print(f"\nSaved: {FINAL_DATASET}")
    print(f"Saved: {SUMMARY_OUT}")
    print(f"Saved: {FOLDS_OUT}")
    print(f"Saved: {COEFFICIENTS_OUT}")
    print(f"Saved: {DIAGNOSTICS_OUT}")


if __name__ == "__main__":
    main()
