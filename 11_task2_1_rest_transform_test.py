"""
Tests a Yeo-Johnson transformation of the rest-days variable for Task 2.1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "task2_1_match_level_dataset_model_d.csv"

TARGET = "goal_difference"

FEATURES = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
    "prior_wc_matches_diff",
    "recent_gd_variability_diff",
]

REST_FEATURE = ["rest_days_diff"]
OTHER_FEATURES = [x for x in FEATURES if x != "rest_days_diff"]


def evaluate_raw_cv(X, y, cv):
    scores = cross_validate(
        LinearRegression(),
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
    )

    mae = -scores["test_mae"]
    rmse = np.sqrt(-scores["test_mse"])
    r2 = scores["test_r2"]

    return mae, rmse, r2


def make_transformed_pipeline():
    """
    Transform only rest_days_diff.
    All other predictors pass through unchanged.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "rest_yeojohnson",
                PowerTransformer(method="yeo-johnson", standardize=False),
                REST_FEATURE,
            ),
            (
                "other_features",
                "passthrough",
                OTHER_FEATURES,
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


def evaluate_transformed_cv(X, y, cv):
    pipeline = make_transformed_pipeline()

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
    )

    mae = -scores["test_mae"]
    rmse = np.sqrt(-scores["test_mse"])
    r2 = scores["test_r2"]

    return mae, rmse, r2


def full_sample_ols_raw(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    bp_p = het_breuschpagan(
        model.resid,
        model.model.exog,
    )[1]

    vif_values = []
    for i, column in enumerate(X_sm.columns):
        if column == "const":
            continue
        vif_values.append({
            "feature": column,
            "VIF": variance_inflation_factor(X_sm.values, i),
        })

    return model, bp_p, pd.DataFrame(vif_values)


def full_sample_ols_transformed(df):
    """
    Full-data diagnostic version of Model E.

    This fit is for model diagnostics only. The CV evaluation above uses
    a leakage-safe Pipeline and fits the transformer within each training fold.
    """
    transformed = df[FEATURES].copy()

    transformer = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )

    transformed["rest_days_diff_yj"] = transformer.fit_transform(
        transformed[["rest_days_diff"]]
    ).ravel()

    transformed = transformed.drop(columns=["rest_days_diff"])

    final_features = [
        "fifa_rank_diff",
        "recent_gf_avg_diff",
        "recent_ga_avg_diff",
        "host_advantage",
        "knockout",
        "prior_wc_matches_diff",
        "recent_gd_variability_diff",
        "rest_days_diff_yj",
    ]

    X_sm = sm.add_constant(transformed[final_features])
    y = df[TARGET]

    model = sm.OLS(y, X_sm).fit()

    bp_p = het_breuschpagan(
        model.resid,
        model.model.exog,
    )[1]

    vif_values = []
    for i, column in enumerate(X_sm.columns):
        if column == "const":
            continue
        vif_values.append({
            "feature": column,
            "VIF": variance_inflation_factor(X_sm.values, i),
        })

    return (
        model,
        bp_p,
        pd.DataFrame(vif_values),
        transformed["rest_days_diff_yj"],
        transformer.lambdas_[0],
    )


def print_cv(name, mae, rmse, r2):
    print(f"\n{name}")
    print("=" * len(name))

    for i in range(len(mae)):
        print(
            f"Fold {i+1}: "
            f"MAE={mae[i]:.4f}, "
            f"RMSE={rmse[i]:.4f}, "
            f"R^2={r2[i]:.4f}"
        )

    print("\nMean ± SD")
    print(f"MAE:  {mae.mean():.4f} ± {mae.std(ddof=1):.4f}")
    print(f"RMSE: {rmse.mean():.4f} ± {rmse.std(ddof=1):.4f}")
    print(f"R^2:  {r2.mean():.4f} ± {r2.std(ddof=1):.4f}")


def main():
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURES]
    y = df[TARGET]

    print("TASK 2.1 - REST-DAYS TRANSFORMATION EXPERIMENT")
    print("===============================================")
    print("Rows:", len(df))
    print("Predictors:", len(FEATURES))

    print("\nRAW REST-DAYS SUMMARY")
    print("---------------------")
    print(df["rest_days_diff"].describe().round(3).to_string())

    extreme = df.loc[
        df["rest_days_diff"].abs().sort_values(ascending=False).index[:5],
        ["match_id", "team_a", "team_b", "rest_days_diff", "goal_difference"],
    ]

    print("\nMOST EXTREME REST-DAYS DIFFERENCES")
    print("----------------------------------")
    print(extreme.to_string(index=False))

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ------------------------------------------------------------
    # Model D - original rest-days feature
    # ------------------------------------------------------------
    raw_mae, raw_rmse, raw_r2 = evaluate_raw_cv(
        X,
        y,
        cv,
    )

    raw_ols, raw_bp_p, raw_vif = full_sample_ols_raw(df)

    # ------------------------------------------------------------
    # Model E - Yeo-Johnson rest-days feature
    # ------------------------------------------------------------
    yj_mae, yj_rmse, yj_r2 = evaluate_transformed_cv(
        X,
        y,
        cv,
    )

    (
        yj_ols,
        yj_bp_p,
        yj_vif,
        transformed_rest,
        yj_lambda,
    ) = full_sample_ols_transformed(df)

    print("\nYEO-JOHNSON TRANSFORMATION")
    print("--------------------------")
    print(f"Lambda: {yj_lambda:.4f}")
    print("Original rest_days_diff range:",
          f"{df['rest_days_diff'].min():.3f} to {df['rest_days_diff'].max():.3f}")
    print("Transformed rest-days range:",
          f"{transformed_rest.min():.3f} to {transformed_rest.max():.3f}")

    print_cv(
        "MODEL D - ORIGINAL REST DAYS",
        raw_mae,
        raw_rmse,
        raw_r2,
    )

    print_cv(
        "MODEL E - YEO-JOHNSON REST DAYS",
        yj_mae,
        yj_rmse,
        yj_r2,
    )

    comparison = pd.DataFrame([
        {
            "model": "Model D - raw rest days",
            "r2": raw_ols.rsquared,
            "adjusted_r2": raw_ols.rsquared_adj,
            "max_vif": raw_vif["VIF"].max(),
            "breusch_pagan_p": raw_bp_p,
            "cv_mae": raw_mae.mean(),
            "cv_rmse": raw_rmse.mean(),
            "cv_r2": raw_r2.mean(),
            "cv_r2_sd": raw_r2.std(ddof=1),
        },
        {
            "model": "Model E - Yeo-Johnson rest days",
            "r2": yj_ols.rsquared,
            "adjusted_r2": yj_ols.rsquared_adj,
            "max_vif": yj_vif["VIF"].max(),
            "breusch_pagan_p": yj_bp_p,
            "cv_mae": yj_mae.mean(),
            "cv_rmse": yj_rmse.mean(),
            "cv_r2": yj_r2.mean(),
            "cv_r2_sd": yj_r2.std(ddof=1),
        },
    ])

    print("\nMODEL COMPARISON")
    print("================")
    print(comparison.round(4).to_string(index=False))

    print("\nMODEL E P-VALUES")
    print("================")
    print(
        yj_ols.pvalues
        .sort_values()
        .round(4)
        .to_string()
    )

    print("\nMODEL E VIF")
    print("===========")
    print(
        yj_vif
        .sort_values("VIF", ascending=False)
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------
    # Inspect the exact cross-validated predictions for extreme rows.
    # ------------------------------------------------------------
    raw_predictions = cross_val_predict(
        LinearRegression(),
        X,
        y,
        cv=cv,
    )

    yj_predictions = cross_val_predict(
        make_transformed_pipeline(),
        X,
        y,
        cv=cv,
    )

    prediction_check = df[
        ["match_id", "team_a", "team_b", "rest_days_diff", TARGET]
    ].copy()

    prediction_check["pred_model_d"] = raw_predictions
    prediction_check["pred_model_e"] = yj_predictions
    prediction_check["abs_error_d"] = (
        prediction_check[TARGET] - prediction_check["pred_model_d"]
    ).abs()
    prediction_check["abs_error_e"] = (
        prediction_check[TARGET] - prediction_check["pred_model_e"]
    ).abs()

    print("\nEXTREME REST-DAYS PREDICTION CHECK")
    print("==================================")
    check_rows = (
        prediction_check
        .assign(abs_rest=prediction_check["rest_days_diff"].abs())
        .sort_values("abs_rest", ascending=False)
        .head(5)
        .drop(columns=["abs_rest"])
    )
    print(check_rows.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
