"""
Compares Model E and Model F for Task 2.1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor


BASE_DIR = Path(__file__).resolve().parent

MODEL_E_RAW_FILE = BASE_DIR / "task2_1_match_level_dataset_model_d.csv"
MODEL_F_FILE = BASE_DIR / "task2_1_model_f_creative_dataset.csv"

TARGET = "goal_difference"

MODEL_E_RAW_FEATURES = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
    "prior_wc_matches_diff",
    "recent_gd_variability_diff",
]

MODEL_E_FULL_FEATURES = [
    "fifa_rank_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "rest_days_diff_yj",
    "host_advantage",
    "knockout",
    "prior_wc_matches_diff",
    "recent_gd_variability_diff",
]

MODEL_F_FEATURES = [
    "fifa_rank_diff",
    "prior_wc_matches_diff",
    "rest_days_diff_yj",
    "host_advantage",
    "recent_gd_trend_diff",
    "recent_scoring_consistency_diff",
    "recent_defensive_resilience_diff",
    "recent_close_game_rate_diff",
]


def model_e_pipeline():
    other = [x for x in MODEL_E_RAW_FEATURES if x != "rest_days_diff"]

    transformer = ColumnTransformer(
        transformers=[
            (
                "rest_yj",
                PowerTransformer(method="yeo-johnson", standardize=False),
                ["rest_days_diff"],
            ),
            ("other", "passthrough", other),
        ]
    )

    return Pipeline([
        ("transform", transformer),
        ("model", LinearRegression()),
    ])


def cv_scores(estimator, X, y, cv):
    scores = cross_validate(
        estimator,
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


def full_ols(df, features):
    X = sm.add_constant(df[features])
    y = df[TARGET]

    model = sm.OLS(y, X).fit()

    bp_p = het_breuschpagan(
        model.resid,
        model.model.exog,
    )[1]

    vif_rows = []

    for i, column in enumerate(X.columns):
        if column == "const":
            continue

        vif_rows.append({
            "feature": column,
            "vif": variance_inflation_factor(X.values, i),
        })

    return (
        model,
        bp_p,
        pd.DataFrame(vif_rows).sort_values("vif", ascending=False),
    )


def main():
    e_raw = pd.read_csv(MODEL_E_RAW_FILE)
    f = pd.read_csv(MODEL_F_FILE)

    # Create Model E full-data transformed column for diagnostics only.
    e_diag = e_raw.copy()

    yj = PowerTransformer(
        method="yeo-johnson",
        standardize=False,
    )

    e_diag["rest_days_diff_yj"] = yj.fit_transform(
        e_diag[["rest_days_diff"]]
    ).ravel()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # Leakage-safe Model E CV.
    e_mae, e_rmse, e_r2 = cv_scores(
        model_e_pipeline(),
        e_raw[MODEL_E_RAW_FEATURES],
        e_raw[TARGET],
        cv,
    )

    f_mae, f_rmse, f_r2 = cv_scores(
        LinearRegression(),
        f[MODEL_F_FEATURES],
        f[TARGET],
        cv,
    )

    e_ols, e_bp, e_vif = full_ols(
        e_diag,
        MODEL_E_FULL_FEATURES,
    )

    f_ols, f_bp, f_vif = full_ols(
        f,
        MODEL_F_FEATURES,
    )

    print("MODEL F TARGET CORRELATIONS")
    print("===========================")

    corr = (
        f[MODEL_F_FEATURES + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    print(corr.round(3).to_string())

    print("\nMODEL F STRONG PREDICTOR CORRELATIONS (|r| >= 0.70)")
    print("==================================================")

    feature_corr = f[MODEL_F_FEATURES].corr()
    found = False

    for i, first in enumerate(MODEL_F_FEATURES):
        for second in MODEL_F_FEATURES[i + 1:]:
            r = feature_corr.loc[first, second]

            if abs(r) >= 0.70:
                found = True
                print(f"{first} vs {second}: r={r:.3f}")

    if not found:
        print("None")

    comparison = pd.DataFrame([
        {
            "model": "Model E - benchmark",
            "r2": e_ols.rsquared,
            "adjusted_r2": e_ols.rsquared_adj,
            "max_vif": e_vif["vif"].max(),
            "breusch_pagan_p": e_bp,
            "cv_mae": e_mae.mean(),
            "cv_rmse": e_rmse.mean(),
            "cv_r2": e_r2.mean(),
            "cv_r2_sd": e_r2.std(ddof=1),
        },
        {
            "model": "Model F",
            "r2": f_ols.rsquared,
            "adjusted_r2": f_ols.rsquared_adj,
            "max_vif": f_vif["vif"].max(),
            "breusch_pagan_p": f_bp,
            "cv_mae": f_mae.mean(),
            "cv_rmse": f_rmse.mean(),
            "cv_r2": f_r2.mean(),
            "cv_r2_sd": f_r2.std(ddof=1),
        },
    ])

    print("\nMODEL E VS MODEL F")
    print("==================")
    print(comparison.round(4).to_string(index=False))

    print("\nMODEL F VIF")
    print("===========")
    print(f_vif.round(3).to_string(index=False))

    print("\nMODEL F P-VALUES")
    print("================")
    print(
        f_ols.pvalues
        .sort_values()
        .round(4)
        .to_string()
    )

    print("\n5-FOLD CV BY FOLD")
    print("=================")

    for i in range(5):
        print(
            f"Fold {i+1}: "
            f"E RMSE={e_rmse[i]:.4f}, E R2={e_r2[i]:.4f} | "
            f"F RMSE={f_rmse[i]:.4f}, F R2={f_r2[i]:.4f}"
        )

if __name__ == "__main__":
    main()
