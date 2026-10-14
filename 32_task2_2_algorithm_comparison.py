"""
Compares regression algorithms for the final Task 2.2 feature set.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor


INPUT = "task2_2_final_dataset.csv"
OUTPUT = "task2_2_algorithm_comparison.csv"
FOLDS_OUT = "task2_2_algorithm_cv_folds.csv"

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


def models():
    return {
        "Multiple Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=3,
            min_samples_leaf=5,
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            random_state=42,
        ),
    }


def main():
    df = pd.read_csv(INPUT)

    required = ["match_id", TARGET, *FEATURES]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isna().any().any():
        raise ValueError("Missing values found in modelling columns")

    if len(df) != 208:
        raise ValueError(f"Expected 208 rows, found {len(df)}")

    if df["match_id"].nunique() != 104:
        raise ValueError("Expected 104 unique matches")

    counts = df.groupby("match_id").size()
    if not (counts == 2).all():
        raise ValueError("Each match must have exactly 2 rows")

    X = df[FEATURES]
    y = df[TARGET]
    groups = df["match_id"]

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

    holdout_rows = []

    baseline_pred = np.repeat(y_train.mean(), len(y_test))
    row = {
        "model": "Mean Baseline",
        "evaluation": "holdout",
        **metrics(y_test, baseline_pred),
    }
    holdout_rows.append(row)

    for name, model in models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        holdout_rows.append(
            {
                "model": name,
                "evaluation": "holdout",
                **metrics(y_test, pred),
            }
        )

    gkf = GroupKFold(n_splits=5)
    fold_rows = []
    cv_summary = []

    all_models = {"Mean Baseline": None, **models()}

    for name, model in all_models.items():
        model_fold_rows = []

        for fold, (tr_idx, te_idx) in enumerate(
            gkf.split(X, y, groups),
            start=1,
        ):
            X_tr = X.iloc[tr_idx]
            X_te = X.iloc[te_idx]
            y_tr = y.iloc[tr_idx]
            y_te = y.iloc[te_idx]

            if name == "Mean Baseline":
                pred = np.repeat(y_tr.mean(), len(y_te))
            else:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)

            row = {
                "model": name,
                "fold": fold,
                **metrics(y_te, pred),
            }
            model_fold_rows.append(row)
            fold_rows.append(row)

        fold_df = pd.DataFrame(model_fold_rows)

        cv_summary.append(
            {
                "model": name,
                "evaluation": "5-fold grouped CV mean",
                "mae": fold_df["mae"].mean(),
                "mse": fold_df["mse"].mean(),
                "rmse": fold_df["rmse"].mean(),
                "r2": fold_df["r2"].mean(),
                "r2_sd": fold_df["r2"].std(ddof=0),
            }
        )

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_df["r2_sd"] = np.nan

    cv_df = pd.DataFrame(cv_summary)

    results = pd.concat(
        [
            holdout_df[
                ["model", "evaluation", "mae", "mse", "rmse", "r2", "r2_sd"]
            ],
            cv_df[
                ["model", "evaluation", "mae", "mse", "rmse", "r2", "r2_sd"]
            ],
        ],
        ignore_index=True,
    )

    results.to_csv(OUTPUT, index=False)
    pd.DataFrame(fold_rows).to_csv(FOLDS_OUT, index=False)

    print("\nTASK 2.2 - ALGORITHM COMPARISON")
    print(f"Rows: {len(df)}")
    print(f"Matches: {df['match_id'].nunique()}")
    print(f"Explanatory variables: {len(FEATURES)}")
    print("Grouped splitting by match_id: yes")

    print("\nHOLD-OUT PERFORMANCE")
    print(
        results[
            results["evaluation"] == "holdout"
        ][
            ["model", "mae", "rmse", "r2"]
        ].sort_values("rmse").round(4).to_string(index=False)
    )

    print("\n5-FOLD GROUPED CV")
    print(
        results[
            results["evaluation"] == "5-fold grouped CV mean"
        ][
            ["model", "mae", "rmse", "r2", "r2_sd"]
        ].sort_values("rmse").round(4).to_string(index=False)
    )

    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {FOLDS_OUT}")


if __name__ == "__main__":
    main()
