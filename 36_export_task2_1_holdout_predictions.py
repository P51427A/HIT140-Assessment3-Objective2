import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer

DATA_FILE = "task2_1_final_abs_dataset.csv"
OUTPUT_FILE = "task2_1_holdout_predictions.csv"
TARGET = "goal_difference_abs"

FEATURES = [
    
    "fifa_rank_diff_abs",
    "recent_net_gd_gap_abs",
    "recent_scoring_consistency_diff_abs",
    "recent_defensive_resilience_diff_abs",
    "rest_days_diff_abs",
    "recent_gd_variability_diff_abs",
    "host_advantage",
    "knockout",
]

df = pd.read_csv(DATA_FILE)

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.20,
    random_state=42
)

train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()

preprocess = ColumnTransformer(
    transformers=[
        (
            "rest",
            PowerTransformer(method="yeo-johnson", standardize=True),
            ["rest_days_diff_abs"],
        )
    ],
    remainder="passthrough",
)

model = Pipeline([
    ("preprocess", preprocess),
    ("model", LinearRegression()),
])

model.fit(train_df[FEATURES], train_df[TARGET])
pred = model.predict(test_df[FEATURES])

id_columns = [
    c for c in ["match_id", "date", "stage", "team_a", "team_b"]
    if c in test_df.columns
]

out = test_df[id_columns].copy()
out["actual_goal_difference"] = test_df[TARGET].values
out["predicted_goal_difference"] = pred
out["residual"] = out["actual_goal_difference"] - out["predicted_goal_difference"]
out["absolute_error"] = out["residual"].abs()

out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(out)} rows to {OUTPUT_FILE}")
print(out.head(10).to_string(index=False))
