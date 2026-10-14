import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

DATA_FILE = "task2_2_final_model_dataset.csv"
OUTPUT_FILE = "task2_2_holdout_predictions.csv"
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

df = pd.read_csv(DATA_FILE)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        df[FEATURES],
        df[TARGET],
        groups=df["match_id"]
    )
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

model = LinearRegression()
model.fit(train_df[FEATURES], train_df[TARGET])

pred = model.predict(test_df[FEATURES])

id_columns = [
    c for c in ["match_id", "date", "stage", "team", "opponent"]
    if c in test_df.columns
]

out = test_df[id_columns].copy()
out["actual_goals"] = test_df[TARGET].values
out["predicted_goals"] = pred
out["residual"] = out["actual_goals"] - out["predicted_goals"]
out["absolute_error"] = out["residual"].abs()

sort_cols = ["match_id"]
if "team" in out.columns:
    sort_cols.append("team")
out = out.sort_values(sort_cols)

out.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(out)} rows to {OUTPUT_FILE}")
print(out.head(12).to_string(index=False))
