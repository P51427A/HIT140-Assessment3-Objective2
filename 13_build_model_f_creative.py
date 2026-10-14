"""
Builds the Model F feature set for Task 2.1.
"""

import json
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "task2_1_match_level_dataset_model_d.csv"
OUTPUT_FILE = BASE_DIR / "task2_1_model_f_creative_dataset.csv"
CACHE_DIR = BASE_DIR / "fifa_api_cache_model_f"

FIFA_API = "https://api.fifa.com/api/v3"
LOOKBACK = 5
REQUEST_DELAY_SECONDS = 0.5

HISTORY_START = "2025-01-01T00:00:00Z"
HISTORY_END = "2026-07-20T23:59:59Z"

ALIASES = {
    "USA": ["United States", "USA"],
    "South Korea": ["Korea Republic", "South Korea"],
    "Cape Verde": ["Cabo Verde", "Cape Verde"],
    "Czech Republic": ["Czechia", "Czech Republic"],
    "D.R. Congo": ["Congo DR", "DR Congo", "Congo, DR", "D.R. Congo"],
    "Ivory Coast": ["Côte d'Ivoire", "Cote d'Ivoire", "Ivory Coast"],
    "Bosnia & Herzegovina": ["Bosnia and Herzegovina", "Bosnia & Herzegovina"],
    "Curacao": ["Curaçao", "Curacao"],
    "Turkey": ["Türkiye", "Turkey"],
    "Iran": ["IR Iran", "Iran"],
}


def normalise_name(value):
    value = str(value).strip().lower()
    replacements = {
        "é": "e", "ë": "e", "ï": "i", "ö": "o", "ü": "u",
        "ç": "c", "ã": "a", "á": "a", "í": "i", "ó": "o",
        "ú": "u", "ô": "o", "è": "e", "ê": "e", "ñ": "n",
        "’": "'", "&": " and ",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def acceptable_keys(master_name):
    names = [master_name] + ALIASES.get(master_name, [])
    return {normalise_name(x) for x in names}


def description(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("Description"):
                return str(item["Description"])
    if isinstance(value, dict) and value.get("Description"):
        return str(value["Description"])
    return ""


def fifa_get(endpoint, params=None):
    params = dict(params or {})
    params.setdefault("language", "en")
    CACHE_DIR.mkdir(exist_ok=True)

    raw = endpoint + "?" + urlencode(sorted((k, str(v)) for k, v in params.items()))
    cache_file = CACHE_DIR / (hashlib.sha256(raw.encode("utf-8")).hexdigest() + ".json")

    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{FIFA_API}/{endpoint.lstrip('/')}"
    if params:
        url += "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    with urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8-sig"))

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    time.sleep(REQUEST_DELAY_SECONDS)
    return data


def get_team_id_table():
    schedules = fifa_get(
        "rankingschedules/all",
        {
            "type": 0,
            "gender": 1,
            "idClient": "64e9afa8-c5c0-413d-882b-bc9e6a81e264",
        },
    )

    rows = schedules.get("Results", [])
    if not rows:
        raise ValueError("No FIFA ranking schedules returned")

    rows = sorted(
        rows,
        key=lambda r: pd.to_datetime(r.get("OfficialDate"), errors="coerce", utc=True),
    )

    schedule_id = rows[-1]["IdRankingSchedule"]

    table = fifa_get(
        "rankingsbyschedule",
        {"rankingScheduleId": schedule_id},
    )

    data = []
    for item in table.get("Results", []):
        name = description(item.get("TeamName"))
        team_id = item.get("IdTeam")
        if name and team_id:
            data.append({
                "fifa_name": name,
                "key": normalise_name(name),
                "id_team": str(team_id),
            })

    return pd.DataFrame(data)


def match_team_ids(teams, fifa_table):
    mapping = {}

    for team in teams:
        keys = acceptable_keys(team)
        direct = fifa_table[fifa_table["key"].isin(keys)]

        if len(direct) == 1:
            mapping[team] = str(direct.iloc[0]["id_team"])
            continue

        possibles = []
        for _, row in fifa_table.iterrows():
            row_key = row["key"]
            if any(k in row_key or row_key in k for k in keys):
                possibles.append(row)

        if len(possibles) != 1:
            raise ValueError(
                f"Could not uniquely match '{team}' to FIFA team ID "
                f"(direct={len(direct)}, fallback={len(possibles)})"
            )

        mapping[team] = str(possibles[0]["id_team"])

    return mapping


def parse_team_match(item, target_team_id):
    date = pd.to_datetime(item.get("Date"), errors="coerce", utc=True)
    if pd.isna(date):
        return None
    date = date.tz_localize(None)

    home = item.get("Home") or {}
    away = item.get("Away") or {}

    home_id = str(home.get("IdTeam") or "")
    away_id = str(away.get("IdTeam") or "")

    try:
        home_score = int(item.get("HomeTeamScore"))
        away_score = int(item.get("AwayTeamScore"))
    except (TypeError, ValueError):
        return None

    if home_id == str(target_team_id):
        gf, ga = home_score, away_score
        opponent = description(away.get("TeamName"))
    elif away_id == str(target_team_id):
        gf, ga = away_score, home_score
        opponent = description(home.get("TeamName"))
    else:
        return None

    return {
        "date": date,
        "gf": gf,
        "ga": ga,
        "gd": gf - ga,
        "opponent": opponent,
    }


def get_recent_history(team_id):
    data = fifa_get(
        "calendar/matches",
        {
            "idTeam": team_id,
            "from": HISTORY_START,
            "to": HISTORY_END,
            "count": 200,
        },
    )

    rows = []
    for item in data.get("Results", []):
        parsed = parse_team_match(item, team_id)
        if parsed is not None:
            rows.append(parsed)

    history = pd.DataFrame(rows)

    if history.empty:
        raise ValueError(f"No usable FIFA history for team ID {team_id}")

    return (
        history.sort_values("date")
        .drop_duplicates(
            subset=["date", "opponent", "gf", "ga"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def engineered_recent_features(history, match_date, team, match_id):
    prior = history[history["date"] < match_date].tail(LOOKBACK).copy()

    if len(prior) != LOOKBACK:
        raise ValueError(
            f"Match {match_id}, {team}: only {len(prior)} prior matches; "
            f"need {LOOKBACK}."
        )

    # x = chronological order from oldest (0) to most recent (4)
    x = np.arange(LOOKBACK, dtype=float)
    gd = prior["gd"].to_numpy(dtype=float)

    # Simple linear slope of recent goal difference.
    gd_trend = float(np.polyfit(x, gd, 1)[0])

    scoring_consistency = float((prior["gf"] >= 1).mean())
    defensive_resilience = float((prior["ga"] <= 1).mean())
    close_game_rate = float((prior["gd"].abs() <= 1).mean())

    return {
        "gd_trend": gd_trend,
        "scoring_consistency": scoring_consistency,
        "defensive_resilience": defensive_resilience,
        "close_game_rate": close_game_rate,
    }


def main():
    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(
        df["date"],
        format="mixed",
        dayfirst=True,
    )

    teams = sorted(set(df["team_a"]) | set(df["team_b"]))

    if len(teams) != 48:
        raise ValueError(f"Expected 48 teams; found {len(teams)}")

    print("Matching teams to FIFA IDs...")
    fifa_table = get_team_id_table()
    team_ids = match_team_ids(teams, fifa_table)

    print("\nLoading FIFA histories...")
    histories = {}

    for i, team in enumerate(teams, 1):
        print(f"[{i:02d}/48] {team}")
        histories[team] = get_recent_history(team_ids[team])

    gd_trend_diff = []
    scoring_consistency_diff = []
    defensive_resilience_diff = []
    close_game_rate_diff = []

    for _, row in df.sort_values(["date", "match_id"]).iterrows():
        match_date = row["date"]
        match_id = int(row["match_id"])
        team_a = row["team_a"]
        team_b = row["team_b"]

        a = engineered_recent_features(
            histories[team_a], match_date, team_a, match_id
        )
        b = engineered_recent_features(
            histories[team_b], match_date, team_b, match_id
        )

        gd_trend_diff.append(round(a["gd_trend"] - b["gd_trend"], 4))
        scoring_consistency_diff.append(
            round(a["scoring_consistency"] - b["scoring_consistency"], 4)
        )
        defensive_resilience_diff.append(
            round(a["defensive_resilience"] - b["defensive_resilience"], 4)
        )
        close_game_rate_diff.append(
            round(a["close_game_rate"] - b["close_game_rate"], 4)
        )

    df["recent_gd_trend_diff"] = gd_trend_diff
    df["recent_scoring_consistency_diff"] = scoring_consistency_diff
    df["recent_defensive_resilience_diff"] = defensive_resilience_diff
    df["recent_close_game_rate_diff"] = close_game_rate_diff

    # Full-dataset transformed value is only for the exported modelling dataset.
    # The later CV comparison fits Yeo-Johnson inside each training fold.
    transformer = PowerTransformer(method="yeo-johnson", standardize=False)
    df["rest_days_diff_yj"] = transformer.fit_transform(
        df[["rest_days_diff"]]
    ).ravel()

    model_f_features = [
        "fifa_rank_diff",
        "prior_wc_matches_diff",
        "rest_days_diff_yj",
        "host_advantage",
        "recent_gd_trend_diff",
        "recent_scoring_consistency_diff",
        "recent_defensive_resilience_diff",
        "recent_close_game_rate_diff",
    ]

    output_cols = [
        "match_id",
        "date",
        "stage",
        "team_a",
        "team_b",
        *model_f_features,
        "goal_difference",
    ]

    output = df[output_cols].copy()

    if len(model_f_features) != 8:
        raise ValueError("Model F must contain exactly 8 explanatory variables")

    if output[model_f_features + ["goal_difference"]].isna().any().any():
        raise ValueError("Model F contains missing modelling values")

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nMODEL F DATASET VALIDATION")
    print("--------------------------")
    print("Rows:", len(output))
    print("Unique matches:", output["match_id"].nunique())
    print("Explanatory variables:", len(model_f_features))
    print(
        "Missing modelling values:",
        int(output[model_f_features + ["goal_difference"]].isna().sum().sum()),
    )

    print("\nMODEL F - EXACT 8 VARIABLES")
    print("---------------------------")
    for i, feature in enumerate(model_f_features, 1):
        print(f"{i}. {feature}")

    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
