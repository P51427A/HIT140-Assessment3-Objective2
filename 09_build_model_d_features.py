"""
Builds the Model D feature set for Task 2.1.
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


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "task2_1_match_level_dataset_model_c.csv"
OUTPUT_FILE = BASE_DIR / "task2_1_match_level_dataset_model_d.csv"
CACHE_DIR = BASE_DIR / "fifa_api_cache_model_d"

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
    """Use FIFA's ranking table only to obtain stable FIFA team IDs."""
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
        key=lambda r: pd.to_datetime(
            r.get("OfficialDate"), errors="coerce", utc=True
        ),
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
        "goal_difference": gf - ga,
        "opponent": opponent,
        "id_match": item.get("IdMatch"),
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

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No usable FIFA history returned for team ID {team_id}"
        )

    return (
        df.sort_values("date")
        .drop_duplicates(
            subset=["date", "opponent", "gf", "ga"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def recent_gd_variability(history, match_date, team, match_id):
    """
    Population SD (ddof=0) of goal difference across the latest 5 matches
    strictly before the target World Cup match.
    """
    prior = history[history["date"] < match_date].tail(LOOKBACK)

    if len(prior) != LOOKBACK:
        raise ValueError(
            f"Match {match_id}, {team}: only {len(prior)} prior matches; "
            f"need {LOOKBACK}."
        )

    return float(
        np.std(
            prior["goal_difference"].to_numpy(dtype=float),
            ddof=0,
        )
    )


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

    print("Matching the 48 teams to FIFA IDs...")
    fifa_table = get_team_id_table()
    team_ids = match_team_ids(teams, fifa_table)

    print("\nLoading FIFA histories...")
    histories = {}

    for i, team in enumerate(teams, 1):
        print(f"[{i:02d}/48] {team}")
        histories[team] = get_recent_history(team_ids[team])

    variability_diffs = []

    for _, row in df.sort_values(["date", "match_id"]).iterrows():
        match_date = row["date"]
        match_id = int(row["match_id"])
        team_a = row["team_a"]
        team_b = row["team_b"]

        a_var = recent_gd_variability(
            histories[team_a], match_date, team_a, match_id
        )
        b_var = recent_gd_variability(
            histories[team_b], match_date, team_b, match_id
        )

        variability_diffs.append(round(a_var - b_var, 4))

    # Model D replaces recent_win_rate_diff but keeps prior_wc_matches_diff.
    df["recent_gd_variability_diff"] = variability_diffs
    df = df.drop(columns=["recent_win_rate_diff"])

    features = [
        "fifa_rank_diff",
        "recent_gf_avg_diff",
        "recent_ga_avg_diff",
        "rest_days_diff",
        "host_advantage",
        "knockout",
        "prior_wc_matches_diff",
        "recent_gd_variability_diff",
    ]

    if len(features) != 8:
        raise ValueError("Model D must have exactly 8 explanatory variables")

    missing = df[features + ["goal_difference"]].isna().sum()

    if missing.any():
        raise ValueError(
            "Missing Model D values:\n"
            + missing[missing > 0].to_string()
        )

    print("\nMODEL D VALIDATION")
    print("------------------")
    print("Rows:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Explanatory variables:", len(features))
    print(
        "Missing modelling values:",
        int(df[features + ["goal_difference"]].isna().sum().sum()),
    )

    print("\nEXACT 8 MODEL D VARIABLES")
    print("-------------------------")
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature}")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nSaved: {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
