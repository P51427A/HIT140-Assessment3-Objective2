"""
Builds the Task 2.1 dataset using FIFA match and ranking data.

Outputs:
- task2_1_match_level_dataset.csv
- task2_1_feature_audit.csv
"""

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "matches_104_raw.csv"
OUTPUT_FILE = BASE_DIR / "task2_1_match_level_dataset.csv"
AUDIT_FILE = BASE_DIR / "task2_1_feature_audit.csv"
CACHE_DIR = BASE_DIR / "fifa_api_cache"

FIFA_API = "https://api.fifa.com/api/v3"
TOURNAMENT_START = pd.Timestamp("2026-06-11", tz="UTC")
HISTORY_START = "2025-01-01T00:00:00Z"
HISTORY_END = "2026-07-20T23:59:59Z"

LOOKBACK = 5
HOSTS = {"Canada", "Mexico", "USA"}
REQUEST_DELAY_SECONDS = 0.7

FEATURES = [
    "fifa_rank_diff",
    "recent_ppg_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "recent_clean_sheet_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]

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
    """Normalise team names for matching between our CSV and FIFA."""
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
    """All acceptable normalised names for one master-file team."""
    names = [master_name] + ALIASES.get(master_name, [])
    return {normalise_name(x) for x in names}


def description(value):
    """Extract Description from FIFA's localised-list fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict) and item.get("Description"):
                return str(item["Description"])
    if isinstance(value, dict):
        if value.get("Description"):
            return str(value["Description"])
    return ""


def iso_date(value):
    """Convert FIFA datetime/date value to pandas Timestamp."""
    return pd.to_datetime(value, errors="coerce", utc=True).tz_localize(None)


def cache_name(endpoint, params):
    raw = endpoint + "?" + urlencode(sorted((k, str(v)) for k, v in params.items()))
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    if len(safe) > 180:
        import hashlib
        safe = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return safe + ".json"


def fifa_get(endpoint, params=None):
    """
    GET JSON from FIFA API and cache it locally.

    This is the same public api.fifa.com domain used by FIFA's own site/app.
    """
    params = dict(params or {})
    params.setdefault("language", "en")

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file = CACHE_DIR / cache_name(endpoint, params)

    if cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{FIFA_API}/{endpoint.lstrip('/')}"
    if params:
        url += "?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            ),
            "Accept": "application/json,text/plain,*/*",
        },
    )

    with urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode("utf-8-sig"))

    with cache_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    time.sleep(REQUEST_DELAY_SECONDS)
    return data


def load_master():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"{MASTER_FILE.name} not found. Put this script beside the master CSV."
        )

    df = pd.read_csv(MASTER_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="raise")

    required = {
        "match_id", "date", "stage", "team_a", "team_b",
        "team_a_goals", "team_b_goals",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Master file missing columns: {sorted(missing)}")

    if len(df) != 104:
        raise ValueError(f"Expected 104 matches; found {len(df)}")

    teams = sorted(set(df["team_a"]) | set(df["team_b"]))
    if len(teams) != 48:
        raise ValueError(f"Expected 48 teams; found {len(teams)}")

    return df, teams


def get_pre_tournament_ranking_schedule():
    """
    Find the latest published men's FIFA ranking date before 11 June 2026.
    """
    data = fifa_get(
        "rankingschedules/all",
        {
            "type": 0,
            "gender": 1,
            # FIFA examples include this client ID. The endpoint is public.
            "idClient": "64e9afa8-c5c0-413d-882b-bc9e6a81e264",
        },
    )

    rows = data.get("Results", [])
    if not rows:
        raise ValueError("FIFA returned no ranking schedules")

    candidates = []
    for row in rows:
        official = pd.to_datetime(row.get("OfficialDate"), errors="coerce")
        schedule_id = row.get("IdRankingSchedule")
        if pd.notna(official) and schedule_id and official < TOURNAMENT_START:
            candidates.append((official, schedule_id))

    if not candidates:
        raise ValueError("Could not find a FIFA ranking schedule before 11 June 2026")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def load_ranking_table():
    """
    Load the full FIFA ranking table from the latest pre-tournament ranking date.

    Returns:
        ranking_date, DataFrame with fifa_name, rank, fifa_points, id_team, id_country
    """
    ranking_date, schedule_id = get_pre_tournament_ranking_schedule()

    data = fifa_get(
        "rankingsbyschedule",
        {"rankingScheduleId": schedule_id},
    )

    rows = []
    for item in data.get("Results", []):
        fifa_name = description(item.get("TeamName"))
        rank = item.get("Rank")
        team_id = item.get("IdTeam")
        country = item.get("IdCountry")
        points = (
            item.get("DecimalTotalPoints")
            if item.get("DecimalTotalPoints") is not None
            else item.get("TotalPoints")
        )

        if fifa_name and rank is not None and team_id:
            rows.append({
                "fifa_name": fifa_name,
                "key": normalise_name(fifa_name),
                "rank": int(rank),
                "fifa_points": points,
                "id_team": str(team_id),
                "id_country": country,
            })

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("FIFA ranking table returned no usable rows")

    return ranking_date, table


def match_master_teams_to_fifa(master_teams, ranking_table):
    """Match all 48 World Cup teams to FIFA ranking rows."""
    mapping = {}

    for team in master_teams:
        keys = acceptable_keys(team)
        matches = ranking_table[ranking_table["key"].isin(keys)]

        if len(matches) != 1:
            # containment fallback for minor naming differences
            target_keys = acceptable_keys(team)
            possibles = []
            for _, row in ranking_table.iterrows():
                row_key = row["key"]
                if any(k in row_key or row_key in k for k in target_keys):
                    possibles.append(row)

            if len(possibles) == 1:
                matched = possibles[0]
            else:
                raise ValueError(
                    f"Could not uniquely match '{team}' to FIFA ranking. "
                    f"Direct matches={len(matches)}, fallback={len(possibles)}"
                )
        else:
            matched = matches.iloc[0]

        mapping[team] = {
            "fifa_name": matched["fifa_name"],
            "rank": int(matched["rank"]),
            "fifa_points": matched["fifa_points"],
            "id_team": str(matched["id_team"]),
            "id_country": matched["id_country"],
        }

    return mapping


def parse_fifa_match(item, target_team_id):
    """
    Convert a FIFA calendar match into one team's perspective.

    Returns None for matches with unusable scores/dates.
    """
    match_date = iso_date(item.get("Date"))
    if pd.isna(match_date):
        return None

    home = item.get("Home") or {}
    away = item.get("Away") or {}

    home_id = str(home.get("IdTeam") or "")
    away_id = str(away.get("IdTeam") or "")

    home_name = description(home.get("TeamName"))
    away_name = description(away.get("TeamName"))

    home_score = item.get("HomeTeamScore")
    away_score = item.get("AwayTeamScore")

    try:
        home_score = int(home_score)
        away_score = int(away_score)
    except (TypeError, ValueError):
        return None

    target_team_id = str(target_team_id)

    if home_id == target_team_id:
        gf, ga = home_score, away_score
        opponent = away_name
    elif away_id == target_team_id:
        gf, ga = away_score, home_score
        opponent = home_name
    else:
        return None

    result = "W" if gf > ga else "D" if gf == ga else "L"

    return {
        "date": match_date,
        "gf": gf,
        "ga": ga,
        "result": result,
        "opponent": opponent,
        "competition": description(item.get("CompetitionName")),
        "id_match": item.get("IdMatch"),
    }


def load_team_history(team, team_info):
    """Load completed FIFA matches for one team from 2025 onward."""
    data = fifa_get(
        "calendar/matches",
        {
            "idTeam": team_info["id_team"],
            "from": HISTORY_START,
            "to": HISTORY_END,
            "count": 200,
        },
    )

    rows = []
    for item in data.get("Results", []):
        parsed = parse_fifa_match(item, team_info["id_team"])
        if parsed is not None:
            rows.append(parsed)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"{team}: FIFA returned no usable historical matches")

    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["date", "opponent", "gf", "ga"], keep="last")
        .reset_index(drop=True)
    )

    return df


def prematch_features(history, match_date, team, match_id):
    """Calculate last-5 statistics using strictly earlier matches."""
    prior = history[history["date"] < match_date].tail(LOOKBACK).copy()

    if len(prior) != LOOKBACK:
        available = len(history[history["date"] < match_date])
        raise ValueError(
            f"Match {match_id}, {team}: only {available} FIFA matches exist "
            f"before {match_date.date()}; need at least {LOOKBACK}."
        )

    points = prior["result"].map({"W": 3, "D": 1, "L": 0})

    return {
        "ppg": float(points.mean()),
        "gf_avg": float(prior["gf"].mean()),
        "ga_avg": float(prior["ga"].mean()),
        "clean_sheet_rate": float((prior["ga"] == 0).mean()),
        "rest_days": int((match_date - prior["date"].max()).days),
        "history_first_date": prior["date"].min().date().isoformat(),
        "history_last_date": prior["date"].max().date().isoformat(),
        "matches_used": int(len(prior)),
        "opponents_used": " | ".join(prior["opponent"].astype(str).tolist()),
    }


def validate_output(df):
    if len(df) != 104:
        raise ValueError(f"Expected 104 rows; found {len(df)}")

    if df["match_id"].nunique() != 104:
        raise ValueError("match_id is not unique")

    if len(FEATURES) != 8:
        raise ValueError(f"Expected exactly 8 explanatory variables; found {len(FEATURES)}")

    missing = df[FEATURES + ["goal_difference"]].isna().sum()
    if missing.any():
        raise ValueError(
            "Missing modelling values:\n"
            + missing[missing > 0].to_string()
        )

    if not set(df["host_advantage"].unique()).issubset({-1, 0, 1}):
        raise ValueError("host_advantage contains unexpected values")

    if not set(df["knockout"].unique()).issubset({0, 1}):
        raise ValueError("knockout contains unexpected values")

    print("\nVALIDATION PASSED")
    print("-----------------")
    print("Rows:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Explanatory variables:", len(FEATURES))
    print("Missing modelling values:", int(df[FEATURES + ["goal_difference"]].isna().sum().sum()))


def main():
    master, teams = load_master()

    print("Finding latest FIFA ranking published before the World Cup...")
    ranking_date, ranking_table = load_ranking_table()
    print("Pre-tournament FIFA ranking date:", ranking_date.date())

    print("\nMatching the 48 World Cup teams to FIFA IDs/ranks...")
    team_info = match_master_teams_to_fifa(teams, ranking_table)
    print("Matched teams:", len(team_info))

    print("\nDownloading/caching FIFA match histories...")
    histories = {}
    for i, team in enumerate(teams, 1):
        print(
            f"[{i:02d}/48] {team}: "
            f"rank={team_info[team]['rank']}, "
            f"idTeam={team_info[team]['id_team']}"
        )
        histories[team] = load_team_history(team, team_info[team])

    rows = []
    audit = []

    for _, match in master.sort_values(["date", "match_id"]).iterrows():
        match_id = int(match["match_id"])
        match_date = match["date"]
        a_team = match["team_a"]
        b_team = match["team_b"]

        a = prematch_features(histories[a_team], match_date, a_team, match_id)
        b = prematch_features(histories[b_team], match_date, b_team, match_id)

        # Positive value means Team A has the stronger (lower-number) FIFA rank.
        rank_diff = team_info[b_team]["rank"] - team_info[a_team]["rank"]

        rows.append({
            "match_id": match_id,
            "date": match_date.date().isoformat(),
            "stage": match["stage"],
            "team_a": a_team,
            "team_b": b_team,

            # EXACTLY EIGHT EXPLANATORY VARIABLES:
            "fifa_rank_diff": rank_diff,
            "recent_ppg_diff": a["ppg"] - b["ppg"],
            "recent_gf_avg_diff": a["gf_avg"] - b["gf_avg"],
            "recent_ga_avg_diff": a["ga_avg"] - b["ga_avg"],
            "recent_clean_sheet_rate_diff":
                a["clean_sheet_rate"] - b["clean_sheet_rate"],
            "rest_days_diff": a["rest_days"] - b["rest_days"],
            "host_advantage":
                int(a_team in HOSTS) - int(b_team in HOSTS),
            "knockout": 0 if match["stage"] == "Group Stage" else 1,

            # RESPONSE VARIABLE:
            "goal_difference":
                int(match["team_a_goals"]) - int(match["team_b_goals"]),
        })

        audit.append({
            "match_id": match_id,
            "world_cup_date": match_date.date().isoformat(),
            "ranking_snapshot_date": ranking_date.date().isoformat(),

            "team_a": a_team,
            "team_a_fifa_name": team_info[a_team]["fifa_name"],
            "team_a_fifa_rank": team_info[a_team]["rank"],
            "team_a_fifa_id": team_info[a_team]["id_team"],
            "team_a_history_first": a["history_first_date"],
            "team_a_history_last": a["history_last_date"],
            "team_a_matches_used": a["matches_used"],
            "team_a_opponents_used": a["opponents_used"],

            "team_b": b_team,
            "team_b_fifa_name": team_info[b_team]["fifa_name"],
            "team_b_fifa_rank": team_info[b_team]["rank"],
            "team_b_fifa_id": team_info[b_team]["id_team"],
            "team_b_history_first": b["history_first_date"],
            "team_b_history_last": b["history_last_date"],
            "team_b_matches_used": b["matches_used"],
            "team_b_opponents_used": b["opponents_used"],
        })

    dataset = pd.DataFrame(rows)

    round_cols = [
        "recent_ppg_diff",
        "recent_gf_avg_diff",
        "recent_ga_avg_diff",
        "recent_clean_sheet_rate_diff",
    ]
    dataset[round_cols] = dataset[round_cols].round(4)

    validate_output(dataset)

    dataset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit).to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {OUTPUT_FILE.name}")
    print(f"Saved: {AUDIT_FILE.name}")

    print("\nEXACT 8 EXPLANATORY VARIABLES")
    print("-----------------------------")
    for i, feature in enumerate(FEATURES, 1):
        print(f"{i}. {feature}")

if __name__ == "__main__":
    main()
