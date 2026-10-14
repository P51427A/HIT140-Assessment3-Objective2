"""
Builds the Task 2.2 team-level modelling dataset.
"""

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MATCH_FILE = BASE_DIR / "matches_104_raw.csv"
OUTPUT_FILE = BASE_DIR / "task2_2_candidate_dataset.csv"
AUDIT_FILE = BASE_DIR / "task2_2_feature_audit.csv"
CACHE_DIR = BASE_DIR / "fifa_api_cache_task2_2"

FIFA_API = "https://api.fifa.com/api/v3"

TOURNAMENT_START = pd.Timestamp("2026-06-11")
RECENT_HISTORY_START = "2025-01-01T00:00:00Z"
RECENT_HISTORY_END = "2026-07-20T23:59:59Z"
WC_HISTORY_START = "1930-01-01T00:00:00Z"
WC_HISTORY_END = "2022-12-31T23:59:59Z"

LOOKBACK = 5
REQUEST_DELAY_SECONDS = 0.4

HOSTS = {"USA", "Canada", "Mexico"}

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

    raw_key = endpoint + "?" + urlencode(
        sorted((k, str(v)) for k, v in params.items())
    )
    cache_name = hashlib.sha256(raw_key.encode("utf-8")).hexdigest() + ".json"
    cache_file = CACHE_DIR / cache_name

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


def get_pre_tournament_rankings():
    """
    Select the latest FIFA men's ranking schedule published before
    the World Cup begins, then return team name, FIFA ID and rank.
    """
    schedules = fifa_get(
        "rankingschedules/all",
        {
            "type": 0,
            "gender": 1,
            "idClient": "64e9afa8-c5c0-413d-882b-bc9e6a81e264",
        },
    )

    rows = []
    for item in schedules.get("Results", []):
        date = pd.to_datetime(
            item.get("OfficialDate"),
            errors="coerce",
            utc=True,
        )
        if pd.isna(date):
            continue

        date = date.tz_localize(None)

        if date < TOURNAMENT_START:
            rows.append((date, item))

    if not rows:
        raise ValueError("No pre-tournament FIFA ranking schedule found")

    snapshot_date, selected = max(rows, key=lambda x: x[0])
    schedule_id = selected["IdRankingSchedule"]

    table = fifa_get(
        "rankingsbyschedule",
        {"rankingScheduleId": schedule_id},
    )

    ranking_rows = []

    for item in table.get("Results", []):
        team_name = description(item.get("TeamName"))
        team_id = item.get("IdTeam")
        rank = item.get("Rank")

        if team_name and team_id and rank is not None:
            ranking_rows.append({
                "fifa_name": team_name,
                "key": normalise_name(team_name),
                "id_team": str(team_id),
                "fifa_rank": int(rank),
            })

    ranking_df = pd.DataFrame(ranking_rows)

    print(
        "FIFA ranking snapshot:",
        snapshot_date.date(),
        f"(schedule {schedule_id})",
    )

    return ranking_df, snapshot_date.date().isoformat()


def match_teams_to_fifa(teams, ranking_df):
    mapping = {}

    for team in teams:
        keys = acceptable_keys(team)
        direct = ranking_df[ranking_df["key"].isin(keys)]

        if len(direct) == 1:
            row = direct.iloc[0]
            mapping[team] = {
                "id_team": str(row["id_team"]),
                "fifa_rank": int(row["fifa_rank"]),
                "fifa_name": row["fifa_name"],
            }
            continue

        possibles = []

        for _, row in ranking_df.iterrows():
            row_key = row["key"]
            if any(k in row_key or row_key in k for k in keys):
                possibles.append(row)

        if len(possibles) != 1:
            raise ValueError(
                f"Could not uniquely match '{team}' to FIFA rankings "
                f"(direct={len(direct)}, fallback={len(possibles)})"
            )

        row = possibles[0]
        mapping[team] = {
            "id_team": str(row["id_team"]),
            "fifa_rank": int(row["fifa_rank"]),
            "fifa_name": row["fifa_name"],
        }

    return mapping


def parse_team_match(item, target_team_id):
    date = pd.to_datetime(
        item.get("Date"),
        errors="coerce",
        utc=True,
    )

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
        gf = home_score
        ga = away_score
        opponent = description(away.get("TeamName"))
    elif away_id == str(target_team_id):
        gf = away_score
        ga = home_score
        opponent = description(home.get("TeamName"))
    else:
        return None

    return {
        "date": date,
        "gf": gf,
        "ga": ga,
        "opponent": opponent,
        "competition": description(item.get("CompetitionName")),
        "id_match": item.get("IdMatch"),
    }


def get_recent_history(team_id):
    data = fifa_get(
        "calendar/matches",
        {
            "idTeam": team_id,
            "from": RECENT_HISTORY_START,
            "to": RECENT_HISTORY_END,
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
        raise ValueError(
            f"No recent FIFA history found for team ID {team_id}"
        )

    return (
        history.sort_values("date")
        .drop_duplicates(
            subset=["date", "opponent", "gf", "ga"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def get_prior_wc_matches(team_id):
    data = fifa_get(
        "calendar/matches",
        {
            "idTeam": team_id,
            "from": WC_HISTORY_START,
            "to": WC_HISTORY_END,
            "count": 500,
        },
    )

    seen = set()
    total = 0

    for item in data.get("Results", []):
        parsed = parse_team_match(item, team_id)

        if parsed is None:
            continue

        competition = parsed["competition"].lower()

        if "world cup" not in competition:
            continue

        if "qualif" in competition:
            continue

        key = parsed["id_match"] or (
            parsed["date"],
            parsed["opponent"],
            parsed["gf"],
            parsed["ga"],
        )

        if key not in seen:
            seen.add(key)
            total += 1

    return total


def recent_features(history, match_date, team, match_id):
    prior = history[history["date"] < match_date].tail(LOOKBACK).copy()

    if len(prior) != LOOKBACK:
        raise ValueError(
            f"Match {match_id}, {team}: only {len(prior)} prior matches; "
            f"need {LOOKBACK}"
        )

    x = np.arange(LOOKBACK, dtype=float)
    gf = prior["gf"].to_numpy(dtype=float)
    ga = prior["ga"].to_numpy(dtype=float)

    recent_gf_avg = float(gf.mean())
    recent_ga_avg = float(ga.mean())

    scoring_momentum = float(np.polyfit(x, gf, 1)[0])
    multi_goal_rate = float((gf >= 2).mean())
    scoring_volatility = float(np.std(gf, ddof=0))

    last_match_date = prior["date"].max()
    rest_days = int((match_date - last_match_date).days)

    return {
        "recent_gf_avg": recent_gf_avg,
        "recent_ga_avg": recent_ga_avg,
        "scoring_momentum": scoring_momentum,
        "multi_goal_rate": multi_goal_rate,
        "scoring_volatility": scoring_volatility,
        "rest_days": rest_days,
        "history_dates": "|".join(
            prior["date"].dt.strftime("%Y-%m-%d").tolist()
        ),
        "history_opponents": "|".join(
            prior["opponent"].fillna("").astype(str).tolist()
        ),
        "history_scores": "|".join(
            (
                prior["gf"].astype(int).astype(str)
                + "-"
                + prior["ga"].astype(int).astype(str)
            ).tolist()
        ),
    }


def main():
    matches = pd.read_csv(MATCH_FILE)

    matches["date"] = pd.to_datetime(
        matches["date"],
        format="%Y-%m-%d"
    )

    required = {
        "match_id",
        "date",
        "stage",
        "team_a",
        "team_b",
        "team_a_goals",
        "team_b_goals",
    }

    missing_cols = required.difference(matches.columns)

    if missing_cols:
        raise ValueError(
            "Missing required match columns: "
            + ", ".join(sorted(missing_cols))
        )

    if len(matches) != 104:
        raise ValueError(
            f"Expected 104 matches; found {len(matches)}"
        )

    teams = sorted(
        set(matches["team_a"]) | set(matches["team_b"])
    )

    if len(teams) != 48:
        raise ValueError(
            f"Expected 48 teams; found {len(teams)}"
        )

    print("Loading pre-tournament FIFA rankings...")
    ranking_df, ranking_snapshot = get_pre_tournament_rankings()
    team_info = match_teams_to_fifa(teams, ranking_df)

    print("\nLoading historical FIFA match data...")
    histories = {}
    prior_wc_counts = {}

    for i, team in enumerate(teams, 1):
        print(f"[{i:02d}/48] {team}")

        team_id = team_info[team]["id_team"]

        histories[team] = get_recent_history(team_id)
        prior_wc_counts[team] = get_prior_wc_matches(team_id)

    rows = []
    audit_rows = []

    for _, match in matches.sort_values(
        ["date", "match_id"]
    ).iterrows():

        match_id = int(match["match_id"])
        match_date = match["date"]

        sides = [
            (
                match["team_a"],
                match["team_b"],
                int(match["team_a_goals"]),
            ),
            (
                match["team_b"],
                match["team_a"],
                int(match["team_b_goals"]),
            ),
        ]

        for team, opponent, goals_scored in sides:
            team_recent = recent_features(
                histories[team],
                match_date,
                team,
                match_id,
            )

            opponent_recent = recent_features(
                histories[opponent],
                match_date,
                opponent,
                match_id,
            )

            attack_matchup_index = (
                team_recent["recent_gf_avg"]
                - opponent_recent["recent_ga_avg"]
            )

            row = {
                "match_id": match_id,
                "date": match_date.date().isoformat(),
                "stage": match["stage"],
                "team": team,
                "opponent": opponent,

                # Exactly 8 explanatory variables
                "fifa_rank": team_info[team]["fifa_rank"],
                "prior_wc_matches": prior_wc_counts[team],
                "rest_days": team_recent["rest_days"],
                "host_team": 1 if team in HOSTS else 0,
                "attack_matchup_index": round(
                    attack_matchup_index, 4
                ),
                "scoring_momentum": round(
                    team_recent["scoring_momentum"], 4
                ),
                "multi_goal_rate": round(
                    team_recent["multi_goal_rate"], 4
                ),
                "scoring_volatility": round(
                    team_recent["scoring_volatility"], 4
                ),

                # Target
                "goals_scored": goals_scored,
            }

            rows.append(row)

            audit_rows.append({
                "match_id": match_id,
                "date": match_date.date().isoformat(),
                "team": team,
                "opponent": opponent,
                "ranking_snapshot": ranking_snapshot,
                "fifa_team_id": team_info[team]["id_team"],
                "fifa_rank": team_info[team]["fifa_rank"],
                "prior_wc_matches": prior_wc_counts[team],
                "team_history_dates": team_recent["history_dates"],
                "team_history_opponents": team_recent["history_opponents"],
                "team_history_scores": team_recent["history_scores"],
                "opponent_history_dates": opponent_recent["history_dates"],
                "opponent_history_opponents": opponent_recent["history_opponents"],
                "opponent_history_scores": opponent_recent["history_scores"],
                "team_recent_gf_avg": round(
                    team_recent["recent_gf_avg"], 4
                ),
                "opponent_recent_ga_avg": round(
                    opponent_recent["recent_ga_avg"], 4
                ),
                "attack_matchup_index": round(
                    attack_matchup_index, 4
                ),
            })

    df = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)

    FEATURES = [
        "fifa_rank",
        "prior_wc_matches",
        "rest_days",
        "host_team",
        "attack_matchup_index",
        "scoring_momentum",
        "multi_goal_rate",
        "scoring_volatility",
    ]

    if len(df) != 208:
        raise ValueError(
            f"Expected 208 team-match rows; found {len(df)}"
        )

    if df["match_id"].nunique() != 104:
        raise ValueError(
            f"Expected 104 unique match IDs; found "
            f"{df['match_id'].nunique()}"
        )

    rows_per_match = df.groupby("match_id").size()

    if not (rows_per_match == 2).all():
        raise ValueError(
            "Every match must produce exactly 2 team rows"
        )

    if len(FEATURES) != 8:
        raise ValueError(
            "Task 2.2 must contain exactly 8 explanatory variables"
        )

    missing_values = int(
        df[FEATURES + ["goals_scored"]]
        .isna()
        .sum()
        .sum()
    )

    if missing_values != 0:
        raise ValueError(
            f"Found {missing_values} missing modelling values"
        )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    audit.to_csv(
        AUDIT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nTASK 2.2 VALIDATION PASSED")
    print("==========================")
    print("Rows:", len(df))
    print("Unique matches:", df["match_id"].nunique())
    print("Rows per match: 2")
    print("Explanatory variables:", len(FEATURES))
    print("Missing modelling values:", missing_values)

    print("\nEXACT 8 EXPLANATORY VARIABLES")
    print("-----------------------------")

    for i, feature in enumerate(FEATURES, 1):
        print(f"{i}. {feature}")

    print("\nOVERLAP WITH FINAL TASK 2.1")
    print("---------------------------")
    print("1. FIFA ranking")
    print("2. Previous World Cup experience")
    print("3. Rest/recovery")
    print("4. Host status")
    print("Total conceptual overlap: 4")

    print("\nNEW TASK 2.2-SPECIFIC VARIABLES")
    print("-------------------------------")
    print("1. attack_matchup_index")
    print("2. scoring_momentum")
    print("3. multi_goal_rate")
    print("4. scoring_volatility")

    print(f"\nSaved: {OUTPUT_FILE.name}")
    print(f"Saved: {AUDIT_FILE.name}")


if __name__ == "__main__":
    main()
