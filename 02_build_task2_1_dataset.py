"""
Builds the Task 2.1 match-level dataset.

Outputs:
- task2_1_match_level_dataset.csv
- task2_1_feature_audit.csv
"""

import csv
import hashlib
import html
import re
import time
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "matches_104_raw.csv"
OUTPUT_FILE = BASE_DIR / "task2_1_match_level_dataset.csv"
AUDIT_FILE = BASE_DIR / "task2_1_feature_audit.csv"
CACHE_DIR = BASE_DIR / "fbref_cache"

FBREF_BASE = "https://fbref.com"
FBREF_WORLD_CUP = "https://fbref.com/en/comps/1/2026/stats/2026-World-Cup-Stats"

HOSTS = {"Canada", "Mexico", "USA"}
LOOKBACK = 5
REQUEST_DELAY_SECONDS = 4

FEATURES = [
    "recent_ppg_diff",
    "recent_gf_avg_diff",
    "recent_ga_avg_diff",
    "recent_clean_sheet_rate_diff",
    "recent_scoring_rate_diff",
    "rest_days_diff",
    "host_advantage",
    "knockout",
]

# Names in our master file that may differ from FBRef's display names.
ALIASES = {
    "USA": "United States",
    "South Korea": "Korea Republic",
    "Cape Verde": "Cabo Verde",
    "Czech Republic": "Czechia",
    "D.R. Congo": "Congo DR",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
    "Turkey": "Türkiye",
}


def normalise_name(value):
    """Create a comparison key for country/team names."""
    value = html.unescape(str(value))
    value = value.replace("&", " and ")
    value = re.sub(r"<[^>]+>", "", value)
    value = value.lower().strip()
    value = (
        value.replace("é", "e")
        .replace("ë", "e")
        .replace("ï", "i")
        .replace("ö", "o")
        .replace("ü", "u")
        .replace("ç", "c")
        .replace("ã", "a")
        .replace("á", "a")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def cached_get(url):
    """Download a page once and cache the HTML locally."""
    CACHE_DIR.mkdir(exist_ok=True)
    name = hashlib.sha256(url.encode("utf-8")).hexdigest() + ".html"
    cached = CACHE_DIR / name

    if cached.exists():
        return cached.read_text(encoding="utf-8")

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=60) as response:
        text = response.read().decode("utf-8", errors="replace")

    cached.write_text(text, encoding="utf-8")
    time.sleep(REQUEST_DELAY_SECONDS)
    return text


def discover_world_cup_teams():
    """
    Discover FBRef team pages from the official 2026 World Cup competition page.

    Returns a dictionary:
        normalised display name -> absolute team page URL
    """
    page = cached_get(FBREF_WORLD_CUP)

    links = {}
    pattern = re.compile(
        r'<a[^>]+href="(?P<href>/en/squads/[0-9a-f]+/[^"]*Men-Stats)"[^>]*>'
        r'(?P<name>.*?)</a>',
        flags=re.I | re.S,
    )

    for match in pattern.finditer(page):
        display = re.sub(r"<[^>]+>", "", match.group("name"))
        display = html.unescape(display).strip()
        if display:
            links[normalise_name(display)] = FBREF_BASE + html.unescape(match.group("href"))

    if len(links) < 48:
        raise ValueError(
            f"Only discovered {len(links)} national-team links on FBRef. "
            "FBRef may have changed its page layout."
        )

    return links


def find_schedule_url(team_page_url):
    """Find the team's 2026 all-competitions Scores & Fixtures URL."""
    page = cached_get(team_page_url)

    # FBRef can expose either /matchlogs/schedule/ or
    # /matchlogs/all_comps/schedule/ links.
    candidates = re.findall(
        r'href="([^"]*/2026/matchlogs/(?:all_comps/)?schedule/[^"]+)"',
        page,
        flags=re.I,
    )

    if not candidates:
        raise ValueError(f"Could not find a 2026 schedule page from {team_page_url}")

    # Prefer all competitions if available.
    candidates = [html.unescape(x) for x in candidates]
    candidates.sort(key=lambda x: ("all_comps" not in x, len(x)))

    href = candidates[0]
    if href.startswith("http"):
        return href
    return FBREF_BASE + href


def flatten_columns(frame):
    """Turn pandas MultiIndex HTML-table columns into simple names."""
    if isinstance(frame.columns, pd.MultiIndex):
        flat = []
        for col in frame.columns:
            parts = [str(x).strip() for x in col if str(x) != "nan"]
            # Usually the final level is the useful label: Date, GF, GA, etc.
            flat.append(parts[-1] if parts else "")
        frame.columns = flat
    else:
        frame.columns = [str(x).strip() for x in frame.columns]
    return frame


def read_schedule(schedule_url):
    """Read a FBRef Scores & Fixtures page into a clean historical table."""
    page = cached_get(schedule_url)
    tables = pd.read_html(StringIO(page))

    selected = None
    for table in tables:
        table = flatten_columns(table.copy())
        needed = {"Date", "Result", "GF", "GA", "Opponent"}
        if needed.issubset(set(table.columns)):
            selected = table
            break

    if selected is None:
        raise ValueError(f"Could not find Scores & Fixtures table at {schedule_url}")

    df = selected[["Date", "Result", "GF", "GA", "Opponent"]].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["GF"] = pd.to_numeric(df["GF"], errors="coerce")
    df["GA"] = pd.to_numeric(df["GA"], errors="coerce")

    # Remove headings, future fixtures, and rows without a completed score.
    df = df.dropna(subset=["Date", "GF", "GA", "Result"])
    df["Result"] = df["Result"].astype(str).str.strip().str[0]
    df = df[df["Result"].isin(["W", "D", "L"])]
    df = df.sort_values("Date").drop_duplicates(
        subset=["Date", "Opponent", "GF", "GA"], keep="last"
    )
    return df.reset_index(drop=True)


def get_team_url(master_name, discovered):
    """Match our team name to the appropriate FBRef page."""
    target = ALIASES.get(master_name, master_name)
    key = normalise_name(target)

    if key in discovered:
        return discovered[key]

    # Safe fallback: exact containment after normalisation.
    possibilities = [
        (name, url)
        for name, url in discovered.items()
        if key in name or name in key
    ]
    if len(possibilities) == 1:
        return possibilities[0][1]

    raise ValueError(
        f"Could not uniquely match '{master_name}' to an FBRef team. "
        f"Target key='{key}', candidates={[x[0] for x in possibilities]}"
    )


def historical_features(history, match_date):
    """
    Calculate a team's pre-match features using only the five latest matches
    with date strictly earlier than match_date.
    """
    prior = history[history["Date"] < match_date].tail(LOOKBACK).copy()

    if len(prior) != LOOKBACK:
        raise ValueError(
            f"Need {LOOKBACK} completed matches before {match_date.date()}, "
            f"but found {len(prior)}."
        )

    points = prior["Result"].map({"W": 3, "D": 1, "L": 0})

    return {
        "recent_ppg": points.mean(),
        "recent_gf_avg": prior["GF"].mean(),
        "recent_ga_avg": prior["GA"].mean(),
        "recent_clean_sheet_rate": (prior["GA"] == 0).mean(),
        "recent_scoring_rate": (prior["GF"] > 0).mean(),
        "rest_days": int((match_date - prior["Date"].max()).days),
        "history_first_date": prior["Date"].min().date().isoformat(),
        "history_last_date": prior["Date"].max().date().isoformat(),
    }


def validate_output(df):
    """Assessment-specific checks before saving."""
    if len(df) != 104:
        raise ValueError(f"Expected 104 rows; found {len(df)}")

    if df["match_id"].nunique() != 104:
        raise ValueError("match_id is not unique")

    if len(FEATURES) != 8:
        raise ValueError(f"Expected exactly 8 explanatory variables; found {len(FEATURES)}")

    missing = df[FEATURES + ["goal_difference"]].isna().sum()
    if missing.any():
        raise ValueError(f"Missing modelling values:\n{missing[missing > 0]}")

    if not set(df["host_advantage"].unique()).issubset({-1, 0, 1}):
        raise ValueError("host_advantage contains an unexpected value")

    if not set(df["knockout"].unique()).issubset({0, 1}):
        raise ValueError("knockout contains an unexpected value")

    print("Checks passed:")
    print("  104 match rows")
    print("  104 unique match IDs")
    print("  exactly 8 explanatory variables")
    print("  no missing explanatory/response values")
    print("  all rolling features use only prior matches")


def main():
    if not MASTER_FILE.exists():
        raise FileNotFoundError(
            f"{MASTER_FILE.name} was not found. Put this script beside the master CSV."
        )

    matches = pd.read_csv(MASTER_FILE)
    matches["date"] = pd.to_datetime(matches["date"])

    required = {
        "match_id", "date", "stage", "team_a", "team_b",
        "team_a_goals", "team_b_goals"
    }
    missing_cols = required.difference(matches.columns)
    if missing_cols:
        raise ValueError(f"Master file is missing columns: {sorted(missing_cols)}")

    teams = sorted(set(matches["team_a"]) | set(matches["team_b"]))
    if len(teams) != 48:
        raise ValueError(f"Expected 48 teams in master file; found {len(teams)}")

    print("Discovering the 48 World Cup teams on FBRef...")
    discovered = discover_world_cup_teams()

    histories = {}
    source_urls = {}

    for number, team in enumerate(teams, 1):
        team_page = get_team_url(team, discovered)
        schedule_url = find_schedule_url(team_page)

        print(f"[{number:02d}/48] {team}")
        print(f"         {schedule_url}")

        histories[team] = read_schedule(schedule_url)
        source_urls[team] = schedule_url

    rows = []
    audit_rows = []

    for _, match in matches.sort_values(["date", "match_id"]).iterrows():
        match_date = match["date"]

        a = historical_features(histories[match["team_a"]], match_date)
        b = historical_features(histories[match["team_b"]], match_date)

        row = {
            "match_id": int(match["match_id"]),
            "date": match_date.date().isoformat(),
            "stage": match["stage"],
            "team_a": match["team_a"],
            "team_b": match["team_b"],

            # Exactly eight explanatory variables:
            "recent_ppg_diff": a["recent_ppg"] - b["recent_ppg"],
            "recent_gf_avg_diff": a["recent_gf_avg"] - b["recent_gf_avg"],
            "recent_ga_avg_diff": a["recent_ga_avg"] - b["recent_ga_avg"],
            "recent_clean_sheet_rate_diff":
                a["recent_clean_sheet_rate"] - b["recent_clean_sheet_rate"],
            "recent_scoring_rate_diff":
                a["recent_scoring_rate"] - b["recent_scoring_rate"],
            "rest_days_diff": a["rest_days"] - b["rest_days"],
            "host_advantage":
                int(match["team_a"] in HOSTS) - int(match["team_b"] in HOSTS),
            "knockout": 0 if match["stage"] == "Group Stage" else 1,

            # Response variable:
            "goal_difference":
                int(match["team_a_goals"]) - int(match["team_b_goals"]),
        }
        rows.append(row)

        audit_rows.append({
            "match_id": int(match["match_id"]),
            "date": match_date.date().isoformat(),
            "team_a": match["team_a"],
            "team_a_fbref_source": source_urls[match["team_a"]],
            "team_a_history_first_date": a["history_first_date"],
            "team_a_history_last_date": a["history_last_date"],
            "team_b": match["team_b"],
            "team_b_fbref_source": source_urls[match["team_b"]],
            "team_b_history_first_date": b["history_first_date"],
            "team_b_history_last_date": b["history_last_date"],
        })

    output = pd.DataFrame(rows)
    validate_output(output)

    # Round rates/averages for a readable CSV while retaining sufficient precision.
    numeric_to_round = [
        "recent_ppg_diff",
        "recent_gf_avg_diff",
        "recent_ga_avg_diff",
        "recent_clean_sheet_rate_diff",
        "recent_scoring_rate_diff",
    ]
    output[numeric_to_round] = output[numeric_to_round].round(4)

    output.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(AUDIT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved {OUTPUT_FILE.name}")
    print(f"Saved {AUDIT_FILE.name}")
    print("\nFinal explanatory variables:")
    for i, feature in enumerate(FEATURES, 1):
        print(f"  {i}. {feature}")


if __name__ == "__main__":
    main()
