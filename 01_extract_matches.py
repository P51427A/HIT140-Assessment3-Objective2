"""
Creates the 104-match World Cup 2026 results dataset.

Output:
matches_104_raw.csv
"""

import csv
import html
import re
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen


RESULTS_PAGE = "https://www.thestatsdontlie.com/football/world-cup-2026/"
BRONZE_PAGE = (
    "https://www.fifa.com/en/tournaments/mens/worldcup/"
    "canadamexicousa2026/articles/france-england-report-highlights-bronze-final"
)
OUTPUT = Path(__file__).with_name("matches_104_raw.csv")
STAGES = {
    "GROUP STAGE": "Group Stage",
    "ROUND OF 32": "Round of 32",
    "ROUND OF 16": "Round of 16",
    "QUARTER FINALS": "Quarterfinals",
    "SEMI FINALS": "Semifinals",
    "FINAL": "Final",
}
FIELDS = (
    "match_id", "date", "stage", "team_a", "team_b",
    "team_a_goals", "team_b_goals", "decided_on_penalties", "source_page",
)


def get_text(url):
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8-sig")


def published_sheet_url(page):
    # The Stats Don't Lie embeds its match-results sheet on the named page.
    match = re.search(
        r'data-orig-src="(https://docs\.google\.com/spreadsheets/d/e/'
        r'[^\"]+/pubhtml\?gid=995472238[^\"]*)"', page
    )
    if not match:
        raise ValueError("Could not find the results sheet on The Stats Don't Lie page")
    embedded = html.unescape(match.group(1))
    return embedded.split("/pubhtml?", 1)[0] + "/pub?gid=995472238&single=true&output=csv"


def extract_matches(sheet_text):
    matches = []
    stage = None
    for cells in csv.reader(StringIO(sheet_text)):
        if not cells:
            continue
        if cells[0] in STAGES:
            stage = STAGES[cells[0]]
            continue
        if not re.fullmatch(r"\d{1,2}/\d{1,2}/2026", cells[0]):
            continue
        if stage is None or len(cells) < 5:
            raise ValueError(f"Unrecognised match row: {cells[:5]}")
        score_a = re.sub(r"\D", "", cells[2])
        score_b = re.sub(r"\D", "", cells[3])
        if not score_a or not score_b:
            raise ValueError(f"Missing score: {cells[:5]}")
        matches.append({
            "date": datetime.strptime(cells[0], "%d/%m/%Y").date().isoformat(),
            "stage": stage,
            "team_a": cells[1].strip(),
            "team_b": cells[4].strip(),
            "team_a_goals": int(score_a),
            "team_b_goals": int(score_b),
            "decided_on_penalties": "yes" if "p" in (cells[2] + cells[3]).lower() else "no",
            "source_page": RESULTS_PAGE,
        })
    return matches


def check(matches):
    problems = []
    if len(matches) != 104:
        problems.append(f"Expected 104 matches; found {len(matches)}")
    teams = {team for match in matches for team in (match["team_a"], match["team_b"])}
    if len(teams) != 48:
        problems.append(f"Expected 48 teams; found {len(teams)}")
    fixtures = [(m["date"], m["team_a"], m["team_b"]) for m in matches]
    duplicates = [key for key, count in Counter(fixtures).items() if count > 1]
    if duplicates:
        problems.append(f"Duplicate matches: {duplicates}")
    for match in matches:
        if not match["team_a"] or not match["team_b"]:
            problems.append(f"Missing team: {match}")
        if match["team_a"] == match["team_b"]:
            problems.append(f"Same team twice: {match}")
        if match["team_a_goals"] is None or match["team_b_goals"] is None:
            problems.append(f"Missing score: {match}")
    if problems:
        raise ValueError("Checks failed:\n" + "\n".join(problems))
    print("Checks passed: 104 matches, 48 teams, no duplicates or missing teams/scores")


def main():
    sheet_url = published_sheet_url(get_text(RESULTS_PAGE))
    matches = extract_matches(get_text(sheet_url))
    if len(matches) != 103:
        raise ValueError("The embedded sheet has changed; review it before adding the missing match")

    # The embedded sheet omits the third-place game. FIFA's published match
    # report records France 4–6 England on 18 July 2026.
    matches.append({
        "date": "2026-07-18", "stage": "Third-place match",
        "team_a": "France", "team_b": "England",
        "team_a_goals": 4, "team_b_goals": 6,
        "decided_on_penalties": "no", "source_page": BRONZE_PAGE,
    })
    matches.sort(key=lambda match: match["date"])
    for number, match in enumerate(matches, 1):
        match["match_id"] = number  # Our row ID; it is not FIFA's official match number.
    check(matches)
    with OUTPUT.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(matches)
    print(f"Saved {OUTPUT.name}")


if __name__ == "__main__":
    main()