"""
Tests the FIFA API endpoints used in this project.
No files are written.
"""

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE = "https://api.fifa.com/api/v3"


def get(endpoint, params):
    url = BASE + "/" + endpoint + "?" + urlencode(params)
    print("\nGET", url)
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )
    with urlopen(request, timeout=60) as r:
        return json.loads(r.read().decode("utf-8-sig"))


schedules = get(
    "rankingschedules/all",
    {
        "type": 0,
        "gender": 1,
        "idClient": "64e9afa8-c5c0-413d-882b-bc9e6a81e264",
        "language": "en",
    },
)
print("Ranking schedules returned:", len(schedules.get("Results", [])))

france = get(
    "rankings/byCountry/FRA",
    {"gender": 1, "language": "en"},
)
fr = france["Results"][0]
print("France rank:", fr.get("Rank"))
print("France IdTeam:", fr.get("IdTeam"))

matches = get(
    "calendar/matches",
    {
        "idTeam": fr.get("IdTeam"),
        "language": "en",
        "count": 20,
        "from": "2025-01-01T00:00:00Z",
        "to": "2026-06-10T23:59:59Z",
    },
)
print("France historical matches returned:", len(matches.get("Results", [])))

if matches.get("Results"):
    sample = matches["Results"][-1]
    print("Sample date:", sample.get("Date"))
    print("Sample score:", sample.get("HomeTeamScore"), "-", sample.get("AwayTeamScore"))

print("\nFIFA API test completed successfully.")
