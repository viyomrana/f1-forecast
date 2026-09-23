"""
F1 Forecast — Starter Data Access Script
==========================================
This script does two things, one at a time, with plenty of print statements
so you can see exactly what's happening at each stage:

  1. Pulls a real FastF1 session (lap times, results) for the 2025 season
     — this is your PERFORMANCE data source.
  2. Pulls driver standings from the Jolpica-F1 API for an older season
     — this is your CONTEXTUAL/historical data source.

Run this with your virtual environment active:
    python f1_data_pull.py

Reminder of the project's hard rule: only 2026-onward data may be used
for performance/telemetry features. Anything from FastF1 for years before
2026 should only ever be treated as contextual, never as a performance
feature, in line with the new regulation era.
"""

import fastf1
import requests
import os


def setup_fastf1_cache():
    """
    FastF1 downloads a LOT of data per session. Without a cache, every
    time you re-run this script it re-downloads everything from scratch.
    This function makes sure a local cache folder exists and tells
    FastF1 to use it.
    """
    cache_dir = "f1_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        print(f"Created cache folder at ./{cache_dir}")
    else:
        print(f"Cache folder already exists at ./{cache_dir}")

    fastf1.Cache.enable_cache(cache_dir)
    print("FastF1 caching enabled.\n")


def pull_fastf1_session(year: int, gp_name: str, session_type: str = "R"):
    """
    Loads a single race weekend session from FastF1.

    year          : e.g. 2025
    gp_name       : e.g. "Bahrain" — FastF1 matches this loosely against
                    the official event name
    session_type  : "R" = Race, "Q" = Qualifying, "FP1"/"FP2"/"FP3" = Practice
    """
    print(f"Requesting FastF1 session: {year} {gp_name} ({session_type})...")
    session = fastf1.get_session(year, gp_name, session_type)
    session.load()  # this is the line that actually triggers the download

    print(f"Loaded session: {session.event['EventName']} {year}")
    print("First 5 rows of results:")
    print(session.results.head())
    print()

    return session


def pull_jolpica_driver_standings(year: int):
    """
    Calls the Jolpica-F1 REST API (a community-run continuation of the
    old Ergast API) to get final driver standings for a given season.
    No API key is required for this endpoint.
    """
    url = f"https://api.jolpi.ca/ergast/f1/{year}/driverStandings.json"
    print(f"Requesting Jolpica-F1 driver standings for {year}...")

    response = requests.get(url, timeout=10)
    response.raise_for_status()  # will raise an error if the request failed

    data = response.json()

    # The JSON structure is nested — this digs down to the actual list
    # of standings. Printing the raw structure once (data) is a good way
    # to explore this yourself if the shape ever looks unfamiliar.
    standings_list = (
        data["MRData"]["StandingsTable"]["StandingsLists"][0]["DriverStandings"]
    )

    print(f"Top 5 drivers, {year} season:")
    for entry in standings_list[:5]:
        driver_name = f"{entry['Driver']['givenName']} {entry['Driver']['familyName']}"
        points = entry["points"]
        position = entry["position"]
        print(f"  {position}. {driver_name} — {points} pts")
    print()

    return standings_list


if __name__ == "__main__":
    print("=== F1 Forecast: Data Access Test ===\n")

    # Step 1: FastF1 (performance data source)
    setup_fastf1_cache()
    pull_fastf1_session(year=2026, gp_name="Barcelona", session_type="R")

    # Step 2: Jolpica-F1 (contextual/historical data source)
    pull_jolpica_driver_standings(year=2024)

    print("Done. If both sections above printed real data, you're ready")
    print("to move on to pre-processing.")