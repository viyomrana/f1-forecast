"""
F1 Forecast - Phase 1: Data Ingestion (fetch and cache half)
============================================================
This script downloads 2026 race data from the Jolpica-F1 API and saves
every response as a raw JSON file in data/raw/. It does NOT clean the
data or build features - that is the job of build_features.py (next step).

For every COMPLETED 2026 round it saves four files:
  - results            : finishing positions, grid positions, status (DNF etc.)
  - qualifying         : Q1 / Q2 / Q3 times
  - driver_standings   : driver championship table AFTER that round
  - constructor_standings : team championship table AFTER that round

Important for later (avoiding leakage): standings saved for round N are the
standings AFTER race N. To predict race N you must use the standings from
round N-1, never round N.

Safe to re-run: files that already exist are skipped, so nothing is
downloaded twice. Run it again after each new race to pick up the new round.

Run this with your virtual environment active:
    python ingest_2026.py

Reminder of the project's hard rule: only 2026 data is used for performance
features, because of the new regulation era.
"""

import json
import os
import time
from datetime import date

import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
SEASON = 2026
RAW_DIR = os.path.join("data", "raw")
PAUSE_SECONDS = 0.5  # small pause between calls to stay inside Jolpica's rate limits

# (name used in file, API endpoint, top-level table in JSON, list key inside the table)
DATASETS = [
    ("results", "results.json", "RaceTable", "Races"),
    ("qualifying", "qualifying.json", "RaceTable", "Races"),
    ("driver_standings", "driverStandings.json", "StandingsTable", "StandingsLists"),
    ("constructor_standings", "constructorStandings.json", "StandingsTable", "StandingsLists"),
]


def fetch_json(url, retries=3):
    """
    Calls the API and returns the parsed JSON.
    If the API says "slow down" (HTTP 429), we wait and try again.
    """
    for attempt in range(1, retries + 1):
        response = requests.get(url, timeout=15)
        if response.status_code == 429:
            wait = 5 * attempt
            print(f"  Rate limited. Waiting {wait}s before retry {attempt}/{retries}...")
            time.sleep(wait)
            continue
        response.raise_for_status()  # raises an error for any other failure
        return response.json()
    raise RuntimeError(f"Gave up after {retries} attempts: {url}")


def get_rounds_to_try():
    """
    Asks the API for the 2026 calendar and returns the rounds whose race date
    has already passed, as a list of (round_number, race_name).
    """
    print(f"Fetching the {SEASON} calendar...")
    data = fetch_json(f"{BASE_URL}/{SEASON}.json?limit=100")
    races = data["MRData"]["RaceTable"]["Races"]
    today = date.today().isoformat()  # e.g. "2026-10-17"; ISO dates compare correctly as text
    rounds = [(int(r["round"]), r["raceName"]) for r in races if r["date"] <= today]
    print(f"Calendar has {len(races)} rounds; {len(rounds)} are dated on or before today.\n")
    return rounds


def fetch_and_cache(name, round_number, endpoint, table, list_key):
    """
    Downloads one dataset for one round and saves it to data/raw/.
    Returns "cached", "downloaded" or "empty".
    "empty" means the API has no data yet (race not finished) - we do NOT
    save anything in that case, so a later run can fill it in.
    """
    path = os.path.join(RAW_DIR, f"{name}_round_{round_number:02d}.json")

    if os.path.exists(path):
        return "cached"

    data = fetch_json(f"{BASE_URL}/{SEASON}/{round_number}/{endpoint}")
    time.sleep(PAUSE_SECONDS)

    if not data["MRData"][table][list_key]:
        return "empty"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return "downloaded"


def print_summary():
    """
    Reads the saved results files and prints how much data we now have,
    so you can check that the download worked.
    """
    files = sorted(f for f in os.listdir(RAW_DIR) if f.startswith("results_round_"))
    total_rows = 0
    for filename in files:
        with open(os.path.join(RAW_DIR, filename), encoding="utf-8") as f:
            data = json.load(f)
        total_rows += len(data["MRData"]["RaceTable"]["Races"][0]["Results"])

    print("=== Summary ===")
    print(f"Rounds with saved results : {len(files)}")
    print(f"Driver result rows        : {total_rows}")
    print(f"Files are in              : ./{RAW_DIR}/")


if __name__ == "__main__":
    print("=== F1 Forecast: 2026 Data Ingestion ===\n")

    os.makedirs(RAW_DIR, exist_ok=True)

    for round_number, race_name in get_rounds_to_try():
        print(f"Round {round_number:02d} - {race_name}")
        results_status = None

        for name, endpoint, table, list_key in DATASETS:
            status = fetch_and_cache(name, round_number, endpoint, table, list_key)
            print(f"  {name:<22} {status}")
            if name == "results":
                results_status = status

        # If a race has no results yet, later rounds won't either - stop here.
        if results_status == "empty":
            print("  No results yet for this round. Stopping.\n")
            break
        print()

    print_summary()
