"""
F1 Forecast - Phase 3 (part 1): Lap Data Ingestion
==================================================
Downloads lap-by-lap race data from FastF1 for every completed 2026 round and
saves one CSV per round in data/raw/laps/. This is the raw material for the
tyre degradation model (lap time vs tyre age, per compound).

It does NOT clean the data or fit any model - that comes next.

Which rounds? Every round that already has a results file in data/raw/
(created by ingest_2026.py), so the two ingestion scripts always agree.

Safe to re-run: rounds that already have a CSV are skipped. If one round
fails (network problem, data not available yet), the script reports it and
moves on to the next one.

Run this with your virtual environment active:
    python ingest_laps_2026.py            # all completed rounds
    python ingest_laps_2026.py 1          # just round 1 (good for a first test)

The first run is slow (FastF1 downloads a lot per race). Later runs are fast
because everything is cached in ./f1_cache (which is in .gitignore).

Project hard rule: only 2026 data is used for performance features.
"""

import os
import sys

import fastf1
import pandas as pd

SEASON = 2026
RAW_DIR = os.path.join("data", "raw")
LAPS_DIR = os.path.join(RAW_DIR, "laps")
CACHE_DIR = "f1_cache"

# Columns we keep from FastF1's laps table (missing ones are skipped safely)
KEEP_COLUMNS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "Stint", "Compound",
    "TyreLife", "FreshTyre", "LapTime_s", "is_pit_in_lap", "is_pit_out_lap",
    "TrackStatus", "IsAccurate", "Position", "Deleted",
]


def completed_rounds():
    """Round numbers that already have a results file from ingest_2026.py."""
    if not os.path.isdir(RAW_DIR):
        raise SystemExit("data/raw/ not found. Run ingest_2026.py first.")
    rounds = []
    for filename in os.listdir(RAW_DIR):
        if filename.startswith("results_round_"):
            rounds.append(int(filename.split("_round_")[1].split(".")[0]))
    return sorted(rounds)


def fetch_laps(round_number):
    """Loads the race session for one round and returns a tidy laps DataFrame."""
    session = fastf1.get_session(SEASON, round_number, "R")
    # We only need laps: no telemetry, weather or race-control messages
    session.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = session.laps.copy()
    if laps.empty:
        return laps, session.event["EventName"]

    # Timedelta columns -> plain seconds / simple flags (easier to model later)
    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps["is_pit_in_lap"] = laps["PitInTime"].notna().astype(int)
    laps["is_pit_out_lap"] = laps["PitOutTime"].notna().astype(int)

    columns = [c for c in KEEP_COLUMNS if c in laps.columns]
    laps = laps[columns]
    laps.insert(0, "round", round_number)
    laps.insert(1, "race_name", session.event["EventName"])
    return laps, session.event["EventName"]


def main():
    print("=== F1 Forecast: 2026 Lap Data Ingestion ===\n")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(LAPS_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(CACHE_DIR)

    rounds = completed_rounds()
    if len(sys.argv) > 1:  # optional: python ingest_laps_2026.py 1 2 3
        wanted = {int(a) for a in sys.argv[1:]}
        rounds = [r for r in rounds if r in wanted]
    print(f"Rounds to process: {rounds}\n")

    downloaded, cached, failed = 0, 0, []

    for round_number in rounds:
        path = os.path.join(LAPS_DIR, f"laps_round_{round_number:02d}.csv")
        if os.path.exists(path):
            print(f"Round {round_number:02d}: cached")
            cached += 1
            continue

        try:
            laps, race_name = fetch_laps(round_number)
        except Exception as error:  # keep going even if one round fails
            print(f"Round {round_number:02d}: FAILED ({type(error).__name__}: {error})")
            failed.append(round_number)
            continue

        if laps.empty:
            print(f"Round {round_number:02d}: {race_name} - no lap data available, not saved")
            failed.append(round_number)
            continue

        laps.to_csv(path, index=False)
        print(f"Round {round_number:02d}: {race_name} - saved {len(laps)} laps")
        downloaded += 1

    print("\n=== Summary ===")
    print(f"Downloaded : {downloaded}")
    print(f"Cached     : {cached}")
    print(f"Failed     : {failed if failed else 'none'}")
    print(f"Files are in ./{LAPS_DIR}/")


if __name__ == "__main__":
    main()
