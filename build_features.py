"""
F1 Forecast - Phase 2: Feature Engineering
==========================================
Reads the raw JSON files saved by ingest_2026.py (data/raw/) and builds ONE
table: one row per driver per race, saved to data/processed/features.csv.

Each row has:
  - identifiers   : round, race name, driver, constructor
  - FEATURES      : things we know BEFORE the race starts
  - TARGETS       : what happened in the race (what the model will predict)

THE LEAKAGE RULE (the most important thing in this file)
--------------------------------------------------------
A feature for race N may only use information available before race N starts:
  - qualifying of round N            -> OK (happens before the race)
  - standings AFTER round N-1        -> OK
  - results of rounds 1 .. N-1       -> OK
  - results or standings of round N  -> NEVER (that is the answer)
This is enforced by shifting standings by one round and by shifting every
"form" feature by one race (.shift(1)) before averaging.

Only 2026 data is used (hard project rule: new regulation era).

Run this with your virtual environment active:
    pip install pandas
    python build_features.py
"""

import json
import os
import re

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")
OUT_FILE = os.path.join(OUT_DIR, "features.csv")
FORM_WINDOW = 3  # "recent form" = average over the previous 3 races


# ---------------------------------------------------------------------------
# 1. Loading raw files
# ---------------------------------------------------------------------------
def load_raw(name):
    """Returns {round_number: parsed_json} for every saved file of one dataset."""
    files = sorted(f for f in os.listdir(RAW_DIR) if f.startswith(f"{name}_round_"))
    if not files:
        raise SystemExit(f"No '{name}' files found in {RAW_DIR}. Run ingest_2026.py first.")
    data = {}
    for filename in files:
        round_number = int(filename.split("_round_")[1].split(".")[0])
        with open(os.path.join(RAW_DIR, filename), encoding="utf-8") as f:
            data[round_number] = json.load(f)
    return data


def parse_lap_time(text):
    """Turns '1:23.456' (or '59.8') into seconds. Missing -> NaN."""
    if not text:
        return np.nan
    match = re.match(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$", text.strip())
    if not match:
        return np.nan
    minutes = int(match.group(1)) if match.group(1) else 0
    return minutes * 60 + float(match.group(2))


def to_int_or_nan(value):
    """Turns '3' into 3. Anything missing or non-numeric (None, '-') becomes NaN."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return np.nan


def is_finisher(status):
    """'Finished' and '+1 Lap', '+2 Laps'... are finishers; anything else is a DNF."""
    return status == "Finished" or status.startswith("+") or status == "Lapped"


# ---------------------------------------------------------------------------
# 2. One table per raw dataset
# ---------------------------------------------------------------------------
def build_results_table(raw):
    rows = []
    for round_number, data in raw.items():
        race = data["MRData"]["RaceTable"]["Races"][0]
        results = race["Results"]
        field_size = len(results)
        for r in results:
            grid = int(r["grid"])
            rows.append({
                "round": round_number,
                "race_name": race["raceName"],
                "driver_id": r["Driver"]["driverId"],
                "constructor_id": r["Constructor"]["constructorId"],
                "grid": grid if grid > 0 else field_size,  # grid 0 = pit-lane start -> back of grid
                "pit_lane_start": int(grid == 0),
                "finish_position": int(r["position"]),
                "points": float(r["points"]),
                "status": r["status"],
                "is_dnf": int(not is_finisher(r["status"])),
            })
    return pd.DataFrame(rows)


def build_qualifying_table(raw):
    rows = []
    for round_number, data in raw.items():
        results = data["MRData"]["RaceTable"]["Races"][0]["QualifyingResults"]
        for q in results:
            times = [parse_lap_time(q.get(k)) for k in ("Q1", "Q2", "Q3")]
            rows.append({
                "round": round_number,
                "driver_id": q["Driver"]["driverId"],
                "quali_position": int(q["position"]),
                "best_quali_time": np.nanmin(times) if not all(np.isnan(times)) else np.nan,
                "reached_q3": int(bool(q.get("Q3"))),
            })
    df = pd.DataFrame(rows)
    # gap to pole in %, so tracks of different lengths are comparable
    pole = df.groupby("round")["best_quali_time"].transform("min")
    df["quali_gap_to_pole_pct"] = (df["best_quali_time"] - pole) / pole * 100
    return df.drop(columns="best_quali_time")


def build_standings_table(raw, list_key, entity_key, id_key, prefix):
    """
    Standings AFTER round R are re-labelled as the 'before race' standings of
    round R+1. That one-line shift is what prevents leakage.
    """
    rows = []
    for as_of_round, data in raw.items():
        entries = data["MRData"]["StandingsTable"]["StandingsLists"][0][list_key]
        for e in entries:
            rows.append({
                "round": as_of_round + 1,  # <-- the leakage shift
                id_key: e[entity_key][id_key],
                f"{prefix}_points_before": float(e.get("points", 0)),
                # Some entries have no 'position' (the API leaves it out for drivers
                # without a ranking); we keep those as empty (NaN) instead of crashing.
                f"{prefix}_position_before": to_int_or_nan(e.get("position", e.get("positionText"))),
                f"{prefix}_wins_before": to_int_or_nan(e.get("wins", 0)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Form features (all shifted by one race)
# ---------------------------------------------------------------------------
def add_form_features(df):
    df = df.sort_values(["driver_id", "round"]).copy()
    g = df.groupby("driver_id")

    def prior_mean(col):
        return g[col].transform(lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean())

    df["driver_avg_finish_recent"] = prior_mean("finish_position")
    df["driver_avg_points_recent"] = prior_mean("points")
    df["driver_dnf_rate_before"] = g["is_dnf"].transform(lambda s: s.shift(1).expanding().mean())

    # How many places the driver usually gains/loses vs the grid (prior races only)
    df["places_gained"] = df["grid"] - df["finish_position"]
    df["driver_avg_places_gained"] = g["places_gained"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    df = df.drop(columns="places_gained")

    # Team form: total points both cars scored in each round, averaged over prior rounds
    team_round = df.groupby(["constructor_id", "round"], as_index=False)["points"].sum()
    team_round = team_round.sort_values(["constructor_id", "round"])
    team_round["team_avg_points_recent"] = team_round.groupby("constructor_id")["points"].transform(
        lambda s: s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()
    )
    df = df.merge(
        team_round[["constructor_id", "round", "team_avg_points_recent"]],
        on=["constructor_id", "round"], how="left",
    )
    return df


# ---------------------------------------------------------------------------
# 4. Putting it together
# ---------------------------------------------------------------------------
def main():
    print("=== F1 Forecast: Phase 2 Feature Engineering ===\n")

    results = build_results_table(load_raw("results"))
    quali = build_qualifying_table(load_raw("qualifying"))
    drv_stand = build_standings_table(
        load_raw("driver_standings"), "DriverStandings", "Driver", "driverId", "driver"
    ).rename(columns={"driverId": "driver_id"})
    con_stand = build_standings_table(
        load_raw("constructor_standings"), "ConstructorStandings", "Constructor", "constructorId", "team"
    ).rename(columns={"constructorId": "constructor_id"})

    df = add_form_features(results)
    df = df.merge(quali, on=["round", "driver_id"], how="left")
    df = df.merge(drv_stand, on=["round", "driver_id"], how="left")
    df = df.merge(con_stand, on=["round", "constructor_id"], how="left")

    # Round 1 has no earlier standings: no points yet. Positions stay empty (NaN);
    # XGBoost handles missing values natively.
    for col in ["driver_points_before", "driver_wins_before", "team_points_before", "team_wins_before"]:
        df[col] = df[col].fillna(0)

    # Targets (what the model will learn to predict)
    df["podium"] = (df["finish_position"] <= 3).astype(int)
    df["points_finish"] = (df["points"] > 0).astype(int)

    id_cols = ["round", "race_name", "driver_id", "constructor_id"]
    target_cols = ["finish_position", "points", "podium", "points_finish", "is_dnf", "status"]
    feature_cols = [c for c in df.columns if c not in id_cols + target_cols]
    df = df[id_cols + feature_cols + target_cols].sort_values(["round", "finish_position"])

    # ----- sanity checks (leakage and completeness) -----
    first_round = df["round"].min()
    assert (df.loc[df["round"] == first_round, "driver_points_before"] == 0).all(), \
        "Round 1 should have zero standings points before the race"
    assert "points" not in feature_cols and "finish_position" not in feature_cols, \
        "A target leaked into the features!"

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_FILE, index=False)

    print(f"Rounds covered : {df['round'].nunique()}")
    print(f"Rows           : {len(df)}  (one per driver per race)")
    print(f"Features       : {len(feature_cols)}")
    print(f"Saved to       : {OUT_FILE}\n")
    print("Feature columns:")
    for c in feature_cols:
        print(f"  - {c}")
    print("\nMissing values per feature (some are expected in early rounds):")
    print(df[feature_cols].isna().sum()[lambda s: s > 0].to_string() or "  none")
    print("\nPreview:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
