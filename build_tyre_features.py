"""
F1 Forecast - Phase 3 (part 3): Lap-Based Features for the Race Model
=====================================================================
Connects the lap data (Phase 3) to the race outcome model (Phase 4).

It reads:
  - data/processed/features.csv   (from build_features.py)
  - data/processed/clean_laps.csv (from tyre_degradation.py)
  - data/raw/results_round_*.json (from ingest_2026.py; only to match FastF1
                                   driver codes such as "RUS" to team names)

and writes data/processed/features_tyre.csv: the same table plus four new columns
for every driver-race row, in two groups.

TYRE group (how fast the team's tyres wear)
  team_deg_avg_before        seconds lost per lap of tyre age (averaged over compounds)
  team_deg_vs_field_before   the same minus the whole field's rate; positive = harder
                             on its tyres than average

PACE group (how quick the team is in the RACE compared with qualifying)
  team_pace_gap_before       average % gap to the quickest team's race pace
                             (same compound, fuel-corrected laps), last 3 races
  team_pace_vs_quali_before  that race gap minus the team's qualifying gap, last 3
                             races. Positive = slower in the race than qualifying
                             suggests, negative = quicker in the race than in qualifying.
                             This is information that qualifying order alone does not hold.

THE LEAKAGE RULE
----------------
For race N, every value is built ONLY from rounds 1 .. N-1 (laps and qualifying of
earlier races). The tyre part re-measures degradation from earlier laps for every
race, and the pace part is shifted by one race. tyre_degradation_by_team.csv uses
the whole season, so it must NOT be merged directly.
Early rounds therefore have empty values (NaN) until enough data has been seen.

Run order:
    python build_features.py
    python ingest_laps_2026.py
    python tyre_degradation.py
    python build_tyre_features.py      <- this script
    python train_model.py

Project hard rule: only 2026 data is used for performance features.
"""

import json
import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
FEATURES_FILE = os.path.join(PROCESSED_DIR, "features.csv")
CLEAN_LAPS_FILE = os.path.join(PROCESSED_DIR, "clean_laps.csv")
OUT_FILE = os.path.join(PROCESSED_DIR, "features_tyre.csv")

MIN_STINTS_FOR_RATE = 3  # same rule as tyre_degradation.py: need this many stints for a rate
MIN_LAPS_PACE = 10       # a team needs this many clean laps on a compound in a race to be compared
MIN_TEAMS_PACE = 3       # and at least this many teams must have run that compound
PACE_WINDOW = 3          # race pace = average of the previous 3 races

TYRE_COLUMNS = ["team_deg_avg_before", "team_deg_vs_field_before"]
PACE_COLUMNS = ["team_pace_gap_before", "team_pace_vs_quali_before"]
NEW_COLUMNS = TYRE_COLUMNS + PACE_COLUMNS


def load_driver_team_map():
    """Returns {(round, 'RUS'): 'mercedes'} built from the raw results files."""
    mapping = {}
    for filename in os.listdir(RAW_DIR):
        if not filename.startswith("results_round_"):
            continue
        round_number = int(filename.split("_round_")[1].split(".")[0])
        with open(os.path.join(RAW_DIR, filename), encoding="utf-8") as f:
            data = json.load(f)
        for r in data["MRData"]["RaceTable"]["Races"][0]["Results"]:
            code = r["Driver"].get("code")
            if code:
                mapping[(round_number, code)] = r["Constructor"]["constructorId"]
    return mapping


# ---------------------------------------------------------------------------
# Tyre group
# ---------------------------------------------------------------------------
def slope_rate(df):
    """Seconds lost per lap of tyre age, only where there are enough stints."""
    enough = (df["n_stints"] >= MIN_STINTS_FOR_RATE) & (df["xx"] > 0)
    return (df["xy"] / df["xx"].where(df["xx"] > 0)).where(enough)


def tyre_features_before_round(agg, before_round):
    """Team tyre features for one race, using only rounds strictly before it."""
    prior = agg[agg["round"] < before_round]
    if prior.empty:
        return pd.DataFrame(columns=["constructor_id"] + TYRE_COLUMNS)

    team = prior.groupby(["constructor_id", "Compound"])[["xy", "xx", "n_laps", "n_stints"]].sum().reset_index()
    field = prior.groupby("Compound")[["xy", "xx", "n_stints"]].sum()
    team["rate"] = slope_rate(team)
    field["field_rate"] = slope_rate(field)
    team = team.join(field["field_rate"], on="Compound")
    team["rel"] = team["rate"] - team["field_rate"]

    rows = []
    for constructor, g in team.dropna(subset=["rate"]).groupby("constructor_id"):
        avg = (g["rate"] * g["n_laps"]).sum() / g["n_laps"].sum()
        g_rel = g.dropna(subset=["rel"])
        vs_field = ((g_rel["rel"] * g_rel["n_laps"]).sum() / g_rel["n_laps"].sum()
                    if len(g_rel) else np.nan)
        rows.append({"constructor_id": constructor,
                     "team_deg_avg_before": avg,
                     "team_deg_vs_field_before": vs_field})
    return pd.DataFrame(rows, columns=["constructor_id"] + TYRE_COLUMNS)


def build_tyre_group(laps, rounds):
    laps = laps.copy()
    laps["xy"] = laps["age_dev"] * laps["time_dev"]
    laps["xx"] = laps["age_dev"] ** 2
    stints = (laps.drop_duplicates(["round", "constructor_id", "Driver", "Stint"])
              .groupby(["round", "constructor_id", "Compound"]).size().rename("n_stints"))
    agg = (laps.groupby(["round", "constructor_id", "Compound"])
           .agg(xy=("xy", "sum"), xx=("xx", "sum"), n_laps=("xy", "size"))
           .join(stints).reset_index())
    per_round = []
    for r in rounds:
        feats = tyre_features_before_round(agg, r)
        if not feats.empty:
            feats.insert(0, "round", r)
            per_round.append(feats)
    if not per_round:
        return pd.DataFrame(columns=["round", "constructor_id"] + TYRE_COLUMNS)
    return pd.concat(per_round, ignore_index=True)


# ---------------------------------------------------------------------------
# Pace group
# ---------------------------------------------------------------------------
def race_gap_by_team_round(laps):
    """
    For each race: how many % slower each team's typical fuel-corrected clean lap is
    than the quickest team on the SAME compound (so tyre choice does not distort it),
    averaged over compounds, weighted by number of laps.
    """
    rows = []
    for round_number, race in laps.groupby("round"):
        parts = []
        for _, g in race.groupby("Compound"):
            stats = g.groupby("constructor_id")["fuel_corrected_s"].agg(median="median", n="size")
            stats = stats[stats["n"] >= MIN_LAPS_PACE]
            if len(stats) < MIN_TEAMS_PACE:
                continue
            best = stats["median"].min()
            stats["gap"] = (stats["median"] - best) / best * 100
            parts.append(stats.reset_index())
        if not parts:
            continue
        both = pd.concat(parts)
        for team, t in both.groupby("constructor_id"):
            rows.append({"round": round_number, "constructor_id": team,
                         "race_gap_pct": np.average(t["gap"], weights=t["n"])})
    return pd.DataFrame(rows, columns=["round", "constructor_id", "race_gap_pct"])


def build_pace_group(features, laps):
    race_gap = race_gap_by_team_round(laps)

    # Team qualifying gap per round, relative to the best team that round
    quali = (features.groupby(["round", "constructor_id"])["quali_gap_to_pole_pct"].mean()
             .rename("quali_gap").reset_index())
    quali["quali_gap"] = quali["quali_gap"] - quali.groupby("round")["quali_gap"].transform("min")

    team_round = quali.merge(race_gap, on=["round", "constructor_id"], how="left")
    team_round["race_vs_quali"] = team_round["race_gap_pct"] - team_round["quali_gap"]
    team_round = team_round.sort_values(["constructor_id", "round"])

    g = team_round.groupby("constructor_id")
    for source, target in [("race_gap_pct", "team_pace_gap_before"),
                           ("race_vs_quali", "team_pace_vs_quali_before")]:
        # shift(1): the value for race N only uses races before N
        team_round[target] = g[source].transform(
            lambda s: s.shift(1).rolling(PACE_WINDOW, min_periods=1).mean())
    return team_round[["round", "constructor_id"] + PACE_COLUMNS]


# ---------------------------------------------------------------------------
def main():
    print("=== F1 Forecast: Lap-Based Features for the Race Model ===\n")
    for path, hint in [(FEATURES_FILE, "build_features.py"), (CLEAN_LAPS_FILE, "tyre_degradation.py")]:
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run {hint} first.")

    features = pd.read_csv(FEATURES_FILE)
    features = features.drop(columns=[c for c in NEW_COLUMNS if c in features.columns])
    laps = pd.read_csv(CLEAN_LAPS_FILE, usecols=[
        "round", "Driver", "Stint", "Compound", "age_dev", "time_dev", "fuel_corrected_s"])

    # Match FastF1 driver codes to Jolpica team names, race by race
    mapping = load_driver_team_map()
    laps["constructor_id"] = [mapping.get((r, d)) for r, d in zip(laps["round"], laps["Driver"])]
    unmapped = int(laps["constructor_id"].isna().sum())
    laps = laps.dropna(subset=["constructor_id"])
    print(f"Clean laps used : {len(laps)}  (could not match to a team: {unmapped})")

    rounds = sorted(features["round"].unique())
    tyre = build_tyre_group(laps, rounds)
    pace = build_pace_group(features, laps)

    out = (features.merge(tyre, on=["round", "constructor_id"], how="left")
           .merge(pace, on=["round", "constructor_id"], how="left"))
    out.to_csv(OUT_FILE, index=False)

    print(f"Rows            : {len(out)}")
    for label, col in [("tyre", "team_deg_avg_before"), ("pace", "team_pace_gap_before")]:
        filled = out[col].notna()
        first = int(out.loc[filled, "round"].min()) if filled.any() else None
        print(f"Rows with {label} features : {int(filled.sum())} ({filled.mean():.0%}), first round with values: {first}")
    print(f"Saved to        : {OUT_FILE}\n")

    latest = out["round"].max()
    print(f"Team features going into round {latest}:")
    table = (out[out["round"] == latest].drop_duplicates("constructor_id")
             [["constructor_id"] + NEW_COLUMNS].sort_values("team_pace_gap_before"))
    print(table.round(4).to_string(index=False))
    print("\nteam_pace_gap_before: % slower than the quickest team in recent races (0 = quickest).")
    print("team_pace_vs_quali_before: positive = slower in races than qualifying suggests.")


if __name__ == "__main__":
    main()
