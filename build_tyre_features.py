"""
F1 Forecast - Phase 3 (part 3): Tyre Features for the Race Model
================================================================
Connects the tyre degradation work (Phase 3) to the race outcome model (Phase 4).

It reads:
  - data/processed/features.csv   (from build_features.py)
  - data/processed/clean_laps.csv (from tyre_degradation.py)
  - data/raw/results_round_*.json (from ingest_2026.py; only to match FastF1
                                   driver codes such as "RUS" to team names)

and writes data/processed/features_tyre.csv: the same table plus two new columns
for every driver-race row:

  team_deg_avg_before      : how many seconds per lap the team's cars lose as their
                             tyres age (averaged over compounds, weighted by laps)
  team_deg_vs_field_before : the same, minus the whole field's rate on the same
                             compounds. Positive = the team wears tyres faster than
                             average, negative = kinder to its tyres.

THE LEAKAGE RULE
----------------
For race N, a team's degradation is measured ONLY from laps of rounds 1 .. N-1.
(tyre_degradation_by_team.csv uses the whole season, so it must NOT be merged
directly: race 5 would then be predicted using laps from races 6 onwards.)
Early rounds therefore have empty values (NaN) until enough stints have been seen.

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
NEW_COLUMNS = ["team_deg_avg_before", "team_deg_vs_field_before"]


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


def slope_rate(df):
    """Seconds lost per lap of tyre age, only where there are enough stints."""
    enough = (df["n_stints"] >= MIN_STINTS_FOR_RATE) & (df["xx"] > 0)
    return (df["xy"] / df["xx"].where(df["xx"] > 0)).where(enough)


def features_before_round(agg, before_round):
    """Team tyre features for one race, using only rounds strictly before it."""
    prior = agg[agg["round"] < before_round]
    if prior.empty:
        return pd.DataFrame(columns=["constructor_id"] + NEW_COLUMNS)

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
    return pd.DataFrame(rows, columns=["constructor_id"] + NEW_COLUMNS)


def main():
    print("=== F1 Forecast: Tyre Features for the Race Model ===\n")
    for path, hint in [(FEATURES_FILE, "build_features.py"), (CLEAN_LAPS_FILE, "tyre_degradation.py")]:
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run {hint} first.")

    features = pd.read_csv(FEATURES_FILE)
    features = features.drop(columns=[c for c in NEW_COLUMNS if c in features.columns])
    laps = pd.read_csv(CLEAN_LAPS_FILE,
                       usecols=["round", "Driver", "Stint", "Compound", "age_dev", "time_dev"])

    # Match FastF1 driver codes to Jolpica team names, race by race
    mapping = load_driver_team_map()
    laps["constructor_id"] = [mapping.get((r, d)) for r, d in zip(laps["round"], laps["Driver"])]
    unmapped = int(laps["constructor_id"].isna().sum())
    laps = laps.dropna(subset=["constructor_id"])
    print(f"Clean laps used : {len(laps)}  (could not match to a team: {unmapped})")

    laps["xy"] = laps["age_dev"] * laps["time_dev"]
    laps["xx"] = laps["age_dev"] ** 2
    stints = (laps.drop_duplicates(["round", "constructor_id", "Driver", "Stint"])
              .groupby(["round", "constructor_id", "Compound"]).size().rename("n_stints"))
    agg = (laps.groupby(["round", "constructor_id", "Compound"])
           .agg(xy=("xy", "sum"), xx=("xx", "sum"), n_laps=("xy", "size"))
           .join(stints).reset_index())

    per_round = []
    for r in sorted(features["round"].unique()):
        feats = features_before_round(agg, r)
        feats.insert(0, "round", r)
        per_round.append(feats)
    tyre = pd.concat(per_round, ignore_index=True)

    out = features.merge(tyre, on=["round", "constructor_id"], how="left")
    out.to_csv(OUT_FILE, index=False)

    filled = out["team_deg_avg_before"].notna()
    print(f"Rows            : {len(out)}")
    print(f"Rows with tyre features : {int(filled.sum())} ({filled.mean():.0%})")
    if filled.any():
        print(f"First round with values : {int(out.loc[filled, 'round'].min())}")
    print(f"Saved to        : {OUT_FILE}\n")
    latest = out["round"].max()
    print(f"Team tyre features going into round {latest}:")
    print(out[out["round"] == latest].drop_duplicates("constructor_id")
          [["constructor_id"] + NEW_COLUMNS].sort_values("team_deg_vs_field_before").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
