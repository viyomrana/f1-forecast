"""
F1 Forecast - Phase 3 (part 2): Tyre Degradation Model
======================================================
Reads the lap files saved by ingest_laps_2026.py (data/raw/laps/) and measures
how much lap time each tyre compound loses per lap of tyre age.

Steps
-----
1. CLEAN the laps: keep only normal green-flag racing laps on dry tyres
   (drop lap 1, pit in/out laps, safety car / yellow laps, deleted laps,
   wet-weather tyres, and unusually slow laps within a stint).
2. FUEL-CORRECT: cars get lighter and quicker as fuel burns, which hides tyre
   wear. We add a fixed amount back per lap so that effect is removed.
3. FIT: inside each stint (one driver, one set of tyres) we compare how lap
   time changes as tyre age grows. Measuring change WITHIN a stint cancels out
   driver skill, car speed and track differences. The pooled slope is the
   degradation rate in seconds lost per lap of tyre age.

Outputs (data/processed/)
-------------------------
  clean_laps.csv                 : the cleaned, fuel-corrected laps
  tyre_degradation_by_race.csv   : degradation rate per race and compound
  tyre_degradation_by_team.csv   : degradation rate per team and compound (whole season)

IMPORTANT ASSUMPTIONS (state these in your report/demo)
-------------------------------------------------------
- FUEL_EFFECT_S_PER_LAP is an ASSUMED constant, not measured from the data.
  It cannot be separated from tyre wear using these columns alone, because
  tyre age and lap number rise together inside a stint. Change the value at
  the top of the file and re-run to see how sensitive the results are.
- Track evolution (the surface getting faster as rubber is laid down) also
  hides some wear, so true degradation is probably slightly higher than shown.
- Rates are only reported when there is enough data (see MIN_* settings).

Run this with your virtual environment active:
    python tyre_degradation.py

Project hard rule: only 2026 data is used for performance features.
"""

import os

import numpy as np
import pandas as pd

LAPS_DIR = os.path.join("data", "raw", "laps")
OUT_DIR = os.path.join("data", "processed")

FUEL_EFFECT_S_PER_LAP = 0.03   # ASSUMPTION: seconds a car gains per lap as fuel burns off
SLOW_LAP_FACTOR = 1.07         # drop laps slower than 107% of the stint's median lap
MIN_LAPS_PER_STINT = 5         # a stint needs at least this many clean laps to be used
MIN_STINTS_FOR_RATE = 3        # a rate is only reported if built from at least this many stints
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def load_laps():
    if not os.path.isdir(LAPS_DIR):
        raise SystemExit(f"{LAPS_DIR} not found. Run ingest_laps_2026.py first.")
    files = sorted(f for f in os.listdir(LAPS_DIR) if f.startswith("laps_round_"))
    if not files:
        raise SystemExit(f"No lap files in {LAPS_DIR}. Run ingest_laps_2026.py first.")
    frames = [
        pd.read_csv(os.path.join(LAPS_DIR, f), dtype={"TrackStatus": str}) for f in files
    ]
    return pd.concat(frames, ignore_index=True)


def clean_laps(laps):
    """Returns (clean_laps, report) where report counts laps lost at each step."""
    report = [("All laps", len(laps))]
    df = laps.copy()

    df = df[df["LapTime_s"].notna()]
    report.append(("Has a lap time", len(df)))

    df = df[df["LapNumber"] > 1]
    report.append(("Not lap 1 (standing start)", len(df)))

    df = df[(df["is_pit_in_lap"] == 0) & (df["is_pit_out_lap"] == 0)]
    report.append(("Not a pit in/out lap", len(df)))

    # TrackStatus "1" means green flag for the whole lap (anything else = yellow, SC, VSC, red)
    df = df[df["TrackStatus"] == "1"]
    report.append(("Green-flag lap", len(df)))

    deleted = df["Deleted"].fillna(False).astype(bool)
    df = df[~deleted]
    report.append(("Not deleted by stewards", len(df)))

    df = df[df["Compound"].isin(DRY_COMPOUNDS) & df["TyreLife"].notna()]
    report.append(("Dry tyre with known age", len(df)))

    # Drop unusually slow laps (traffic, mistakes) relative to the stint's own median
    keys = ["round", "Driver", "Stint"]
    stint_median = df.groupby(keys)["LapTime_s"].transform("median")
    df = df[df["LapTime_s"] <= stint_median * SLOW_LAP_FACTOR]
    report.append((f"Not >{int((SLOW_LAP_FACTOR - 1) * 100)}% slower than stint median", len(df)))

    # Keep only stints with enough laps to measure a trend
    stint_size = df.groupby(keys)["LapTime_s"].transform("size")
    df = df[stint_size >= MIN_LAPS_PER_STINT].copy()
    report.append((f"Stint has >= {MIN_LAPS_PER_STINT} clean laps", len(df)))

    # Fuel correction: add back the time gained from burning fuel
    df["fuel_corrected_s"] = df["LapTime_s"] + FUEL_EFFECT_S_PER_LAP * df["LapNumber"]
    return df, report


def add_within_stint_deviations(df):
    """
    For each stint, subtract the stint average from tyre age and lap time.
    What remains is 'how did lap time change as the tyres aged', free of
    driver, car and track differences.
    """
    keys = ["round", "Driver", "Stint"]
    df = df.copy()
    df["age_dev"] = df["TyreLife"] - df.groupby(keys)["TyreLife"].transform("mean")
    df["time_dev"] = df["fuel_corrected_s"] - df.groupby(keys)["fuel_corrected_s"].transform("mean")
    return df


def degradation_rates(df, group_cols):
    """
    Pooled slope through the origin: seconds lost per extra lap of tyre age.
    Positive = tyres get slower as they age.
    """
    rows = []
    for key, g in df.groupby(group_cols):
        n_stints = g.groupby(["round", "Driver", "Stint"]).ngroups
        if n_stints < MIN_STINTS_FOR_RATE:
            continue
        denom = (g["age_dev"] ** 2).sum()
        if denom == 0:
            continue
        slope = (g["age_dev"] * g["time_dev"]).sum() / denom
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(group_cols, key)),
                     "deg_s_per_lap": round(slope, 4),
                     "n_stints": n_stints,
                     "n_laps": len(g)})
    return pd.DataFrame(rows)


def main():
    print("=== F1 Forecast: Phase 3 Tyre Degradation ===\n")
    laps = load_laps()
    print(f"Loaded {len(laps)} laps from {laps['round'].nunique()} rounds")
    print(f"Assumed fuel effect: {FUEL_EFFECT_S_PER_LAP} s/lap (change at top of file to test sensitivity)\n")

    df, report = clean_laps(laps)
    print("Cleaning steps (laps remaining):")
    for label, n in report:
        print(f"  {label:<45} {n:>6}")

    df = add_within_stint_deviations(df)

    by_race = degradation_rates(df, ["round", "race_name", "Compound"])
    by_team = degradation_rates(df, ["Team", "Compound"])
    season = degradation_rates(df, ["Compound"])

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "clean_laps.csv"), index=False)
    by_race.to_csv(os.path.join(OUT_DIR, "tyre_degradation_by_race.csv"), index=False)
    by_team.to_csv(os.path.join(OUT_DIR, "tyre_degradation_by_team.csv"), index=False)

    print("\n--- Whole season, by compound (seconds lost per lap of tyre age) ---")
    print(season.to_string(index=False))

    print("\n--- By team and compound ---")
    print(by_team.sort_values(["Compound", "deg_s_per_lap"]).to_string(index=False))

    print(f"\nSaved to {OUT_DIR}: clean_laps.csv, tyre_degradation_by_race.csv, "
          f"tyre_degradation_by_team.csv")
    print("\nHow to read this: 0.05 means the car loses about 0.05 s every extra lap on that "
          "tyre.\nValues near zero or negative usually mean the fuel assumption or track "
          "evolution is masking wear.")


if __name__ == "__main__":
    main()
