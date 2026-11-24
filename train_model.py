"""
F1 Forecast - Phase 4 (version 4): Race Outcome Prediction + Extended Backtest
==============================================================================
Reads the feature table, tries several prediction setups, and tests each one
HONESTLY on races it has never seen.

What is new in version 4
------------------------
1. No more near-duplicate columns. "grid" and "quali_position" are almost the same
   number, which made the linear model's coefficients meaningless. The simple setups
   now use quali_position plus grid_penalty (= grid minus quali_position, i.e. how
   many places a driver lost to a penalty).
2. Optional feature groups from build_tyre_features.py are each tested with and
   without: the TYRE group (team_deg_*) and the race PACE group (team_pace_*).
   Setups with a group get a suffix such as _tyre or _pace.
3. The verdict now compares the chosen model with the STRONGEST simple rule
   (usually "finish in qualifying order"), not only with the grid.

Kept from version 3: every race from round 4 onward is a test race; each is
predicted using only the races before it; the best setup is chosen on the first
half of the test races and scored on the second half.

Simple rules (no model)
-----------------------
  grid_baseline / quali_baseline / standings_baseline

Models
------
  xgb_v1_position      : XGBoost predicts the finishing position
  xgb_vs_grid          : XGBoost predicts places gained or lost versus the grid
  xgb_vs_grid_simple   : same, fewer features, more cautious model
  ridge_vs_grid_simple : plain linear model on the same few features

Metrics (per race, then averaged)
---------------------------------
  mae              : average gap between predicted and actual position (lower = better)
  mae_finishers    : same, but only for drivers who finished (DNFs are mostly luck)
  rank_corr        : how well predicted order matches real order (1.0 = perfect)
  winner_correct   : share of races where the predicted winner actually won
  podium_overlap   : share of the real top 3 found in the predicted top 3
  top10_overlap    : share of the real top 10 found in the predicted top 10

Outputs
-------
  data/processed/backtest_predictions.csv : every test prediction next to the real result
  data/processed/backtest_metrics.csv     : every setup's metrics for every test race
  models/best_model.joblib                : best setup, retrained on ALL rounds
  models/best_model_info.json             : which setup won and which features it uses

Run this with your virtual environment active:
    python train_model.py          # test on every race from round 4 onward
    python train_model.py 5        # test on the last 5 rounds only

Project hard rule: only 2026 data is used for performance features.
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURES_FILE = os.path.join("data", "processed", "features.csv")
LAP_FEATURES_FILE = os.path.join("data", "processed", "features_tyre.csv")
PREDICTIONS_FILE = os.path.join("data", "processed", "backtest_predictions.csv")
METRICS_FILE = os.path.join("data", "processed", "backtest_metrics.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")
INFO_FILE = os.path.join(MODEL_DIR, "best_model_info.json")

MIN_TRAIN_ROUNDS = 3     # the first test race is the one after this many rounds
MIN_RACES_TO_SPLIT = 6   # need at least this many test races to split them in two

# Optional feature groups: name used in setup names -> column prefix
FEATURE_GROUPS = {"tyre": "team_deg_", "pace": "team_pace_"}

ID_COLS = ["round", "race_name", "driver_id", "constructor_id"]
TARGET_COLS = ["finish_position", "points", "podium", "points_finish", "is_dnf", "status"]

SIMPLE_FEATURES = [
    "quali_position", "grid_penalty", "quali_gap_to_pole_pct",
    "team_position_before", "team_avg_points_recent", "driver_avg_points_recent",
]

XGB_V1_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
XGB_SIMPLE_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=5.0, random_state=42)

METRICS = ["mae", "mae_finishers", "rank_corr", "winner_correct", "podium_overlap", "top10_overlap"]


def load_data():
    path = LAP_FEATURES_FILE if os.path.exists(LAP_FEATURES_FILE) else FEATURES_FILE
    if not os.path.exists(path):
        raise SystemExit(f"{FEATURES_FILE} not found. Run build_features.py first.")
    df = pd.read_csv(path)
    if {"grid", "quali_position"} <= set(df.columns):
        df["grid_penalty"] = df["grid"] - df["quali_position"]
    all_features = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]
    groups = {name: [c for c in all_features if c.startswith(prefix)]
              for name, prefix in FEATURE_GROUPS.items()}
    groups = {name: cols for name, cols in groups.items() if cols}
    grouped = {c for cols in groups.values() for c in cols}
    base = [c for c in all_features if c not in grouped]
    return df, base, groups, path


def build_setups(base, groups):
    """
    Each setup: model builder (None = rule-based baseline), features, target
    ('position' or 'delta' = change versus grid), and for baselines the column to sort by.
    """
    simple = [c for c in SIMPLE_FEATURES if c in base]
    setups = {
        "grid_baseline": dict(model=None, features=[], target="delta", score_col="grid"),
        "quali_baseline": dict(model=None, features=[], target="delta", score_col="quali_position"),
        "standings_baseline": dict(model=None, features=[], target="delta", score_col="driver_position_before"),
    }
    models = {
        "xgb_v1_position": (lambda: XGBRegressor(**XGB_V1_PARAMS), base, "position"),
        "xgb_vs_grid": (lambda: XGBRegressor(**XGB_V1_PARAMS), base, "delta"),
        "xgb_vs_grid_simple": (lambda: XGBRegressor(**XGB_SIMPLE_PARAMS), simple, "delta"),
        "ridge_vs_grid_simple": (
            lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
            simple, "delta"),
    }
    for name, (builder, feats, target) in models.items():
        setups[name] = dict(model=builder, features=feats, target=target)
        for group, cols in groups.items():
            setups[f"{name}_{group}"] = dict(model=builder, features=feats + cols, target=target)
    return setups, list(models)


def training_target(df, target):
    return df["finish_position"] - df["grid"] if target == "delta" else df["finish_position"]


def predict_scores(setup, train, test):
    """Returns one number per test driver; LOWER means 'finishes earlier'."""
    if setup["model"] is None:
        return test[setup["score_col"]].fillna(99).values
    model = setup["model"]()
    model.fit(train[setup["features"]], training_target(train, setup["target"]))
    raw = model.predict(test[setup["features"]])
    return test["grid"].values + raw if setup["target"] == "delta" else raw


def predicted_ranks(race, score_col):
    """1 = predicted to finish first. Ties are broken by grid slot, then name, never by the result."""
    order = race.sort_values([score_col, "grid", "driver_id"], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(order) + 1), index=order.index)
    return ranks.reindex(race.index)


def score_race(race, score_col):
    pred_rank = predicted_ranks(race, score_col)
    actual = race["finish_position"]
    finished = race["is_dnf"] == 0
    pred_top3 = set(race.loc[pred_rank <= 3, "driver_id"])
    real_top3 = set(race.loc[actual <= 3, "driver_id"])
    pred_top10 = set(race.loc[pred_rank <= 10, "driver_id"])
    real_top10 = set(race.loc[actual <= 10, "driver_id"])
    predicted_winner = race.loc[pred_rank.idxmin(), "finish_position"]
    return {
        "mae": (pred_rank - actual).abs().mean(),
        "mae_finishers": (pred_rank[finished] - actual[finished]).abs().mean(),
        "rank_corr": pred_rank.corr(actual),
        "winner_correct": int(predicted_winner == 1),
        "podium_overlap": len(pred_top3 & real_top3) / 3,
        "top10_overlap": len(pred_top10 & real_top10) / 10,
    }, pred_rank


def run_backtest(df, setups, test_rounds):
    rows, frames = [], []
    for r in test_rounds:
        train = df[df["round"] < r]
        test = df[df["round"] == r].copy()
        for name, setup in setups.items():
            test["predicted_score"] = predict_scores(setup, train, test)
            scores, ranks = score_race(test, "predicted_score")
            test["predicted_rank"] = ranks
            rows.append({"setup": name, "round": r, **scores})
            frames.append(test.assign(setup=name)[
                ["setup", "round", "race_name", "driver_id", "constructor_id", "grid",
                 "predicted_score", "predicted_rank", "finish_position"]])
    return pd.DataFrame(rows), pd.concat(frames)


def summarise(results, order):
    return results.groupby("setup")[METRICS].mean().reindex(order)


def compare_text(gap, wins, n_races):
    """gap = other's MAE minus the chosen model's MAE (positive = chosen model is better)."""
    if gap >= 0.1 and wins > n_races / 2:
        return "clearly better"
    if gap > -0.1:
        return "about level"
    return "worse"


def main():
    last_n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=== F1 Forecast: Phase 4 Race Outcome Model (version 4) ===\n")
    df, base, groups, path = load_data()
    setups, base_model_names = build_setups(base, groups)
    model_names = [n for n, s in setups.items() if s["model"] is not None]
    baseline_names = [n for n, s in setups.items() if s["model"] is None]

    rounds = sorted(df["round"].unique())
    test_rounds = rounds[MIN_TRAIN_ROUNDS:]
    if last_n:
        test_rounds = test_rounds[-last_n:]
    if len(test_rounds) < 1:
        raise SystemExit(f"Only {len(rounds)} rounds available: too few to test.")

    group_text = " + ".join(f"{len(cols)} {name}" for name, cols in groups.items()) or "no extra groups"
    print(f"Data file: {path}")
    print(f"Rows: {len(df)} | Rounds: {len(rounds)} | Features: {len(base)} base + {group_text}")
    print(f"Backtest: {len(test_rounds)} test races (rounds {test_rounds[0]} to {test_rounds[-1]}), "
          f"each predicted using only earlier rounds")
    if not groups:
        print("No tyre or pace features found (run build_tyre_features.py to add them).")

    results, predictions = run_backtest(df, setups, test_rounds)
    order = list(setups.keys())

    print("\n--- All test races, average per setup ---")
    print(summarise(results, order).round(3).to_string())
    print("\nmae and mae_finishers: lower is better. All other columns: higher is better.")

    # Choose on the first half of the test races, report on the second half
    if len(test_rounds) >= MIN_RACES_TO_SPLIT:
        half = len(test_rounds) // 2
        select_rounds, final_rounds = test_rounds[:half], test_rounds[half:]
        honest = True
    else:
        select_rounds = final_rounds = test_rounds
        honest = False
    sel = summarise(results[results["round"].isin(select_rounds)], order)
    best_name = sel.loc[model_names, "mae"].idxmin()

    print(f"\n--- Picking the best model on rounds {select_rounds[0]} to {select_rounds[-1]}, "
          f"then scoring it on rounds {final_rounds[0]} to {final_rounds[-1]} ---")
    if not honest:
        print("(Too few test races to split: this score is optimistic.)")
    final_table = summarise(results[results["round"].isin(final_rounds)], order)
    print(final_table.round(3).to_string())
    print(f"\nChosen setup: {best_name}")

    per_race = results[results["round"].isin(final_rounds)].pivot(index="round", columns="setup", values="mae")
    strongest = final_table.loc[baseline_names, "mae"].idxmin()
    best_final = final_table.loc[best_name, "mae"]
    n_final = len(final_rounds)
    print(f"Held-out races: chosen model MAE {best_final:.3f}")
    for label, other in [("grid baseline", "grid_baseline"), (f"strongest rule ({strongest})", strongest)]:
        other_mae = final_table.loc[other, "mae"]
        wins = int((per_race[best_name] < per_race[other]).sum())
        verdict = compare_text(other_mae - best_final, wins, n_final)
        print(f"  vs {label}: {other_mae:.3f} -> {verdict} (better in {wins} of {n_final} races)")
    print("Winner picked correctly:",
          f"{best_name} {final_table.loc[best_name, 'winner_correct']:.0%}",
          f"| grid baseline {final_table.loc['grid_baseline', 'winner_correct']:.0%}",
          f"| {strongest} {final_table.loc[strongest, 'winner_correct']:.0%}")

    for group in groups:
        print(f"\n--- Do the {group} features help? (average MAE over ALL test races) ---")
        overall = summarise(results, order)["mae"]
        for name in base_model_names:
            with_g, without_g = overall[f"{name}_{group}"], overall[name]
            verdict = ("helps" if with_g < without_g - 0.01
                       else "hurts" if with_g > without_g + 0.01 else "no clear change")
            print(f"  {name:<22} without {without_g:.3f} | with {with_g:.3f} | {verdict}")

    print(f"\n--- {best_name}, race by race (all test races) ---")
    per_race_table = results[results["setup"] == best_name].drop(columns="setup").round(2)
    names = df.drop_duplicates("round").set_index("round")["race_name"]
    per_race_table.insert(1, "race", per_race_table["round"].map(names))
    print(per_race_table.to_string(index=False))

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    results.to_csv(METRICS_FILE, index=False)
    print(f"\nSaved test predictions to {PREDICTIONS_FILE}")
    print(f"Saved per-race metrics to {METRICS_FILE}")

    best = setups[best_name]
    final_model = best["model"]()
    final_model.fit(df[best["features"]], training_target(df, best["target"]))
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    with open(INFO_FILE, "w", encoding="utf-8") as f:
        json.dump({"setup": best_name, "target": best["target"], "features": best["features"],
                   "trained_on_rounds": int(df["round"].nunique())}, f, indent=2)
    print(f"Saved best model ({best_name}, trained on all {df['round'].nunique()} rounds) to {MODEL_FILE}")

    if hasattr(final_model, "feature_importances_"):
        print("\nMost influential features (best model):")
        imp = pd.Series(final_model.feature_importances_, index=best["features"])
        print(imp.sort_values(ascending=False).head(8).round(3).to_string())
    elif hasattr(final_model, "named_steps"):
        stats = final_model[0].statistics_
        kept = [f for f, s in zip(best["features"], stats) if not np.isnan(s)]
        coefs = pd.Series(final_model[-1].coef_, index=kept)
        print("\nLargest standardised coefficients (best model; negative = finishes earlier than the grid slot):")
        print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
