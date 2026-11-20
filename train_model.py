"""
F1 Forecast - Phase 4 (version 3): Race Outcome Prediction + Extended Backtest
==============================================================================
Reads the feature table, tries several prediction setups, and tests each one
HONESTLY on races it has never seen.

What is new in version 3
------------------------
1. Bigger test: every race from round 4 onward is a test race (about 17 instead
   of 5). Each is predicted using ONLY the races before it (walk-forward).
2. More baselines: besides "everyone finishes where they started" (grid), it now
   also tries "finish in qualifying order" and "finish in championship order".
3. Tyre features: if data/processed/features_tyre.csv exists (made by
   build_tyre_features.py), every model is also run WITH the tyre features
   (names ending in _tyre), so you can see whether they help.
4. Fairer winner pick: the best setup is chosen on the FIRST half of the test
   races and scored on the SECOND half, which it never influenced.
5. Tie-breaks no longer use the real finishing order (v2 let ties in the grid
   baseline be broken by the true result, which flattered the baseline slightly).

Setups
------
  grid_baseline / quali_baseline / standings_baseline : no model, just a rule
  xgb_v1_position      : XGBoost predicts the finishing position
  xgb_vs_grid          : XGBoost predicts places gained or lost versus the grid
  xgb_vs_grid_simple   : same, fewer features, more cautious model
  ridge_vs_grid_simple : plain linear model on the same few features
  ...each also with "_tyre" if tyre features are available.

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
TYRE_FEATURES_FILE = os.path.join("data", "processed", "features_tyre.csv")
PREDICTIONS_FILE = os.path.join("data", "processed", "backtest_predictions.csv")
METRICS_FILE = os.path.join("data", "processed", "backtest_metrics.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")
INFO_FILE = os.path.join(MODEL_DIR, "best_model_info.json")

MIN_TRAIN_ROUNDS = 3     # the first test race is the one after this many rounds
MIN_RACES_TO_SPLIT = 6   # need at least this many test races to split them in two
TYRE_PREFIX = "team_deg_"

ID_COLS = ["round", "race_name", "driver_id", "constructor_id"]
TARGET_COLS = ["finish_position", "points", "podium", "points_finish", "is_dnf", "status"]

SIMPLE_FEATURES = [
    "grid", "quali_position", "quali_gap_to_pole_pct",
    "team_position_before", "team_avg_points_recent", "driver_avg_points_recent",
]

XGB_V1_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
XGB_SIMPLE_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=5.0, random_state=42)

METRICS = ["mae", "mae_finishers", "rank_corr", "winner_correct", "podium_overlap", "top10_overlap"]


def load_data():
    path = TYRE_FEATURES_FILE if os.path.exists(TYRE_FEATURES_FILE) else FEATURES_FILE
    if not os.path.exists(path):
        raise SystemExit(f"{FEATURES_FILE} not found. Run build_features.py first.")
    df = pd.read_csv(path)
    all_features = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]
    tyre = [c for c in all_features if c.startswith(TYRE_PREFIX)]
    base = [c for c in all_features if c not in tyre]
    return df, base, tyre, path


def build_setups(base, tyre):
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
        if tyre:
            setups[name + "_tyre"] = dict(model=builder, features=feats + tyre, target=target)
    return setups


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


def main():
    last_n = int(sys.argv[1]) if len(sys.argv) > 1 else None

    print("=== F1 Forecast: Phase 4 Race Outcome Model (version 3) ===\n")
    df, base, tyre, path = load_data()
    setups = build_setups(base, tyre)
    model_names = [n for n, s in setups.items() if s["model"] is not None]
    baseline_names = [n for n, s in setups.items() if s["model"] is None]

    rounds = sorted(df["round"].unique())
    test_rounds = rounds[MIN_TRAIN_ROUNDS:]
    if last_n:
        test_rounds = test_rounds[-last_n:]
    if len(test_rounds) < 1:
        raise SystemExit(f"Only {len(rounds)} rounds available: too few to test.")

    print(f"Data file: {path}")
    print(f"Rows: {len(df)} | Rounds: {len(rounds)} | Features: {len(base)} base + {len(tyre)} tyre")
    print(f"Backtest: {len(test_rounds)} test races (rounds {test_rounds[0]} to {test_rounds[-1]}), "
          f"each predicted using only earlier rounds")
    if not tyre:
        print("No tyre features found (run build_tyre_features.py to add them).")

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

    fin = results[results["round"].isin(final_rounds)].pivot(index="round", columns="setup", values="mae")
    best_final, grid_final = final_table.loc[best_name, "mae"], final_table.loc["grid_baseline", "mae"]
    wins = int((fin[best_name] < fin["grid_baseline"]).sum())
    print(f"Held-out races: MAE {best_final:.3f} vs grid baseline {grid_final:.3f}; "
          f"better than the grid in {wins} of {len(final_rounds)} races")
    for b in baseline_names:
        if b != "grid_baseline":
            print(f"   (for comparison, {b}: MAE {final_table.loc[b, 'mae']:.3f})")
    print("Winner picked correctly:",
          f"{best_name} {final_table.loc[best_name, 'winner_correct']:.0%}",
          f"| grid baseline {final_table.loc['grid_baseline', 'winner_correct']:.0%}")
    gap = grid_final - best_final
    if gap >= 0.1 and wins > len(final_rounds) / 2:
        verdict = "clearly beats the grid baseline on races it was not chosen on."
    elif gap > -0.1:
        verdict = "is about level with the grid baseline: no clear difference on the held-out races."
    else:
        verdict = "does NOT beat the grid baseline on the held-out races (an honest, normal result)."
    print("Verdict:", verdict)

    if tyre:
        print("\n--- Do the tyre features help? (average MAE over ALL test races) ---")
        overall = summarise(results, order)["mae"]
        for name in model_names:
            if not name.endswith("_tyre"):
                with_t, without_t = overall[name + "_tyre"], overall[name]
                verdict = "helps" if with_t < without_t - 0.01 else "hurts" if with_t > without_t + 0.01 else "no clear change"
                print(f"  {name:<22} without {without_t:.3f} | with {with_t:.3f} | {verdict}")

    print(f"\n--- {best_name}, race by race (all test races) ---")
    per_race = results[results["setup"] == best_name].drop(columns="setup").round(2)
    names = df.drop_duplicates("round").set_index("round")["race_name"]
    per_race.insert(1, "race", per_race["round"].map(names))
    print(per_race.to_string(index=False))

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
