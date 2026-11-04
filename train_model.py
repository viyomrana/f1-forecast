"""
F1 Forecast - Phase 4 (version 2): Race Outcome Prediction + Backtest
=====================================================================
Reads data/processed/features.csv (built by build_features.py), tries several
prediction setups, and tests each one HONESTLY on races it has never seen.

What is new in version 2
------------------------
Version 1 tried to predict the finishing position directly and did not beat the
simple guess "everyone finishes where they started". Version 2 compares five
setups in one run so you can see which idea works best:

  grid_baseline       : no model. Everyone finishes where they started.
  xgb_v1_position     : version 1 (XGBoost, all features, predicts finishing position)
  xgb_vs_grid         : XGBoost predicts how many places a driver GAINS or LOSES
                        versus the grid slot (starts from the strong baseline)
  xgb_vs_grid_simple  : same, but fewer features and a simpler, more cautious model
  ridge_vs_grid_simple: a plain linear model on the same few features
                        (very stable when there is little data)

How the honest test works (walk-forward backtest)
-------------------------------------------------
For each of the last N rounds (default 5), every setup is trained ONLY on rounds
BEFORE that round, then predicts that round. Predicted order is compared with
the real result. A random split would let the model learn from races that
happen AFTER the ones it predicts.

Metrics (per race, then averaged)
---------------------------------
  MAE (positions)  : average gap between predicted and actual position (lower = better)
  Rank correlation : how well predicted order matches real order (1.0 = perfect)
  Winner correct   : share of races where the predicted winner actually won
  Podium overlap   : share of the real top 3 found in the predicted top 3
  Top-10 overlap   : share of the real top 10 found in the predicted top 10

Outputs
-------
  data/processed/backtest_predictions.csv : every test prediction next to the real result
  models/best_model.joblib                : best setup, retrained on ALL rounds
  models/best_model_info.json             : which setup won and which features it uses

Note: the "best" setup is picked using the same last-N races it is scored on, so
its score is slightly optimistic. Say so in your report.

Run this with your virtual environment active:
    pip install xgboost scikit-learn
    python train_model.py          # backtest on the last 5 rounds
    python train_model.py 3        # backtest on the last 3 rounds

Project hard rule: only 2026 data is used for performance features.
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

FEATURES_FILE = os.path.join("data", "processed", "features.csv")
PREDICTIONS_FILE = os.path.join("data", "processed", "backtest_predictions.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "best_model.joblib")
INFO_FILE = os.path.join(MODEL_DIR, "best_model_info.json")

DEFAULT_TEST_ROUNDS = 5
MIN_TRAIN_ROUNDS = 3

ID_COLS = ["round", "race_name", "driver_id", "constructor_id"]
TARGET_COLS = ["finish_position", "points", "podium", "points_finish", "is_dnf", "status"]

# The few features most likely to matter (used by the "simple" setups)
SIMPLE_FEATURES = [
    "grid", "quali_position", "quali_gap_to_pole_pct",
    "team_position_before", "team_avg_points_recent", "driver_avg_points_recent",
]

XGB_V1_PARAMS = dict(n_estimators=200, max_depth=3, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42)
XGB_SIMPLE_PARAMS = dict(n_estimators=150, max_depth=2, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         reg_lambda=5.0, random_state=42)

METRICS = ["mae", "rank_corr", "winner_correct", "podium_overlap", "top10_overlap"]


def load_data():
    if not os.path.exists(FEATURES_FILE):
        raise SystemExit(f"{FEATURES_FILE} not found. Run build_features.py first.")
    df = pd.read_csv(FEATURES_FILE)
    all_features = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]
    return df, all_features


def build_setups(all_features):
    """
    Each setup: how to build the model, which features it sees, and whether it
    predicts the finishing position ('position') or the change versus the grid ('delta').
    A setup with model=None is the no-model baseline.
    """
    simple = [c for c in SIMPLE_FEATURES if c in all_features]
    return {
        "grid_baseline": dict(model=None, features=[], target="delta"),
        "xgb_v1_position": dict(
            model=lambda: XGBRegressor(**XGB_V1_PARAMS), features=all_features, target="position"),
        "xgb_vs_grid": dict(
            model=lambda: XGBRegressor(**XGB_V1_PARAMS), features=all_features, target="delta"),
        "xgb_vs_grid_simple": dict(
            model=lambda: XGBRegressor(**XGB_SIMPLE_PARAMS), features=simple, target="delta"),
        "ridge_vs_grid_simple": dict(
            model=lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        Ridge(alpha=10.0)),
            features=simple, target="delta"),
    }


def training_target(df, target):
    return df["finish_position"] - df["grid"] if target == "delta" else df["finish_position"]


def predict_scores(setup, train, test):
    """Returns one number per test driver; LOWER means 'finishes earlier'."""
    if setup["model"] is None:
        return test["grid"].values
    model = setup["model"]()
    model.fit(train[setup["features"]], training_target(train, setup["target"]))
    raw = model.predict(test[setup["features"]])
    return test["grid"].values + raw if setup["target"] == "delta" else raw


def score_race(race, score_col):
    pred_rank = race[score_col].rank(method="first")
    actual = race["finish_position"]
    pred_top3 = set(race.loc[pred_rank <= 3, "driver_id"])
    real_top3 = set(race.loc[actual <= 3, "driver_id"])
    pred_top10 = set(race.loc[pred_rank <= 10, "driver_id"])
    real_top10 = set(race.loc[actual <= 10, "driver_id"])
    predicted_winner = race.loc[pred_rank.idxmin(), "finish_position"]
    return {
        "mae": (pred_rank - actual).abs().mean(),
        "rank_corr": pred_rank.corr(actual),
        "winner_correct": int(predicted_winner == 1),
        "podium_overlap": len(pred_top3 & real_top3) / 3,
        "top10_overlap": len(pred_top10 & real_top10) / 10,
    }


def run_backtest(df, setups, n_test_rounds):
    rounds = sorted(df["round"].unique())
    if len(rounds) - n_test_rounds < MIN_TRAIN_ROUNDS:
        raise SystemExit(
            f"Only {len(rounds)} rounds available: too few to test on {n_test_rounds} "
            f"and still train on at least {MIN_TRAIN_ROUNDS}. Use a smaller number.")
    test_rounds = rounds[-n_test_rounds:]

    rows, frames = [], []
    for r in test_rounds:
        train = df[df["round"] < r]
        test = df[df["round"] == r].copy()
        for name, setup in setups.items():
            test["predicted_score"] = predict_scores(setup, train, test)
            test["predicted_rank"] = test["predicted_score"].rank(method="first").astype(int)
            rows.append({"setup": name, "round": r, **score_race(test, "predicted_score")})
            frames.append(test.assign(setup=name)[
                ["setup", "round", "race_name", "driver_id", "constructor_id", "grid",
                 "predicted_score", "predicted_rank", "finish_position"]])
    return pd.DataFrame(rows), pd.concat(frames)


def main():
    n_test_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEST_ROUNDS

    print("=== F1 Forecast: Phase 4 Race Outcome Model (version 2) ===\n")
    df, all_features = load_data()
    setups = build_setups(all_features)
    print(f"Rows: {len(df)} | Rounds: {df['round'].nunique()} | Features available: {len(all_features)}")
    print(f"Backtest: last {n_test_rounds} rounds, each predicted using only earlier rounds")
    print(f"Setups compared: {', '.join(setups)}")

    results, predictions = run_backtest(df, setups, n_test_rounds)
    summary = results.groupby("setup")[METRICS].mean().reindex(setups.keys())

    print("\n--- Average over test races (all setups) ---")
    print(summary.round(3).to_string())
    print("\nmae: lower is better. All other columns: higher is better.")

    model_only = summary.drop(index="grid_baseline")
    best_name = model_only["mae"].idxmin()
    baseline_mae = summary.loc["grid_baseline", "mae"]
    best_mae = summary.loc[best_name, "mae"]
    print(f"\nBest model setup by position error: {best_name} (MAE {best_mae:.3f} vs grid baseline {baseline_mae:.3f})")
    print("Verdict:", "beats the grid baseline." if best_mae < baseline_mae
          else "does NOT beat the grid baseline (the grid is a very strong guess; that is a normal, honest result).")

    print("\n--- Best setup, race by race ---")
    per_race = results[results["setup"] == best_name].drop(columns="setup").round(2)
    names = df.drop_duplicates("round").set_index("round")["race_name"]
    per_race.insert(1, "race", per_race["round"].map(names))
    print(per_race.to_string(index=False))

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nSaved test predictions (all setups) to {PREDICTIONS_FILE}")

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


if __name__ == "__main__":
    main()
