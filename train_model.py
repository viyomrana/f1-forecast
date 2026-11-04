"""
F1 Forecast - Phase 4: Race Outcome Prediction (XGBoost) + Backtest
===================================================================
Reads data/processed/features.csv (built by build_features.py), trains an
XGBoost model to predict each driver's finishing position, and tests it
HONESTLY on races the model has never seen.

How the honest test works (walk-forward backtest)
-------------------------------------------------
For each of the last N rounds (default 5), we:
  1. train the model ONLY on rounds BEFORE that round,
  2. predict that round,
  3. compare the predicted finishing order with the real result.
A random train/test split would let the model learn from races that happen
AFTER the ones it predicts, which makes accuracy look better than it is.

We also compare against a simple baseline: "everyone finishes where they
started on the grid". If the model can't beat that, it isn't adding value.

Metrics (computed per race, then averaged)
------------------------------------------
  MAE (positions)  : average gap between predicted and actual position (lower = better)
  Rank correlation : how well the predicted order matches the real order (1.0 = perfect)
  Winner correct   : 1 if the predicted winner actually won
  Podium overlap   : share of the real top 3 found in the predicted top 3
  Top-10 overlap   : share of the real top 10 found in the predicted top 10

Outputs
-------
  data/processed/backtest_predictions.csv : every test prediction next to the real result
  models/xgb_position.json                : model retrained on ALL rounds
  models/feature_columns.json             : the feature names the model expects

Run this with your virtual environment active:
    pip install xgboost scikit-learn
    python train_model.py          # backtest on the last 5 rounds
    python train_model.py 3        # backtest on the last 3 rounds

Project hard rule: only 2026 data is used for performance features.
"""

import json
import os
import sys

import pandas as pd
from xgboost import XGBRegressor

FEATURES_FILE = os.path.join("data", "processed", "features.csv")
PREDICTIONS_FILE = os.path.join("data", "processed", "backtest_predictions.csv")
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "xgb_position.json")
FEATURE_LIST_FILE = os.path.join(MODEL_DIR, "feature_columns.json")

DEFAULT_TEST_ROUNDS = 5
MIN_TRAIN_ROUNDS = 3  # never train on fewer rounds than this

ID_COLS = ["round", "race_name", "driver_id", "constructor_id"]
TARGET_COLS = ["finish_position", "points", "podium", "points_finish", "is_dnf", "status"]

# Small, cautious settings: we only have a few hundred rows, so a simple model
# (shallow trees, slow learning) is less likely to memorise noise.
MODEL_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def load_data():
    if not os.path.exists(FEATURES_FILE):
        raise SystemExit(f"{FEATURES_FILE} not found. Run build_features.py first.")
    df = pd.read_csv(FEATURES_FILE)
    feature_cols = [c for c in df.columns if c not in ID_COLS + TARGET_COLS]
    return df, feature_cols


def score_race(race, score_col):
    """
    Scores one race. `score_col` holds a number per driver where LOWER means
    'finishes earlier' (the model's predicted position, or the grid slot).
    """
    pred_rank = race[score_col].rank(method="first")  # 1 = predicted winner
    actual = race["finish_position"]

    pred_top3 = set(race.loc[pred_rank <= 3, "driver_id"])
    real_top3 = set(race.loc[actual <= 3, "driver_id"])
    pred_top10 = set(race.loc[pred_rank <= 10, "driver_id"])
    real_top10 = set(race.loc[actual <= 10, "driver_id"])
    predicted_winner = race.loc[pred_rank.idxmin(), "finish_position"]

    return {
        "mae": (pred_rank - actual).abs().mean(),
        "rank_corr": pred_rank.corr(actual),  # correlation of ranks = Spearman
        "winner_correct": int(predicted_winner == 1),
        "podium_overlap": len(pred_top3 & real_top3) / 3,
        "top10_overlap": len(pred_top10 & real_top10) / 10,
    }


def run_backtest(df, feature_cols, n_test_rounds):
    rounds = sorted(df["round"].unique())
    if len(rounds) - n_test_rounds < MIN_TRAIN_ROUNDS:
        raise SystemExit(
            f"Only {len(rounds)} rounds available: too few to test on {n_test_rounds} "
            f"and still train on at least {MIN_TRAIN_ROUNDS}. Use a smaller number."
        )
    test_rounds = rounds[-n_test_rounds:]

    model_rows, baseline_rows, prediction_frames = [], [], []

    for r in test_rounds:
        train = df[df["round"] < r]  # strictly earlier races only
        test = df[df["round"] == r].copy()

        model = XGBRegressor(**MODEL_PARAMS)
        model.fit(train[feature_cols], train["finish_position"])
        test["predicted_score"] = model.predict(test[feature_cols])
        test["predicted_rank"] = test["predicted_score"].rank(method="first").astype(int)

        model_scores = score_race(test, "predicted_score")
        baseline_scores = score_race(test, "grid")
        model_rows.append({"round": r, "race": test["race_name"].iloc[0], **model_scores})
        baseline_rows.append({"round": r, **baseline_scores})

        prediction_frames.append(
            test[["round", "race_name", "driver_id", "constructor_id", "grid",
                  "predicted_score", "predicted_rank", "finish_position"]]
        )

    return pd.DataFrame(model_rows), pd.DataFrame(baseline_rows), pd.concat(prediction_frames)


def print_results(model_df, baseline_df):
    metrics = ["mae", "rank_corr", "winner_correct", "podium_overlap", "top10_overlap"]

    print("\n--- Per race (model) ---")
    show = model_df.copy()
    for c in metrics:
        show[c] = show[c].round(2)
    print(show.to_string(index=False))

    print("\n--- Average over test races: model vs grid baseline ---")
    summary = pd.DataFrame({
        "model": model_df[metrics].mean(),
        "grid_baseline": baseline_df[metrics].mean(),
    }).round(3)
    summary.index = [
        "MAE (positions, lower is better)",
        "Rank correlation",
        "Winner correct",
        "Podium overlap",
        "Top-10 overlap",
    ]
    print(summary.to_string())

    beat = model_df["mae"].mean() < baseline_df["mae"].mean()
    print("\nVerdict on position error:",
          "model beats the grid baseline." if beat
          else "model does NOT beat the grid baseline (common with small data; see notes).")


def main():
    n_test_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEST_ROUNDS

    print("=== F1 Forecast: Phase 4 Race Outcome Model ===\n")
    df, feature_cols = load_data()
    print(f"Rows: {len(df)} | Rounds: {df['round'].nunique()} | Features: {len(feature_cols)}")
    print(f"Backtest: last {n_test_rounds} rounds, each predicted using only earlier rounds")

    model_df, baseline_df, predictions = run_backtest(df, feature_cols, n_test_rounds)
    print_results(model_df, baseline_df)

    predictions.to_csv(PREDICTIONS_FILE, index=False)
    print(f"\nSaved test predictions to {PREDICTIONS_FILE}")

    # Final model: retrain on ALL rounds (this is the one used for future races)
    final_model = XGBRegressor(**MODEL_PARAMS)
    final_model.fit(df[feature_cols], df["finish_position"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save_model(MODEL_FILE)
    with open(FEATURE_LIST_FILE, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"Saved final model (trained on all {df['round'].nunique()} rounds) to {MODEL_FILE}")

    print("\nMost influential features (final model):")
    importance = pd.Series(final_model.feature_importances_, index=feature_cols)
    print(importance.sort_values(ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
