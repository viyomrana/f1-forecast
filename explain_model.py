"""
F1 Forecast - Phase 5: Explainability (SHAP)
============================================
Explains WHY the chosen model predicts what it predicts.

SHAP splits each prediction into a starting value plus one push per feature.
The model predicts either "places gained or lost versus the grid slot" (target
'delta') or the finishing position directly (target 'position'). In both cases
the pushes are in PLACES, and:

    positive push = moves the driver LATER (a worse finish)
    negative push = moves the driver EARLIER (a better finish)

Reads
-----
  models/best_model.joblib and models/best_model_info.json   (from train_model.py)
  data/processed/features_tyre.csv (or features.csv)         (the same table the model used)

Writes
------
  data/processed/shap_values.csv   : one row per driver per race, one SHAP column per feature
  reports/shap_global.png          : which features matter most overall
  reports/shap_round_<N>.png       : heat-map of every driver's pushes in one race
  reports/shap_round_<N>_<driver>.png : bar chart for one driver (only if you name one)

Run this with your virtual environment active:
    pip install shap matplotlib
    python explain_model.py                  # latest race
    python explain_model.py 12               # round 12
    python explain_model.py 12 norris        # round 12, plus a chart for one driver

Important caveat (say this in your report): the saved model was trained on ALL
rounds, so explanations of past races describe how the model reasons, not an
out-of-sample test. The honest accuracy test is the backtest in train_model.py.

Project hard rule: only 2026 data is used for performance features.
"""

import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

FEATURES_FILE = os.path.join("data", "processed", "features.csv")
LAP_FEATURES_FILE = os.path.join("data", "processed", "features_tyre.csv")
MODEL_FILE = os.path.join("models", "best_model.joblib")
INFO_FILE = os.path.join("models", "best_model_info.json")
SHAP_FILE = os.path.join("data", "processed", "shap_values.csv")
REPORT_DIR = "reports"

EARLIER_COLOUR = "#2b6cb0"  # blue: pushes the driver to finish earlier
LATER_COLOUR = "#c0392b"    # red: pushes the driver to finish later


def load_everything():
    for path in (MODEL_FILE, INFO_FILE):
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found. Run train_model.py first.")
    model = joblib.load(MODEL_FILE)
    with open(INFO_FILE, encoding="utf-8") as f:
        info = json.load(f)

    data_path = LAP_FEATURES_FILE if os.path.exists(LAP_FEATURES_FILE) else FEATURES_FILE
    df = pd.read_csv(data_path)
    if {"grid", "quali_position"} <= set(df.columns):
        df["grid_penalty"] = df["grid"] - df["quali_position"]  # same derived column as train_model.py
    missing = [c for c in info["features"] if c not in df.columns]
    if missing:
        raise SystemExit(f"Columns the model needs are missing: {missing}. "
                         f"Re-run build_tyre_features.py and train_model.py.")
    return model, info, df


def compute_shap(model, info, df):
    """Returns (shap_values as DataFrame with one column per feature, base_value, predictions)."""
    features = info["features"]
    X = df[features]

    if hasattr(model, "named_steps"):
        # Linear pipeline: impute -> scale -> Ridge. Explain the Ridge step on the processed data.
        prepared = model[:-1].transform(X)
        kept = [f for f, s in zip(features, model[0].statistics_) if not np.isnan(s)]
        # use every row as the background (default would randomly sample 100 of them)
        masker = shap.maskers.Independent(prepared, max_samples=len(prepared))
        explainer = shap.LinearExplainer(model[-1], masker)
        values = explainer.shap_values(prepared)
        names = kept
    else:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        names = features

    base = float(np.ravel(explainer.expected_value)[0])
    predictions = model.predict(X)

    # Sanity check: base value + all pushes must add up to the model's own prediction
    error = float(np.abs(base + np.asarray(values).sum(axis=1) - predictions).max())
    return pd.DataFrame(np.asarray(values), columns=names, index=df.index), base, predictions, error


def predicted_rank(group):
    order = group.sort_values(["predicted_score", "grid", "driver_id"], kind="mergesort")
    return pd.Series(np.arange(1, len(order) + 1), index=order.index).reindex(group.index)


def build_table(df, info, shap_df, base, predictions):
    out = df[["round", "race_name", "driver_id", "constructor_id", "grid", "finish_position"]].copy()
    out["base_value"] = base
    out["model_output"] = predictions
    out["predicted_score"] = out["grid"] + predictions if info["target"] == "delta" else predictions
    out["predicted_rank"] = out.groupby("round", group_keys=False)[
        ["predicted_score", "grid", "driver_id"]].apply(predicted_rank)
    shap_cols = shap_df.add_prefix("shap_")
    return pd.concat([out, shap_cols], axis=1)


def meaning(info):
    return ("places gained (-) or lost (+) versus the grid slot" if info["target"] == "delta"
            else "places versus the average finishing position")


def global_chart(shap_df, info):
    importance = shap_df.abs().mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(importance) + 1.8))
    ax.barh(importance.index, importance.values, color=EARLIER_COLOUR)
    ax.set_xlabel("Average push on the prediction (places)")
    ax.set_title(f"Which features matter most ({info['setup']})")
    for y, v in enumerate(importance.values):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, importance.max() * 1.15)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, "shap_global.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def race_chart(race, shap_cols, round_number, info):
    race = race.sort_values("predicted_rank")
    matrix = race[shap_cols].values
    limit = max(float(np.abs(matrix).max()), 0.5)
    fig, ax = plt.subplots(figsize=(1.2 * len(shap_cols) + 4.5, 0.36 * len(race) + 2.2))
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(shap_cols)))
    ax.set_xticklabels([c.replace("shap_", "").replace("_", "\n") for c in shap_cols], fontsize=8)
    ax.set_yticks(range(len(race)))
    ax.set_yticklabels([f"{int(r.predicted_rank)}. {r.driver_id} (grid {int(r.grid)})"
                        for r in race.itertuples()], fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            strong = abs(matrix[i, j]) > 0.55 * limit  # white text on the darkest cells
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if strong else "black")
    ax.set_title(f"Round {round_number}: {race['race_name'].iloc[0]}\n"
                 f"Why each driver is predicted where they are (red = later, blue = earlier)", fontsize=10)
    fig.colorbar(image, ax=ax, label="places")
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"shap_round_{round_number}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def driver_chart(row, shap_cols, round_number, info):
    pushes = row[shap_cols].astype(float).rename(lambda c: c.replace("shap_", ""))
    pushes = pushes.reindex(pushes.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(pushes) + 2.2))
    colours = [LATER_COLOUR if v > 0 else EARLIER_COLOUR for v in pushes.values]
    ax.barh(pushes.index, pushes.values, color=colours)
    ax.axvline(0, color="#444444", linewidth=0.8)
    for y, v in enumerate(pushes.values):
        ax.text(v, y, f" {v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("Push on the prediction (places; red = later, blue = earlier)")
    ax.set_title(f"{row['driver_id']}, round {round_number}: grid {int(row['grid'])}, "
                 f"predicted rank {int(row['predicted_rank'])}", fontsize=10)
    margin = max(float(pushes.abs().max()) * 0.25, 0.2)
    ax.set_xlim(pushes.min() - margin, pushes.max() + margin)
    fig.tight_layout()
    path = os.path.join(REPORT_DIR, f"shap_round_{round_number}_{row['driver_id']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    print("=== F1 Forecast: Phase 5 Explainability (SHAP) ===\n")
    model, info, df = load_everything()
    print(f"Model    : {info['setup']} (predicts: {meaning(info)})")
    print(f"Features : {', '.join(info['features'])}")

    shap_df, base, predictions, error = compute_shap(model, info, df)
    print(f"Check    : starting value + feature pushes match the model's own prediction "
          f"(largest difference {error:.6f})")
    if error > 1e-3:
        print("WARNING: the pushes do not add up to the prediction. Do not trust these charts yet.")

    table = build_table(df, info, shap_df, base, predictions)
    shap_cols = [c for c in table.columns if c.startswith("shap_")]
    os.makedirs(REPORT_DIR, exist_ok=True)
    table.to_csv(SHAP_FILE, index=False)

    rounds = sorted(table["round"].unique())
    round_number = int(sys.argv[1]) if len(sys.argv) > 1 else int(rounds[-1])
    if round_number not in rounds:
        raise SystemExit(f"Round {round_number} not in the data (available: {rounds[0]} to {rounds[-1]}).")
    driver_query = sys.argv[2].lower() if len(sys.argv) > 2 else None

    print(f"\n--- Which features matter most overall (average push, in places) ---")
    importance = shap_df.abs().mean().sort_values(ascending=False)
    print(importance.round(3).to_string())

    race = table[table["round"] == round_number].sort_values("predicted_rank")
    print(f"\n--- Round {round_number}: {race['race_name'].iloc[0]} ---")
    print("Biggest reason for each driver's predicted position (push in places; + = later, - = earlier):")
    print(f"{'pred':>4}  {'driver':<14}{'grid':>4}  {'actual':>6}  biggest reasons")
    for r in race.itertuples():
        pushes = pd.Series({c: getattr(r, c) for c in shap_cols})
        top = pushes.reindex(pushes.abs().sort_values(ascending=False).index).head(2)
        reasons = ", ".join(f"{name.replace('shap_', '')} {value:+.1f}" for name, value in top.items())
        print(f"{int(r.predicted_rank):>4}  {r.driver_id:<14}{int(r.grid):>4}  {int(r.finish_position):>6}  {reasons}")

    paths = [global_chart(shap_df, info), race_chart(race, shap_cols, round_number, info)]
    if driver_query:
        match = race[race["driver_id"].str.contains(driver_query, case=False)]
        if match.empty:
            print(f"\nNo driver matching '{driver_query}' in round {round_number}.")
        else:
            paths.append(driver_chart(match.iloc[0], shap_cols, round_number, info))

    print(f"\nSaved table  : {SHAP_FILE}")
    print("Saved charts : " + ", ".join(paths))
    print("\nNote: the model was trained on all rounds, so this explains how it reasons, "
          "not how well it predicts. The honest accuracy test is the backtest in train_model.py.")


if __name__ == "__main__":
    main()
