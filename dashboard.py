"""
F1 Forecast - Phase 6: Streamlit Dashboard
==========================================
A read-only front end for everything the earlier phases produced. It does NOT
retrain anything: it reads the saved files and shows them.

Reads
-----
  models/best_model_info.json               (train_model.py)  which setup won, its features
  data/processed/backtest_predictions.csv   (train_model.py)  honest predictions for every test race
  data/processed/backtest_metrics.csv       (train_model.py)  per-race scores for every setup
  data/processed/shap_values.csv            (explain_model.py) per-driver feature pushes
  data/processed/tyre_degradation_by_*.csv  (tyre_degradation.py)
  reports/shap_*.png                        (explain_model.py)

Run (from the f1-forecast folder):
    pip install streamlit
    streamlit run dashboard.py

Project hard rule: only 2026 data is used for performance features.
"""

import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

PROCESSED = os.path.join("data", "processed")
REPORTS = "reports"
INFO_FILE = os.path.join("models", "best_model_info.json")
PRED_FILE = os.path.join(PROCESSED, "backtest_predictions.csv")
METRICS_FILE = os.path.join(PROCESSED, "backtest_metrics.csv")
SHAP_FILE = os.path.join(PROCESSED, "shap_values.csv")
DEG_TEAM_FILE = os.path.join(PROCESSED, "tyre_degradation_by_team.csv")
DEG_RACE_FILE = os.path.join(PROCESSED, "tyre_degradation_by_race.csv")

BLUE = "#2b6cb0"   # earlier / better
RED = "#c0392b"    # later / worse
MIN_RACES_TO_SPLIT = 6  # same rule as train_model.py

st.set_page_config(page_title="F1 Forecast", page_icon="🏎️", layout="wide")


# ---------- loading ----------

@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_info():
    if not os.path.exists(INFO_FILE):
        return None
    with open(INFO_FILE, encoding="utf-8") as f:
        return json.load(f)


def need(df, label, script):
    """Stops the page with a friendly message if a file is missing."""
    if df is None:
        st.error(f"{label} not found. Run `{script}` first.")
        st.stop()
    return df


def bar(series, x_title, by_sign=False, height_per_bar=28):
    """Horizontal bar chart, largest absolute value on top."""
    data = series.rename("value").rename_axis("feature").reset_index()
    order = series.abs().sort_values(ascending=False).index.tolist()
    if by_sign:
        data["direction"] = np.where(data["value"] > 0, "later", "earlier")
        colour = alt.Color("direction:N", legend=None,
                           scale=alt.Scale(domain=["earlier", "later"], range=[BLUE, RED]))
    else:
        colour = alt.value(BLUE)
    chart = (alt.Chart(data).mark_bar()
             .encode(x=alt.X("value:Q", title=x_title),
                     y=alt.Y("feature:N", sort=order, title=None),
                     color=colour,
                     tooltip=["feature", alt.Tooltip("value:Q", format="+.2f")])
             .properties(height=max(120, height_per_bar * len(data))))
    return chart


# ---------- data ----------

info = need(load_info(), INFO_FILE, "train_model.py")
preds_all = need(load_csv(PRED_FILE), PRED_FILE, "train_model.py")
metrics_all = need(load_csv(METRICS_FILE), METRICS_FILE, "train_model.py")
chosen = info["setup"]

preds = preds_all[preds_all["setup"] == chosen].copy()
race_names = (preds_all.drop_duplicates("round").set_index("round")["race_name"].to_dict())

# ---------- sidebar ----------

with st.sidebar:
    st.title("🏎️ F1 Forecast")
    st.caption("Race outcome prediction, 2026 season")
    st.markdown(f"**Chosen model:** `{chosen}`")
    target_text = ("places gained or lost versus the grid slot" if info["target"] == "delta"
                   else "finishing position")
    st.markdown(f"**Predicts:** {target_text}")
    st.markdown(f"**Trained on:** {info['trained_on_rounds']} rounds")
    with st.expander("Features used"):
        for feature in info["features"]:
            st.write(f"- {feature}")
    st.caption("Only 2026 data is used for performance features, because of the "
               "new regulation era.")

tab_pred, tab_shap, tab_backtest, tab_tyre, tab_about = st.tabs(
    ["Race predictions", "Why (SHAP)", "Backtest", "Tyre degradation", "About"])

# ---------- tab 1: race predictions ----------

with tab_pred:
    st.subheader("Predicted vs actual finishing order")
    st.caption("These are out-of-sample predictions: each race was predicted using only "
               "the races before it.")
    rounds = sorted(preds["round"].unique())
    round_number = st.selectbox("Race", rounds, index=len(rounds) - 1,
                                format_func=lambda r: f"Round {r}: {race_names.get(r, '')}")
    race = preds[preds["round"] == round_number].sort_values("predicted_rank")
    scores = metrics_all[(metrics_all["setup"] == chosen) & (metrics_all["round"] == round_number)]
    grid_scores = metrics_all[(metrics_all["setup"] == "grid_baseline") & (metrics_all["round"] == round_number)]

    if not scores.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average position error", f"{scores['mae'].iloc[0]:.2f}",
                  delta=(f"{scores['mae'].iloc[0] - grid_scores['mae'].iloc[0]:+.2f} vs grid"
                         if not grid_scores.empty else None),
                  delta_color="inverse")
        c2.metric("Order match (rank corr.)", f"{scores['rank_corr'].iloc[0]:.2f}")
        c3.metric("Podium overlap", f"{scores['podium_overlap'].iloc[0]:.0%}")
        c4.metric("Top 10 overlap", f"{scores['top10_overlap'].iloc[0]:.0%}")

    predicted_winner = race.iloc[0]["driver_id"]
    actual_winner = race.loc[race["finish_position"].idxmin(), "driver_id"]
    st.write(f"**Predicted winner:** {predicted_winner}  |  **Actual winner:** {actual_winner}")

    table = pd.DataFrame({
        "Predicted": race["predicted_rank"].astype(int),
        "Driver": race["driver_id"],
        "Team": race["constructor_id"],
        "Grid": race["grid"].astype(int),
        "Actual finish": race["finish_position"].astype(int),
    })
    table["Error (places)"] = table["Predicted"] - table["Actual finish"]
    st.dataframe(table, hide_index=True)
    st.caption("Error: negative = the model expected a better finish than the driver got.")

# ---------- tab 2: SHAP ----------

with tab_shap:
    st.subheader("Why does the model predict what it predicts?")
    shap_df = need(load_csv(SHAP_FILE), SHAP_FILE, "explain_model.py")
    shap_cols = [c for c in shap_df.columns if c.startswith("shap_")]

    st.markdown("**Which features matter most overall**")
    importance = shap_df[shap_cols].abs().mean()
    importance.index = [c.replace("shap_", "") for c in importance.index]
    st.altair_chart(bar(importance, "Average push on the prediction (places)"))

    st.markdown("**One driver in one race**")
    col_a, col_b = st.columns(2)
    shap_rounds = sorted(shap_df["round"].unique())
    shap_round = col_a.selectbox("Race ", shap_rounds, index=len(shap_rounds) - 1,
                                 format_func=lambda r: f"Round {r}: "
                                 f"{shap_df.loc[shap_df['round'] == r, 'race_name'].iloc[0]}")
    race_shap = shap_df[shap_df["round"] == shap_round].sort_values("predicted_rank")
    driver = col_b.selectbox("Driver", race_shap["driver_id"].tolist())
    row = race_shap[race_shap["driver_id"] == driver].iloc[0]

    st.write(f"**{driver}**: grid {int(row['grid'])}, predicted rank {int(row['predicted_rank'])}, "
             f"actual finish {int(row['finish_position'])}")
    pushes = row[shap_cols].astype(float)
    pushes.index = [c.replace("shap_", "") for c in pushes.index]
    st.altair_chart(bar(pushes, "Push on the prediction (places; red = later, blue = earlier)",
                        by_sign=True))

    heatmap = os.path.join(REPORTS, f"shap_round_{shap_round}.png")
    if os.path.exists(heatmap):
        with st.expander("Heat-map for the whole race"):
            st.image(heatmap)

    st.info("The saved model was trained on all rounds, so these charts show how the model "
            "reasons, not how well it predicts. The honest accuracy test is the Backtest tab.")

# ---------- tab 3: backtest ----------

with tab_backtest:
    st.subheader("How good is it, honestly?")
    st.caption("Walk-forward backtest: every race from round 4 onward is predicted using only "
               "earlier races. Lower position error is better.")

    avg = metrics_all.groupby("setup")[["mae", "mae_finishers", "rank_corr", "winner_correct",
                                        "podium_overlap", "top10_overlap"]].mean()
    test_rounds = sorted(metrics_all["round"].unique())

    if len(test_rounds) >= MIN_RACES_TO_SPLIT:
        half = len(test_rounds) // 2
        final_rounds = test_rounds[half:]
    else:
        final_rounds = test_rounds
    final = (metrics_all[metrics_all["round"].isin(final_rounds)]
             .groupby("setup")[["mae", "winner_correct"]].mean())
    baselines = [s for s in final.index if s.endswith("_baseline")]
    strongest = final.loc[baselines, "mae"].idxmin()
    per_race = (metrics_all[metrics_all["round"].isin(final_rounds)]
                .pivot(index="round", columns="setup", values="mae"))

    st.markdown(f"**Held-out races (rounds {final_rounds[0]} to {final_rounds[-1]}, "
                f"{len(final_rounds)} races)**")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Chosen model", f"{final.loc[chosen, 'mae']:.3f}")
    c2.metric("Grid baseline", f"{final.loc['grid_baseline', 'mae']:.3f}")
    c3.metric(f"Strongest rule ({strongest})", f"{final.loc[strongest, 'mae']:.3f}")
    wins_grid = int((per_race[chosen] < per_race["grid_baseline"]).sum())
    wins_best = int((per_race[chosen] < per_race[strongest]).sum())
    st.write(f"Better than the grid baseline in **{wins_grid} of {len(final_rounds)}** races, "
             f"better than {strongest} in **{wins_best} of {len(final_rounds)}**. "
             f"Winner picked correctly: model {final.loc[chosen, 'winner_correct']:.0%}, "
             f"grid baseline {final.loc['grid_baseline', 'winner_correct']:.0%}.")

    st.markdown("**Position error per race**")
    show = [s for s in [chosen, "grid_baseline", "quali_baseline"] if s in metrics_all["setup"].unique()]
    line = (metrics_all[metrics_all["setup"].isin(show)]
            .pivot(index="round", columns="setup", values="mae")[show])
    st.line_chart(line)

    st.markdown("**All setups, average over every test race**")
    st.dataframe(avg.round(3).rename_axis("setup").reset_index(), hide_index=True)
    st.caption("mae and mae_finishers: lower is better. All other columns: higher is better.")

# ---------- tab 4: tyre degradation ----------

with tab_tyre:
    st.subheader("Tyre degradation (Phase 3)")
    st.caption("Built from 2026 lap data only.")
    deg_team = load_csv(DEG_TEAM_FILE)
    deg_race = load_csv(DEG_RACE_FILE)
    if deg_team is None and deg_race is None:
        st.warning("No tyre degradation files found. Run `tyre_degradation.py` first.")
    if deg_team is not None:
        st.markdown("**By team**")
        st.dataframe(deg_team, hide_index=True)
    if deg_race is not None:
        st.markdown("**By race**")
        st.dataframe(deg_race, hide_index=True)
    st.info("In the backtest the tyre features gave no reliable gain over the simpler "
            "setup, so the chosen model does not use them.")

# ---------- tab 5: about ----------

with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
**F1 Forecast** is an end-to-end machine-learning pipeline that predicts Formula 1 race
outcomes and models tyre degradation, built on the 2026 season.

**Data:** Jolpica-F1 API (results, qualifying, standings) and FastF1 (lap data).
Only 2026 data feeds performance features, because the 2026 regulations changed the cars
and power units too much for older seasons to be comparable.

**Pipeline:** ingestion, feature engineering, tyre degradation, XGBoost / Ridge models,
walk-forward backtest, SHAP explanations, and this dashboard.

**Honest result:** the chosen model beats the grid-order baseline clearly, and is about
level with the qualifying-order baseline. It is not good at picking race winners. That is
expected: a few races, many unpredictable events (safety cars, strategy, reliability).

**Cloud design:** containerised services on AWS ECS/Fargate, with S3 for data, DynamoDB
for predictions, EventBridge for scheduled ingestion, and CloudWatch for monitoring.
        """
    )
