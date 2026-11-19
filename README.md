# F1 Forecast

**An ML pipeline for Formula 1 race prediction.**

F1 Forecast is an end-to-end machine learning project that predicts Formula 1 race outcomes and models tyre degradation, using data from the 2026 season. It is being built as an academic project, with a cloud deployment (AWS) planned once the Python pipeline is complete.

> **Status:** Work in progress. Phases 1-3 are implemented. Phase 4 (race outcome prediction) has a working version 2 with a walk-forward backtest, and is being refined. Phases 5-7 are planned.

---

## Why 2026 only?

The 2026 season introduced a new regulation era: revised car dimensions, a new power unit split, and the removal of DRS. Cars from earlier seasons are no longer a fair guide to current performance, so this project follows one hard rule:

**Only 2026 data is used for performance and telemetry features.** Pre-2026 data may be used only for non-performance, contextual features.

---

## Data sources

| Source | Used for |
|---|---|
| [Jolpica-F1 API](https://api.jolpi.ca/ergast/f1) | Race results, qualifying, driver and constructor standings |
| [FastF1](https://docs.fastf1.dev/) | Lap-level race data (tyre compound, tyre age, pit stops, lap times) |

---

## Project phases

| # | Phase | Status | Script(s) |
|---|---|---|---|
| 1 | Data ingestion | Done | `ingest_2026.py` |
| 2 | Feature engineering | Done | `build_features.py` |
| 3 | Tyre degradation modelling | Done (not yet used as a model input) | `ingest_laps_2026.py`, `tyre_degradation.py` |
| 4 | Race outcome prediction (XGBoost) | In progress (v2 implemented) | `train_model.py` |
| 5 | Explainability (SHAP) | Planned | |
| 6 | Streamlit dashboard | Planned | |
| 7 | Backtesting and validation | Planned (a first walk-forward backtest already exists inside Phase 4) | |

---

## How the pipeline fits together

```
Jolpica-F1 API --> ingest_2026.py ------> data/raw/*.json
                                              |
                                              v
                                      build_features.py --> data/processed/features.csv
                                                                      |
FastF1 --> ingest_laps_2026.py --> data/raw/laps/*.csv                v
                  |                                            train_model.py --> models/best_model.joblib
                  v                                                   |          data/processed/backtest_predictions.csv
          tyre_degradation.py --> data/processed/                     v
                                  clean_laps.csv                (next: SHAP, dashboard)
                                  tyre_degradation_by_*.csv
```

---

## Repository contents

| File | Purpose |
|---|---|
| `f1_data_pull.py` | Small smoke test that confirms FastF1 and Jolpica-F1 can be reached |
| `ingest_2026.py` | Phase 1: downloads and caches 2026 results, qualifying and standings as raw JSON |
| `build_features.py` | Phase 2: builds one row per driver per race (features + targets) with leakage protection |
| `ingest_laps_2026.py` | Phase 3a: downloads lap-by-lap race data from FastF1, one CSV per round |
| `tyre_degradation.py` | Phase 3b: cleans and fuel-corrects laps, then estimates seconds lost per lap of tyre age, per compound, race and team |
| `train_model.py` | Phase 4: compares several prediction setups with a walk-forward backtest and saves the best model |
| `requirements.txt` | Python dependencies |
| `F1_Forecast_Synopsis.pdf` | Project synopsis |

The `data/`, `f1_cache/` and virtual environment folders are not committed. Running the scripts below recreates them.

---

## Getting started

**Requirements:** Python 3.10+ and Git.

```bash
# 1. Clone the repository
git clone https://github.com/viyomrana/f1-forecast.git
cd f1-forecast

# 2. Create and activate a virtual environment
python -m venv f1env
f1env\Scripts\activate          # Windows
# source f1env/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the pipeline

Run the scripts in this order. Each one reads the output of the one before it.

```bash
python f1_data_pull.py        # optional: checks that both data sources are reachable
python ingest_2026.py         # Phase 1: raw results, qualifying, standings -> data/raw/
python build_features.py      # Phase 2: feature table -> data/processed/features.csv
python ingest_laps_2026.py    # Phase 3a: lap data -> data/raw/laps/ (slow on first run)
python tyre_degradation.py    # Phase 3b: degradation rates -> data/processed/
python train_model.py         # Phase 4: backtest + best model -> models/
python train_model.py 3       # optional: backtest on the last 3 rounds instead of 5
```

The ingestion scripts are safe to re-run: files that already exist are skipped, so re-run them after each new race to pick up the latest round. FastF1 downloads are cached in `f1_cache/`.

---

## Design notes

### Avoiding leakage

A feature for race N may only use information available before race N starts.

- Qualifying of round N is allowed (it happens before the race).
- Standings after round N-1 are allowed. Standings saved for round N are the standings *after* race N, so `build_features.py` shifts them forward by one round.
- Results of rounds 1 to N-1 are allowed. All "recent form" features use `.shift(1)` before averaging.
- Results or standings of round N are never used as features.

### Honest evaluation

The model is tested fairly: to predict a race, it only learns from the races before it, never from later ones. Its predictions are compared with a simple guess that needs no model at all: every driver finishes where they started. A model is only useful if it beats that guess. See the top of `train_model.py` for the details.

### Tyre degradation assumptions

- Degradation is measured **within each stint** (one driver, one set of tyres), which cancels out driver skill, car speed and track differences.
- The fuel effect (`FUEL_EFFECT_S_PER_LAP`, default 0.03 s per lap) is an **assumed constant**, not measured. It cannot be separated from tyre wear with these columns alone.
- Track evolution also hides some wear, so true degradation is probably slightly higher than reported.
- Rates are only reported when enough stints are available.

---

## Known limitations

- Only a handful of 2026 races exist so far, so backtest results are noisy and should be read as indicative, not conclusive.
- The "best" setup is chosen using the same last-N races it is scored on, so its score is slightly optimistic.
- Tyre degradation results are not yet used as inputs to the race outcome model.
- DNFs are inherently hard to predict and are currently part of the prediction error.

---

## Roadmap

- Feed tyre degradation into the race outcome model as leakage-safe features
- Extend the backtest and add more baselines and metrics
- Phase 5: SHAP explanations for model predictions
- Phase 6: Streamlit dashboard
- Phase 7: full backtesting and validation report
- Containerise and deploy to AWS (ECS/Fargate)

---

## Author

Viyom Rana, B.Tech CSE, GLA University.
