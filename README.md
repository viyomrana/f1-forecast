# F1 Forecast

**An ML pipeline for Formula 1 race prediction.**

F1 Forecast is an end-to-end machine learning project that predicts Formula 1 race outcomes, explains those predictions, and models tyre degradation, using data from the 2026 season. It is being built as an academic project, with a Streamlit dashboard and an AWS deployment planned once the Python pipeline is complete.

> **Status:** Work in progress. Phases 1-5 are implemented. Phase 6 (dashboard) and Phase 7 (formal validation report) are planned, followed by cloud deployment.

---

## Why 2026 only?

The 2026 season introduced a new regulation era: revised car dimensions, a new power unit split, and the removal of DRS. Cars from earlier seasons are no longer a fair guide to current performance, so this project follows one hard rule:

**Only 2026 data is used for performance and telemetry features.** Pre-2026 data may be used only for non-performance, contextual features.

---

## Results so far

Every race from round 4 onward is a test race, predicted using only the races before it. The best model is chosen on rounds 4-11 and scored on rounds 12-20, which it never influenced.

| Method (rounds 12-20) | Average position error (lower is better) |
|---|---|
| Everyone finishes where they start (grid) | 3.62 |
| Everyone finishes where they qualified | 3.20 |
| **Chosen model (Ridge regression on six features)** | **3.16** |

- The model clearly beats the grid rule, mostly because it accounts for grid penalties.
- It is about level with the qualifying-order rule (better in 3 of 9 races). Qualifying already captures most of a car's speed.
- It picks the race winner in 33% of the held-out races, against 67% for the simple rules. It tends to pull front-runners toward the middle.
- Tyre degradation features and a race-pace feature were tested and gave no reliable improvement.
- SHAP shows qualifying position dominates (about 2.5 places of average push), followed by grid penalty (0.9) and team standing (0.8).

Only a handful of races exist so far, so these numbers are indicative, not conclusive.

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
| 3 | Tyre degradation and lap-based features | Done | `ingest_laps_2026.py`, `tyre_degradation.py`, `build_tyre_features.py` |
| 4 | Race outcome prediction with walk-forward backtest | Done (v4) | `train_model.py` |
| 5 | Explainability (SHAP) | Done | `explain_model.py` |
| 6 | Streamlit dashboard | Planned | |
| 7 | Formal backtesting and validation report | Planned | |

---

## How the pipeline fits together

```
Jolpica-F1 API --> ingest_2026.py --> data/raw/*.json
                                          |
                                          v
                                  build_features.py --> data/processed/features.csv
                                                                  |
FastF1 --> ingest_laps_2026.py --> data/raw/laps/*.csv            |
                  |                                               |
                  v                                               v
          tyre_degradation.py --> clean_laps.csv --> build_tyre_features.py --> features_tyre.csv
                                                                                      |
                                                                                      v
                                                          train_model.py --> models/best_model.joblib
                                                                                      |
                                                                                      v
                                                          explain_model.py --> reports/*.png
```

---

## Repository contents

| File | Purpose |
|---|---|
| `f1_data_pull.py` | Small smoke test that confirms FastF1 and Jolpica-F1 can be reached |
| `ingest_2026.py` | Phase 1: downloads and caches 2026 results, qualifying and standings as raw JSON |
| `build_features.py` | Phase 2: one row per driver per race (features and targets), with leakage protection |
| `ingest_laps_2026.py` | Phase 3: downloads lap-by-lap race data from FastF1, one CSV per round |
| `tyre_degradation.py` | Phase 3: cleans and fuel-corrects laps, then estimates seconds lost per lap of tyre age |
| `build_tyre_features.py` | Phase 3: turns lap data into leakage-safe team tyre and race-pace features |
| `train_model.py` | Phase 4: compares simple rules and models with a walk-forward backtest, saves the best model |
| `explain_model.py` | Phase 5: SHAP explanations and charts for the saved model |
| `reports/` | Charts produced by `explain_model.py` |
| `requirements.txt` | Python dependencies |
| `F1_Forecast_Synopsis.pdf` | Project synopsis |

The `data/`, `models/`, `f1_cache/` and virtual environment folders are not committed. Running the scripts below recreates them.

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
python ingest_2026.py         # raw results, qualifying, standings -> data/raw/
python build_features.py      # feature table -> data/processed/features.csv
python ingest_laps_2026.py    # lap data -> data/raw/laps/ (slow on first run)
python tyre_degradation.py    # tyre wear rates -> data/processed/
python build_tyre_features.py # lap-based features -> data/processed/features_tyre.csv
python train_model.py         # backtest and best model -> models/
python explain_model.py 20    # SHAP charts for round 20 -> reports/
python explain_model.py 20 norris   # also a chart for one driver
```

The ingestion scripts are safe to re-run: files that already exist are skipped, so re-run them after each new race. FastF1 downloads are cached in `f1_cache/`.

> **Note:** Recent FastF1 versions may ask for a Formula 1 (F1TV) account login when downloading new race data. Rounds that are already downloaded are cached and do not need it.

---

## Design notes

### Avoiding leakage

A feature for race N may only use information available before race N starts.

- Qualifying of round N is allowed (it happens before the race).
- Standings after round N-1 are allowed. Standings saved for round N are the standings *after* race N, so `build_features.py` shifts them forward by one round.
- Results and laps of rounds 1 to N-1 are allowed. Form, tyre and race-pace features are all built from earlier rounds only.
- Results or standings of round N are never used as features.

### Honest evaluation

The model is tested fairly: to predict a race, it only learns from the races before it, never from later ones. Its predictions are compared with simple rules that need no model: everyone finishes where they started, or where they qualified. A model is only useful if it beats those rules. See the top of `train_model.py` for the details.

### Tyre degradation assumptions

- Degradation is measured **within each stint** (one driver, one set of tyres), which cancels out driver skill, car speed and track differences.
- The fuel effect (`FUEL_EFFECT_S_PER_LAP`, default 0.03 s per lap) is an **assumed constant**, not measured. It cannot be separated from tyre wear with these columns alone.
- Track evolution also hides some wear, so true degradation is probably slightly higher than reported. Some soft-tyre rates come out near zero or negative for that reason.

### Explainability

SHAP splits each prediction into a starting value plus one push per feature, measured in places. A positive push moves a driver later, a negative push moves them earlier. Pushes are measured against the average driver, and the model predicts places gained or lost versus the grid slot, so a pole sitter can only be pushed later. The saved model is trained on all rounds, so the charts show how the model reasons, not how well it predicts. The backtest is the accuracy test.

---

## Known limitations

- Only a handful of 2026 races exist, so results are noisy.
- The best model is chosen from many candidates on the first half of the test races, so some luck is possible even with the held-out second half.
- The model treats penalty recovery the same for every car and pulls front-runners toward the middle, so it picks fewer winners than the simple rules.
- DNFs are inherently hard to predict and are part of the prediction error.
- Tyre degradation estimates are noisy and depend on the assumed fuel effect.

---

## Roadmap

- Phase 6: Streamlit dashboard with predictions, SHAP charts and tyre analysis
- Phase 7: formal backtesting and validation report
- Pit-strategy "what if" analysis using the tyre degradation results
- Containerise and deploy to AWS (ECS/Fargate) with CI/CD

---

## Author

Viyom Rana, B.Tech CSE, GLA University.
