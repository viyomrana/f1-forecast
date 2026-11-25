# F1 Forecast

A machine-learning pipeline that predicts Formula 1 race results for the 2026 season, explains its predictions, and shows everything in a Streamlit dashboard.

> **Data rule:** only 2026 data is used for performance features. The 2026 regulations changed the cars and power units too much for older seasons to be comparable.

## What it does

1. Downloads 2026 race, qualifying, standings and lap data
2. Builds features for each driver in each race
3. Measures tyre degradation per team and per race
4. Trains models to predict finishing order and tests them honestly on unseen races
5. Explains each prediction with SHAP
6. Presents the results in a dashboard

## Data sources

- [Jolpica-F1 API](https://api.jolpi.ca/ergast/f1): results, qualifying, standings
- [FastF1](https://docs.fastf1.dev): lap data

## Project phases

| Phase | Script | What it does |
|---|---|---|
| 1 | `ingest_2026.py` | Downloads results, qualifying and standings for every completed 2026 round into `data/raw/` |
| 2 | `build_features.py` | Cleans the raw data and builds the feature table (`data/processed/features.csv`) |
| 3 | `ingest_laps_2026.py`, `tyre_degradation.py` | Downloads lap data and measures how quickly each team's tyres degrade |
| 3b | `build_tyre_features.py` | Adds tyre and race-pace features (`features_tyre.csv`) |
| 4 | `train_model.py` | Compares several models and baselines with a walk-forward backtest, saves the best one |
| 5 | `explain_model.py` | SHAP explanations: which features push each driver earlier or later |
| 6 | `dashboard.py` | Streamlit dashboard |

`f1_data_pull.py` is a small starter script used to test API access.

## Results

Every race from round 4 onward is predicted using only the races before it. The best setup is chosen on the first half of the test races and scored on the second half (17 test races in total).

| Method | Average position error (held-out races) |
|---|---|
| **Chosen model (`ridge_vs_grid_simple`)** | **3.162** |
| Qualifying-order baseline | 3.202 |
| Grid-order baseline | 3.616 |

- The model is clearly better than the grid baseline and about level with the qualifying-order baseline.
- Tyre and race-pace features gave no reliable gain, so the chosen model does not use them.
- It is not good at picking race winners. With so few races and so many unpredictable events (safety cars, strategy, reliability), that is expected.
- SHAP charts describe how the model reasons. The saved model was trained on all rounds, so the honest accuracy test is the backtest.

## Dashboard

```
streamlit run dashboard.py
```

Opens at http://localhost:8501. Tabs:

- **Race predictions:** predicted vs actual finishing order for any test race
- **Why (SHAP):** feature importance, and the reasons behind any driver's prediction
- **Backtest:** model vs baselines on held-out races
- **Tyre degradation:** degradation by team and by race
- **About:** project summary

<!-- Add screenshots after committing them to reports/screenshots/:
![Race predictions](reports/screenshots/dashboard_predictions.png)
-->

## How to run

```
pip install -r requirements.txt

python ingest_2026.py          # Phase 1: download race data
python build_features.py       # Phase 2: build features
python ingest_laps_2026.py     # Phase 3: download lap data
python tyre_degradation.py     # Phase 3: tyre degradation
python build_tyre_features.py  # Phase 3b: tyre features
python train_model.py          # Phase 4: train and backtest
python explain_model.py        # Phase 5: SHAP (optional: python explain_model.py 12 norris)
streamlit run dashboard.py     # Phase 6: dashboard
```

Re-run `ingest_2026.py` after each new race to pick up the new round, then repeat the later steps.

The `data/`, `models/` and `f1_cache/` folders are not stored in this repository. The scripts above regenerate them.

## Project layout

```
f1-forecast/
├── ingest_2026.py          Phase 1
├── build_features.py       Phase 2
├── ingest_laps_2026.py     Phase 3
├── tyre_degradation.py     Phase 3
├── build_tyre_features.py  Phase 3b
├── train_model.py          Phase 4
├── explain_model.py        Phase 5
├── dashboard.py            Phase 6
├── f1_data_pull.py         API access test
├── requirements.txt
└── reports/                SHAP charts
```

## Planned: cloud deployment

Containerise the dashboard and run it on AWS (ECS/Fargate, ECR, S3, DynamoDB, EventBridge, CloudWatch), with CI/CD through GitHub Actions.

## Tech

Python, pandas, scikit-learn, XGBoost, SHAP, FastF1, Streamlit
