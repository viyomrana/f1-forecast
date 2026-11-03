# F1 Forecast

**A cloud-native ML pipeline for Formula 1 race prediction.**

F1 Forecast is an end-to-end machine learning project that aims to predict Formula 1 race outcomes and model tyre degradation and pit strategy, using data from the 2026 season. It is being built as an academic cloud computing project and is designed to be deployed on AWS.

> **Status:** Work in progress. Phase 1 (data ingestion) is implemented. The remaining phases are planned.

---

## Why 2026 only?

The 2026 season introduced a new regulation era: revised car dimensions, a new power unit split, and the removal of DRS. Cars from earlier seasons are no longer a fair guide to current performance, so this project follows one hard rule:

**Only 2026 data is used for performance and telemetry features.** Pre-2026 data may be used only for non-performance, contextual features.

---

## Data sources

| Source | Used for |
|---|---|
| [Jolpica-F1 API](https://api.jolpi.ca/ergast/f1) | Race results, qualifying, driver and constructor standings |
| [FastF1](https://docs.fastf1.dev/) | Lap-level session data (tyres, pit stops, lap times) |

---

## Project phases

| # | Phase | Status |
|---|---|---|
| 1 | Data ingestion | Done |
| 2 | Feature engineering | Planned |
| 3 | Tyre degradation modelling | Planned |
| 4 | Race outcome prediction (XGBoost) | Planned |
| 5 | Explainability (SHAP) | Planned |
| 6 | Streamlit dashboard | Planned |
| 7 | Backtesting | Planned |

---

## Repository contents

| File | Purpose |
|---|---|
| `f1_data_pull.py` | Small smoke test that confirms FastF1 and Jolpica-F1 can be reached |
| `ingest_2026.py` | Phase 1: downloads and caches 2026 results, qualifying and standings as raw JSON |
| `F1_Forecast_Synopsis.pdf` | Project synopsis |

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
pip install fastf1 requests
```

### Run the data access test

```bash
python f1_data_pull.py
```

### Run Phase 1 ingestion

```bash
python ingest_2026.py
```

This downloads four datasets for every completed 2026 round into `data/raw/`:

- results
- qualifying
- driver standings (after that round)
- constructor standings (after that round)

The script is safe to re-run: files that already exist are skipped, so re-run it after each new race to pick up the latest round.

> **Avoiding leakage:** standings saved for round N are the standings *after* race N. To predict race N, use the standings from round N-1.

---

## Planned architecture (AWS)

- **Compute:** ECS on Fargate (scheduled ingestion task, prediction service, Streamlit dashboard service)
- **Storage:** S3 (raw and processed data), DynamoDB (predictions and SHAP values)
- **Containers:** ECR
- **Networking:** Application Load Balancer
- **Scheduling:** EventBridge
- **Observability:** CloudWatch
- **CI/CD:** GitHub Actions

---

## Author

Viyom Rana, B.Tech CSE, GLA University.
