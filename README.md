# NBA All-Star Prediction

A reproducible machine-learning pipeline that predicts which NBA players will
be selected as All-Stars, using **pre-All-Star-selection** game-log statistics
to eliminate label leakage.

---

## Project structure

```
NBA_ALLSTAR_PREDICTION/
├── notebooks/
│   ├── 01_build_dataset.ipynb   # fetch game logs, build labelled dataset
│   ├── 02_train.ipynb           # season-based split, train pipeline, evaluate
│   └── 03_predict_2025_26.ipynb # predict All-Stars for the 2025-26 season
├── src/
│   ├── data_fetcher.py          # NBA API helpers + caching
│   ├── feature_engineering.py  # per-season feature computation (Oct-1 → Feb-1)
│   └── evaluation.py           # PR-AUC, ROC-AUC, Precision@K, Recall@K
├── data/                        # generated datasets (git-ignored)
├── models/                      # saved pipeline artifacts (git-ignored)
├── requirements.txt
└── README.md
```

---

## Design decisions

| Concern | Original approach | New approach |
|---|---|---|
| Feature window | Full-season career totals | Oct 1 → **Feb 1** per season |
| Train/test split | Random row split | Season-based split |
| Scaler fit | Entire dataset | Train fold only (inside pipeline) |
| SMOTE | Entire dataset | Train fold only (imblearn Pipeline) |
| Missing value imputation | Index ±1 neighbours | PLAYER_ID-grouped, year-sorted interpolation |
| Preprocessing consistency | Ad-hoc per notebook | Single saved `pipeline.joblib` |
| Primary metrics | Accuracy, ROC-AUC | PR-AUC, Precision@24, Recall@24, ROC-AUC |
| Prediction target | 2024-25 | **2025-26** |

---

## Quickstart

### 1. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Obtain All-Star labels

Download `allstar.csv` from Kaggle:
<https://www.kaggle.com/datasets/ethankeyes/nba-all-star-players-and-stats-1980-2022>

Place the file at `data/allstar.csv`.  The build notebook will automatically
append labels for the 2022-23 and 2023-24 seasons.

### 3. Build the historical dataset

```bash
jupyter nbconvert --to notebook --execute notebooks/01_build_dataset.ipynb
```

This fetches game logs from the NBA API (results are cached to
`data/cache/*.parquet`), aggregates per-player stats from Oct 1 through Feb 1
of each season, attaches All-Star labels, and writes
`data/historical_features.parquet`.

> **Note**: The first run makes many API calls and may take 20–40 minutes.
> Subsequent runs read from the cache and complete in seconds.

### 4. Train the pipeline

```bash
jupyter nbconvert --to notebook --execute notebooks/02_train.ipynb
```

Trains an `imblearn.Pipeline` (ColumnTransformer → SMOTE → XGBClassifier)
using a season-based split, evaluates on test seasons, and saves the
fitted pipeline to `models/pipeline.joblib`.

### 5. Predict 2025-26 All-Stars

```bash
jupyter nbconvert --to notebook --execute notebooks/03_predict_2025_26.ipynb
```

Fetches 2025-26 game logs up to Feb 1, 2026, runs the saved pipeline, and
outputs a ranked list of the top 24 predicted All-Stars.

---

## Evaluation metrics

| Metric | Description |
|---|---|
| PR-AUC | Average Precision across all thresholds (imbalance-robust) |
| ROC-AUC | Secondary ranking metric |
| Precision@24 | Fraction of top-24 predictions that are actual All-Stars |
| Recall@24 | Fraction of actual All-Stars captured in top-24 |

---

## Reproducibility

All random processes use `random_state=42`.  Game-log data is cached to
`data/cache/` so re-runs produce identical results without re-hitting the API.

---

## Legacy notebook

The original single-notebook approach is preserved as `nba.ipynb` for reference.
