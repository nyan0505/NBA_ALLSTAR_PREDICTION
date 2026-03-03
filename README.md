# NBA All-Star Prediction

This project builds a model to predict whether an NBA player will be selected as an **All-Star** based on in-season performance.  
The pipeline is: **data collection → preprocessing/alignment → labeling → feature generation → model training/evaluation**.

---

## 1) Data Collection (Notebook 01)

In `notebooks/01_build_dataset.ipynb`, player-season data is collected using the NBA API.

- Use `nba_api` to get all player IDs and fetch each player's career stats
- Save the combined table as `all_career.csv`
- Load `data/allstar.csv` and keep only the required columns: `first, last, team, year`

> Goal: build a clean player-season base table for labeling and modeling.

---

## 2) Preprocessing and Team Name Unification

To improve match accuracy during labeling, team abbreviations are standardized.

### 2-1. `all_career` cleanup
- Remove `TOT` rows  
  (`TOT` is an aggregate row for players who played on multiple teams in one season)
- Apply team code mapping (left → right)

```python
known_mappings = {
    "PHL": "PHI",
    "PHX": "PHO",
    "GOS": "GSW",
    "SAN": "SAS",
    "UTH": "UTA",
}
```

### 2-2. `allstar.csv` cleanup
- Fix inconsistent/legacy team codes

```python
allstar_team_mapping = {
    "BRK": "BKN",
    "CHO": "CHA",
    "WSB": "WAS",
}
```

---

## 3) All-Star Label Assignment

Create a binary `allstar` label by matching records between `all_career` and `allstar_df`.

- Add `first`, `last` columns to `all_career` (fetched from player ID)
- Match by `(first, last, year, team)`
- Assign `allstar = 1` if matched, otherwise `0`
- Save output: `data/all_career_1980_with_allstar.csv`

---

## 4) Name Standardization and Manual Exceptions (BLOCK 3)

Additional post-processing is applied to reduce false mismatches caused by name formatting differences.

- Remove suffixes (`Jr`, `III`, etc.)
- Fix blank/misaligned name fields
- Apply manual exceptions (e.g., Yao Ming, Jaren Jackson Jr, Steven Smith, etc.)
- Rebuild labels and validate unmatched cases
- Save:
  - `data/all_career_1980_with_allstar.csv`
  - `data/allstar_corrected.csv`

---

## 5) Feature Generation for Modeling

Based on the notebook design, the project aggregates player stats from **Oct 1 to Feb 1** (pre-All-Star cutoff) for each season.

- Prevents data leakage by excluding post-selection information
- Saves model-ready features to `data/historical_features.parquet`

---

## 6) Modeling

The environment includes `scikit-learn`, `imbalanced-learn`, and `xgboost` for classification with class imbalance handling.

Typical workflow:
1. Train/validation split
2. Class imbalance handling (weights or resampling)
3. Model training
4. Evaluation (Precision/Recall/F1, etc.)
5. Reproducible experiments using saved features/labels

---

## Main Outputs

- `data/allstar.csv`: cleaned All-Star reference data
- `data/allstar_corrected.csv`: corrected All-Star table after manual name fixes
- `data/all_career_1980_with_allstar.csv`: labeled player-season dataset
- `data/historical_features.parquet`: modeling feature dataset

---

## Environment

- Python
- Key packages:
  - `pandas`, `numpy`
  - `nba_api`
  - `unidecode`
  - `scikit-learn`, `imbalanced-learn`, `xgboost`
  - `pyarrow`, `joblib`, `matplotlib`, `tqdm`

---

## Recommended Run Order

1. Place `data/allstar.csv` in the `data/` directory
2. Run `notebooks/01_build_dataset.ipynb`
3. Verify generated label/feature files
4. Run training notebook/scripts for model fitting and evaluation