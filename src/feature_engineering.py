"""
feature_engineering.py – Build feature DataFrames from player game logs.

Key design decisions
--------------------
* Features are computed by aggregating game logs from Oct 1 through Feb 1
  of each season.  This prevents label leakage from using full-season totals.
* Missing percentage stats (FG_PCT, FG3_PCT, FT_PCT) are imputed per player
  using a year-sorted, PLAYER_ID-grouped approach instead of the leaky index±1
  method used in the original notebook.
* The returned DataFrame contains one row per (PLAYER_ID, season_year) and
  is ready to feed into the training pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_fetcher import (
    STAT_COLS,
    TEAM_ABBR_MAP,
    fetch_season_gamelogs,
    load_allstar_labels,
)

# ---------------------------------------------------------------------------
# Per-season feature computation
# ---------------------------------------------------------------------------

_AGG_SUM = ["MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
            "GP"]
_AGG_FIRST = ["PLAYER_NAME", "TEAM_ABBREVIATION"]

# Percentage stats are recomputed from totals (more accurate than mean of
# per-game percentages).
_PCT_COLS = {"FG_PCT": ("FGM", "FGA"),
             "FG3_PCT": ("FG3M", "FG3A"),
             "FT_PCT": ("FTM", "FTA")}


def _season_to_cutoff_year(season: str) -> int:
    """Return the calendar year of the Feb-1 cutoff for *season*.

    '2019-20' → 2020  (Oct 2019 – Feb 1 **2020**)
    """
    return int(season.split("-")[0]) + 1


def _season_to_start_year(season: str) -> int:
    """Return the start calendar year of *season*. '2019-20' → 2019."""
    return int(season.split("-")[0])


def build_features_for_season(
    season: str,
    cache_dir: Path | str = "data/cache",
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Return per-player feature rows for *season* using the Oct-1 → cutoff window.

    Parameters
    ----------
    season : str
        NBA season string, e.g. ``'2019-20'``.
    cache_dir : Path | str
        Directory used for game-log cache files.
    end_date : str | pd.Timestamp | None
        Inclusive cutoff date. If None, defaults to Feb-1 of cutoff year.

    Returns
    -------
    pd.DataFrame
        One row per player. Columns include PLAYER_ID, PLAYER_NAME, team,
        year, PLAYER_AGE, GP, and aggregated stat columns.
    """
    start_year = _season_to_start_year(season)
    cutoff_year = _season_to_cutoff_year(season)

    start_date = pd.Timestamp(f"{start_year}-10-01")
    cutoff_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(f"{cutoff_year}-02-01")

    gl = fetch_season_gamelogs(season, cache_dir=cache_dir)

    gl = gl.copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"], errors="coerce")
    gl = gl.dropna(subset=["GAME_DATE"])

    # Add GP counter (each row = 1 game)
    gl["GP"] = 1

    # Filter to Oct-1 → cutoff_date window
    mask = (gl["GAME_DATE"] >= start_date) & (gl["GAME_DATE"] <= cutoff_date)
    gl = gl.loc[mask]

    if gl.empty:
        return pd.DataFrame()

    # Aggregate
    agg_dict: dict[str, str | list] = {col: "sum" for col in _AGG_SUM}
    agg_dict["PLAYER_NAME"] = "first"
    agg_dict["TEAM_ABBREVIATION"] = "last"

    grouped = gl.groupby("PLAYER_ID").agg(agg_dict).reset_index()

    # Recompute percentage stats from totals
    for pct_col, (num_col, denom_col) in _PCT_COLS.items():
        denom = grouped[denom_col].replace(0, np.nan)
        grouped[pct_col] = grouped[num_col] / denom

    grouped.rename(columns={"TEAM_ABBREVIATION": "team"}, inplace=True)
    grouped["team"] = grouped["team"].replace(TEAM_ABBR_MAP)

    grouped["year"] = start_year
    grouped["PLAYER_AGE"] = np.nan

    return grouped

# ---------------------------------------------------------------------------
# Historical dataset construction
# ---------------------------------------------------------------------------

def build_historical_dataset(
    seasons: Sequence[str],
    allstar_csv: Path | str,
    cache_dir: Path | str = "data/cache",
) -> pd.DataFrame:
    """Build the labelled training dataset for all *seasons*.

    For each season:
    1. Compute per-player features from game logs up to Feb 1.
    2. Join with All-Star labels on (PLAYER_ID, year) where possible,
       falling back to (normalised name, year) for rows without PLAYER_ID in
       the label file.

    Parameters
    ----------
    seasons : Sequence[str]
        Ordered list of season strings, e.g. ``['2008-09', ..., '2022-23']``.
    allstar_csv : Path | str
        Path to the All-Star label CSV.
    cache_dir : Path | str
        Game-log cache directory.

    Returns
    -------
    pd.DataFrame
        Combined feature + label DataFrame.  Label column is ``allstar``
        (0 or 1).
    """
    labels = load_allstar_labels(allstar_csv)

    season_frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"  Building features for {season} …", flush=True)
        feats = build_features_for_season(season, cache_dir=cache_dir)
        if feats.empty:
            print(f"    → no data, skipping.", flush=True)
            continue
        season_frames.append(feats)

    if not season_frames:
        raise ValueError("No feature data was produced for any of the requested seasons.")

    df = pd.concat(season_frames, ignore_index=True)

    # ---- Attach All-Star labels ----------------------------------------
    # Build a lookup: (PLAYER_ID, year) → 1
    labels_by_id = labels.dropna(subset=["PLAYER_ID"]).copy()
    labels_by_id["PLAYER_ID"] = labels_by_id["PLAYER_ID"].astype(int)
    id_year_set = set(
        zip(labels_by_id["PLAYER_ID"].astype(int), labels_by_id["year"].astype(int))
    )

    def _mark_allstar(row: pd.Series) -> int:
        pid = int(row["PLAYER_ID"]) if pd.notna(row["PLAYER_ID"]) else None
        yr = int(row["year"])
        if pid is not None and (pid, yr) in id_year_set:
            return 1
        return 0

    df["allstar"] = df.apply(_mark_allstar, axis=1)

    # ---- PLAYER_ID-grouped, year-sorted missing value imputation ---------
    df = _impute_missing_values(df)

    return df


# ---------------------------------------------------------------------------
# Missing value imputation (PLAYER_ID-grouped, year-sorted)
# ---------------------------------------------------------------------------

_IMPUTE_COLS = ["FG_PCT", "FG3_PCT", "FT_PCT", "PLAYER_AGE"]


def _impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing stat values grouped by PLAYER_ID, sorted by year.

    For each player, the series of annual values is interpolated linearly
    (forward-fill then backward-fill) within that player's own history.
    Rows where all imputation attempts fail are dropped.
    """
    df = df.sort_values(["PLAYER_ID", "year"]).copy()

    for col in _IMPUTE_COLS:
        if col not in df.columns:
            continue
        df[col] = (
            df.groupby("PLAYER_ID")[col]
            .transform(lambda s: s.interpolate(method="linear")
                                  .ffill()
                                  .bfill())
        )

    # Drop rows still containing NaN in required columns (excl. PLAYER_AGE
    # which we allow to stay NaN and let the imputer handle in the pipeline)
    required_cols = [c for c in _IMPUTE_COLS if c != "PLAYER_AGE" and c in df.columns]
    df = df.dropna(subset=required_cols)

    return df
