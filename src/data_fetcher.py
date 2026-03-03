"""
data_fetcher.py – Utilities for fetching NBA game-log data and All-Star labels.

All network calls are cached to disk (Parquet) so that repeated notebook runs
do not hit the NBA API unnecessarily.
"""

from __future__ import annotations
from pathlib import Path
import time
import warnings
from nba_api.stats.static import players
from unidecode import unidecode
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import players as nba_players_static
from unidecode import unidecode

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

# Mapping of team abbreviation variants → canonical form used in training data
TEAM_ABBR_MAP: dict[str, str] = {
    "PHL": "PHI",
    "PHX": "PHO",
    "WAS": "WSB",
    "CHA": "CHO",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "NJN": "BKN",
    "GOS": "GSW",
    "UTH": "UTA",
}

# Numeric stat columns available from LeagueGameLog player endpoint
STAT_COLS = [
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
]


# ---------------------------------------------------------------------------
# Game-log fetching with disk cache
# ---------------------------------------------------------------------------

def fetch_season_gamelogs(
    season: str,
    cache_dir: Path | str = _DEFAULT_CACHE_DIR,
    retries: int = 5,
    initial_sleep: float = 1.0,
) -> pd.DataFrame:
    """Return a DataFrame of all *player* game logs for *season*.

    Results are cached to ``<cache_dir>/gamelogs_<season>.parquet``.

    Parameters
    ----------
    season : str
        NBA season string, e.g. ``'2019-20'``.
    cache_dir : Path | str
        Directory to store cached parquet files.
    retries : int
        Number of retry attempts on network errors.
    initial_sleep : float
        Initial back-off time in seconds (doubles on each retry).

    Returns
    -------
    pd.DataFrame
        Raw game-log rows with at minimum the columns:
        PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GAME_DATE, plus stat cols.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"gamelogs_{season}.parquet"

    if cache_path.exists():
        return pd.read_parquet(cache_path)

    sleep_time = initial_sleep
    for attempt in range(retries):
        try:
            gl = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
            )
            df = gl.get_data_frames()[0]
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            df.to_parquet(cache_path, index=False)
            return df
        except Exception as exc:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(sleep_time)
                sleep_time *= 2
            else:
                raise RuntimeError(
                    f"Failed to fetch game logs for season {season} after {retries} attempts."
                ) from exc

    raise RuntimeError(f"Unexpected error fetching season {season}")


# ---------------------------------------------------------------------------
# Player name fetching
# ---------------------------------------------------------------------------

def fetch_player_name(player_id, retries=5, initial_sleep_time=1):
    sleep_time = initial_sleep_time
    for attempt in range(retries):
        try:
            player_info = players.find_player_by_id(int(player_id))
            if player_info:
                return {
                    "first_name": unidecode(player_info.get("first_name", "")).strip(),
                    "last_name": unidecode(player_info.get("last_name", "")).strip(),
                }
            return None
        except Exception as e:
            print(f"Error fetching player_id {player_id}: {e}")
            if attempt < retries - 1:
                time.sleep(sleep_time)
                sleep_time += 1
            else:
                print(f"Failed to fetch player_id {player_id} after {retries} attempts.")
    return None

# ---------------------------------------------------------------------------
# All-Star label loading and player-ID mapping
# ---------------------------------------------------------------------------

def load_allstar_labels(
    allstar_csv: Path | str,
) -> pd.DataFrame:
    """Load All-Star labels and add a PLAYER_ID column.

    The Kaggle dataset (ethankeyes/nba-all-star-players-and-stats-1980-2022)
    identifies players by ``first`` / ``last`` name and ``year`` (season start
    year, e.g. 2019 for the 2019-20 season).  This function resolves each row
    to a ``PLAYER_ID`` using the nba_api static player registry.

    Rows that cannot be matched are kept but ``PLAYER_ID`` will be ``NaN``.

    Parameters
    ----------
    allstar_csv : Path | str
        Path to the ``allstar.csv`` label file.

    Returns
    -------
    pd.DataFrame
        Columns: PLAYER_ID (int, nullable), first, last, year (int), team.
    """
    df = pd.read_csv(allstar_csv)

    # Ensure expected columns are present
    required = {"first", "last", "year"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"allstar.csv must contain columns {required}; found {list(df.columns)}"
        )

    name_map = _build_name_to_id_map()

    player_ids: list[int | None] = []
    for _, row in df.iterrows():
        key = _normalize_name(str(row["first"]), str(row["last"]))
        player_ids.append(name_map.get(key))

    df["PLAYER_ID"] = player_ids
    df["year"] = df["year"].astype(int)
    return df


def _build_name_to_id_map() -> dict[str, int]:
    """Build a normalised-name → PLAYER_ID mapping from the nba_api static list."""
    name_map: dict[str, int] = {}
    for p in nba_players_static.get_players():
        key = _normalize_name(p["first_name"], p["last_name"])
        name_map[key] = p["id"]
    return name_map


def _normalize_name(first: str, last: str) -> str:
    """Return a lowercase ASCII key for fuzzy name matching."""
    return unidecode(f"{first.strip().lower()} {last.strip().lower()}")