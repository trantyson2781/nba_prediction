"""Read the frozen parquet cache produced by scripts/fetch_cache.py."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"


def load(endpoint: str, season: str) -> pd.DataFrame:
    """Load one cached endpoint/season, e.g. load("games", "2023-24")."""
    path = CACHE_DIR / endpoint / f"{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No cache at {path}. Run `python scripts/fetch_cache.py` first.")
    return pd.read_parquet(path)


def load_many(endpoint: str, seasons) -> pd.DataFrame:
    """Concatenate one endpoint across several seasons (SEASON column preserved)."""
    return pd.concat([load(endpoint, s) for s in seasons], ignore_index=True)
