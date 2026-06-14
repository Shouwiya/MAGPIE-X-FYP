"""Input/output helpers shared by MAGPIE_X scripts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_table(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV/TSV/VCF-like table with delimiter inferred from suffix."""

    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path, **kwargs)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, **kwargs)
    if suffixes.endswith(".tsv") or suffixes.endswith(".txt") or suffixes.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, sep=None, engine="python", **kwargs)


def open_text(path: str | Path):
    """Open plain text or gzip-compressed text."""

    path = Path(path)
    if "".join(path.suffixes).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)

