"""Dataset splitting strategies for rigorous pathogenicity evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from magpie_x.utils.io import ensure_dir, write_json


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    mode: str

    def to_dict(self) -> dict[str, list[int] | str]:
        return {
            "mode": self.mode,
            "train": self.train.astype(int).tolist(),
            "val": self.val.astype(int).tolist(),
            "test": self.test.astype(int).tolist(),
        }


def _val_from_train(train_idx: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 2 else 1
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def temporal_split(
    df: pd.DataFrame,
    date_col: str = "submission_date",
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    cutoff: str | None = None,
    seed: int = 13,
) -> SplitIndices:
    dates = pd.to_datetime(df[date_col], errors="coerce").fillna(pd.Timestamp("2000-01-01"))
    order = np.argsort(dates.to_numpy())
    if cutoff:
        cutoff_ts = pd.Timestamp(cutoff)
        test = np.where(dates >= cutoff_ts)[0]
        train_pool = np.where(dates < cutoff_ts)[0]
    else:
        n_test = max(1, int(round(len(df) * test_fraction)))
        test = np.sort(order[-n_test:])
        train_pool = np.sort(order[:-n_test])
    train, val = _val_from_train(train_pool, val_fraction, seed)
    return SplitIndices(train=train, val=val, test=np.sort(test), mode="temporal")


def gene_held_out_split(
    df: pd.DataFrame,
    gene_col: str = "gene",
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 13,
) -> SplitIndices:
    rng = np.random.default_rng(seed)
    genes = np.array(sorted(df[gene_col].astype(str).unique()))
    rng.shuffle(genes)
    n_test_genes = max(1, int(round(len(genes) * test_fraction)))
    test_genes = set(genes[:n_test_genes])
    test = df.index[df[gene_col].astype(str).isin(test_genes)].to_numpy()
    train_pool = df.index[~df[gene_col].astype(str).isin(test_genes)].to_numpy()
    train, val = _val_from_train(train_pool, val_fraction, seed)
    return SplitIndices(train=np.sort(train), val=val, test=np.sort(test), mode="gene_held_out")


def rare_variant_split(
    df: pd.DataFrame,
    af_col: str = "gnomad_af",
    rare_threshold: float = 0.01,
    val_fraction: float = 0.2,
    seed: int = 13,
) -> SplitIndices:
    af = pd.to_numeric(df.get(af_col, pd.Series(np.nan, index=df.index)), errors="coerce").fillna(0.0)
    test = df.index[af < rare_threshold].to_numpy()
    if len(test) == 0:
        test = df.nsmallest(max(1, len(df) // 5), af_col).index.to_numpy()
    train_pool = df.index.difference(test).to_numpy()
    if len(train_pool) < 3:
        train_pool = df.index.to_numpy()
    train, val = _val_from_train(np.array(train_pool), val_fraction, seed)
    return SplitIndices(train=np.sort(train), val=val, test=np.sort(test), mode="rare")


def make_split(df: pd.DataFrame, config: dict, seed: int = 13) -> SplitIndices:
    mode = str(config.get("mode", "temporal"))
    if mode == "temporal":
        return temporal_split(
            df,
            test_fraction=float(config.get("test_fraction", 0.2)),
            val_fraction=float(config.get("val_fraction", 0.2)),
            cutoff=config.get("temporal_cutoff"),
            seed=seed,
        )
    if mode in {"gene", "gene_held_out"}:
        return gene_held_out_split(
            df,
            test_fraction=float(config.get("heldout_gene_fraction", config.get("test_fraction", 0.2))),
            val_fraction=float(config.get("val_fraction", 0.2)),
            seed=seed,
        )
    if mode == "rare":
        return rare_variant_split(
            df,
            rare_threshold=float(config.get("rare_af_threshold", 0.01)),
            val_fraction=float(config.get("val_fraction", 0.2)),
            seed=seed,
        )
    raise ValueError(f"Unknown split mode: {mode}")


def save_split(split: SplitIndices, output_dir: str | Path) -> None:
    output_dir = ensure_dir(output_dir)
    write_json(split.to_dict(), output_dir / "split_indices.json")
    for name in ("train", "val", "test"):
        pd.DataFrame({"row_index": getattr(split, name)}).to_csv(output_dir / f"{name}_indices.csv", index=False)

