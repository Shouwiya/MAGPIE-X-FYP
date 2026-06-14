from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.eval.calibration import calibration_report
from magpie_x.eval.metrics import threshold_table
from magpie_x.eval.plots import save_calibration_plot, save_confusion_matrix, save_pr_curve, save_roc_curve
from magpie_x.utils.io import ensure_dir, load_yaml, read_table, write_dataframe, write_json
from magpie_x.utils.logging import setup_logger


def _choose_probability_column(df: pd.DataFrame, requested: str) -> str:
    if requested != "auto":
        return requested
    for col in ("probability_isotonic", "probability_temperature", "probability"):
        if col in df.columns:
            return col
    raise ValueError("No probability column found.")


def _evaluate_one(
    predictions: pd.DataFrame,
    output_dir: Path,
    run_name: str,
    probability_col: str,
    thresholds: list[float],
    n_bins: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, group in predictions.groupby("split"):
        report = calibration_report(group["label"].to_numpy(), group[probability_col].to_numpy(), n_bins=n_bins)
        report.update({"run_name": run_name, "split": split_name, "probability_column": probability_col})
        rows.append(report)
        if split_name == "test":
            write_dataframe(threshold_table(group["label"].to_numpy(), group[probability_col].to_numpy(), thresholds), output_dir / f"{run_name}_threshold_table.csv")
            y_true = group["label"].to_numpy()
            y_prob = group[probability_col].to_numpy()
            if len(np.unique(y_true)) == 2:
                plot_dir = ensure_dir(output_dir / "plots" / run_name)
                save_roc_curve(y_true, y_prob, plot_dir)
                save_pr_curve(y_true, y_prob, plot_dir)
                save_calibration_plot(y_true, y_prob, plot_dir, n_bins=n_bins)
                save_confusion_matrix(y_true, y_prob, plot_dir)
    return pd.DataFrame(rows)


def run(
    eval_config_path: str | Path = ROOT / "configs" / "eval.yaml",
    predictions: list[str] | None = None,
    output_dir: str | Path | None = None,
    probability_column: str = "auto",
) -> Path:
    cfg = load_yaml(eval_config_path)
    logger = setup_logger("magpie_x.evaluate_model")
    out = ensure_dir(output_dir or cfg.get("output_dir", ROOT / "outputs" / "toy" / "results"))
    pred_specs = predictions or [str(ROOT / "outputs" / "toy" / "experiments" / "default" / "predictions.csv")]
    thresholds = [float(x) for x in cfg.get("thresholds", [0.5])]
    n_bins = int(cfg.get("calibration", {}).get("n_bins", 10))
    all_metrics: list[pd.DataFrame] = []

    for spec in pred_specs:
        if "=" in spec:
            run_name, path_text = spec.split("=", 1)
        else:
            path = Path(spec)
            run_name, path_text = path.parent.name, spec
        df = read_table(path_text)
        col = _choose_probability_column(df, probability_column)
        all_metrics.append(_evaluate_one(df, out, run_name, col, thresholds, n_bins))

    metrics = pd.concat(all_metrics, ignore_index=True)
    metrics_path = out / "metrics_by_split.csv"
    write_dataframe(metrics, metrics_path)
    ablation = metrics[metrics["split"] == "test"].copy()
    configured = pd.DataFrame({"configured_ablation": cfg.get("ablations", [])})
    write_dataframe(ablation, out / "ablation_results.csv")
    write_dataframe(configured, out / "ablation_plan.csv")
    write_json({"n_runs": len(pred_specs), "probability_column": probability_column}, out / "evaluation_metadata.json")
    logger.info("Evaluation complete. Metrics: %s", metrics_path)
    return metrics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MAGPIE_X predictions and save publication-ready outputs.")
    parser.add_argument("--eval-config", default=str(ROOT / "configs" / "eval.yaml"))
    parser.add_argument("--predictions", nargs="+", required=True, help="Prediction CSVs, optionally name=path.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--probability-column", default="auto")
    args = parser.parse_args()
    run(args.eval_config, args.predictions, args.output_dir, args.probability_column)


if __name__ == "__main__":
    main()

