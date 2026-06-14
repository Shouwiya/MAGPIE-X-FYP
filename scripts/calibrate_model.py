from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from magpie_x.eval.calibration import calibration_report
from magpie_x.models.calibrator import IsotonicCalibrator, TemperatureScaler
from magpie_x.utils.io import ensure_dir, load_yaml, read_table, write_dataframe, write_json
from magpie_x.utils.logging import setup_logger


def run(
    eval_config_path: str | Path = ROOT / "configs" / "eval.yaml",
    predictions_path: str | Path = ROOT / "outputs" / "toy" / "experiments" / "default" / "predictions.csv",
    output_dir: str | Path | None = None,
) -> Path:
    cfg = load_yaml(eval_config_path)
    logger = setup_logger("magpie_x.calibrate_model")
    pred_path = Path(predictions_path)
    out = ensure_dir(output_dir or pred_path.parent / "calibration")
    predictions = read_table(pred_path)
    val = predictions[predictions["split"] == "val"].copy()
    if val.empty:
        raise ValueError("Predictions must contain a validation split for calibration.")

    temp = TemperatureScaler().fit(val["logit"].to_numpy(), val["label"].to_numpy())
    iso = IsotonicCalibrator().fit(val["probability"].to_numpy(), val["label"].to_numpy())
    calibrated = predictions.copy()
    calibrated["probability_temperature"] = temp.predict_proba(calibrated["logit"].to_numpy())
    calibrated["probability_isotonic"] = iso.predict_proba(calibrated["probability"].to_numpy())
    output_path = out / "calibrated_predictions.csv"
    write_dataframe(calibrated, output_path)
    temp.save(out / "temperature_scaler.pkl")
    iso.save(out / "isotonic_calibrator.pkl")

    n_bins = int(cfg.get("calibration", {}).get("n_bins", 10))
    rows: list[dict[str, object]] = []
    for split_name, group in calibrated.groupby("split"):
        for method, col in {
            "uncalibrated": "probability",
            "temperature": "probability_temperature",
            "isotonic": "probability_isotonic",
        }.items():
            report = calibration_report(group["label"].to_numpy(), group[col].to_numpy(), n_bins=n_bins)
            report.update({"split": split_name, "method": method})
            rows.append(report)
    metrics = pd.DataFrame(rows)
    write_dataframe(metrics, out / "calibration_metrics.csv")
    write_json({"temperature": temp.temperature, "n_validation": int(len(val))}, out / "calibration_metadata.json")
    logger.info("Calibration complete. Calibrated predictions: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MAGPIE_X probabilities.")
    parser.add_argument("--eval-config", default=str(ROOT / "configs" / "eval.yaml"))
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.eval_config, args.predictions, args.output_dir)


if __name__ == "__main__":
    main()

