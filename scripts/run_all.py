from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_features import run as build_features
from build_graph import run as build_graph
from calibrate_model import run as calibrate_model
from evaluate_model import run as evaluate_model
from prepare_data import run as prepare_data
from train_model import run as train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full MAGPIE_X dry-run workflow.")
    parser.add_argument("--data-config", default=str(ROOT / "configs" / "data.yaml"))
    parser.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    parser.add_argument("--train-config", default=str(ROOT / "configs" / "train.yaml"))
    parser.add_argument("--eval-config", default=str(ROOT / "configs" / "eval.yaml"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "toy"))
    parser.add_argument("--skip-tabular", action="store_true", help="Only train the GAT run.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    processed = output_root / "processed"
    features = output_root / "features"
    graphs = output_root / "graphs"
    experiments = output_root / "experiments"
    results = output_root / "results"

    prepare_data(args.data_config, processed, dry_run=True)
    build_features(args.data_config, processed, features)
    graph_path = build_graph(args.data_config, args.model_config, features, graphs)

    prediction_specs: list[str] = []
    if not args.skip_tabular:
        tabular = train_model(
            args.data_config,
            args.model_config,
            args.train_config,
            architecture="tabular",
            feature_dir=features,
            graph_path=graph_path,
            output_dir=experiments / "tabular_only",
        )
        prediction_specs.append(f"tabular_only={tabular.predictions_path}")

    gat = train_model(
        args.data_config,
        args.model_config,
        args.train_config,
        architecture="gat",
        feature_dir=features,
        graph_path=graph_path,
        output_dir=experiments / "tabular_graph_missense_expert",
    )
    calibrated = calibrate_model(args.eval_config, gat.predictions_path, experiments / "tabular_graph_missense_expert" / "calibration")
    prediction_specs.append(f"tabular_graph_missense_expert={calibrated}")
    evaluate_model(args.eval_config, prediction_specs, results)


if __name__ == "__main__":
    main()

