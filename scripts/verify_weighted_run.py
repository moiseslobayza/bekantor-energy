"""Recompute saved validation-only metrics; verify control/artifact hashes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.forecasting.calendar_experiment import aggregate_metrics
from bekantor_energy.forecasting.weighted_experiment import weighted_validation_rows
from bekantor_energy.forecasting.windows import WindowSet


def verify(run_dir, model_dir):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))["thresholds_kwh"]
    sources = json.loads((run_dir / "baseline_sources.json").read_text(encoding="utf-8"))
    preserved = json.loads((run_dir / "preservation.json").read_text(encoding="utf-8"))
    if not preserved["verified"]:
        raise ValueError("Earlier artifact preservation was not verified")
    for path, expected_hash in preserved["original_artifacts_before"].items():
        if Path(path).name.startswith("test"):
            raise ValueError("Held-out source in preservation allowlist")
        if sha256(Path(path)) != expected_hash:
            raise ValueError(f"Previous artifact changed: {path}")
    rows, first_y, first_origins, hashes = [], None, None, {}
    for seed in config["control"]["seeds"]:
        for variant in ("mse", "weighted_mse"):
            path = (Path(sources[str(seed)]) if variant == "mse"
                    else model_dir / f"seed_{seed}" / variant / "validation_predictions.npz")
            with np.load(path, allow_pickle=False) as saved:
                actual, predicted, origins = saved["actual_kwh"], saved["predicted_kwh"], saved["origins"]
            if first_y is None:
                first_y, first_origins = actual, origins
            np.testing.assert_array_equal(actual, first_y)
            np.testing.assert_array_equal(origins, first_origins)
            windows = WindowSet(np.empty((len(actual), 168)), actual, pd.DatetimeIndex(origins), {})
            rows.extend(weighted_validation_rows(windows, predicted, thresholds, seed, variant))
            if variant == "weighted_mse":
                directory = path.parent
                report = json.loads((directory / "training.json").read_text(encoding="utf-8"))
                if report["checkpoint_monitor"] != "val_rmse" or report["parameters"] != 22904 or report["input_size"] != 312:
                    raise ValueError("Training contract changed")
                history = pd.read_csv(directory / "epochs.csv")
                if report["best_epoch"] != int(np.argmin(history.val_rmse.to_numpy()) + 1):
                    raise ValueError("Checkpoint was not selected by unweighted validation RMSE")
                objective = report["loss"]["config"]
                if (objective["normal_weight"], objective["peak_weight"]) != (1, 2):
                    raise ValueError("Loss weighting differs from predeclared 1/2")
                for name in ("best.keras", "training.json", "epochs.csv", "validation_predictions.npz",
                             "bundle/model.keras", "bundle/normalization.json", "bundle/predictor.json",
                             "bundle/training_objective.json"):
                    hashes[str(directory / name)] = sha256(directory / name)
    frame, aggregate, paired = aggregate_metrics(rows, ("mse", "weighted_mse"))
    for filename, expected in (("metrics_by_seed.csv", frame), ("metrics_aggregate.csv", aggregate), ("paired_deltas.csv", paired)):
        actual = pd.read_csv(run_dir / filename, dtype={"group": str})
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_names=False, rtol=1e-12, atol=1e-12)
        hashes[str(run_dir / filename)] = sha256(run_dir / filename)
    for path in (run_dir / "plots").glob("*.png"):
        hashes[str(path)] = sha256(path)
    for name in ("config.json", "thresholds.json", "windows.json", "normalization.json", "baseline_sources.json",
                 "fits.csv", "reproducibility.json", "global_guardrail.json", "global_tradeoffs.csv", "preservation.json"):
        hashes[str(run_dir / name)] = sha256(run_dir / name)
    result = {"status": "passed", "evaluation_partition": "validation", "new_weighted_fits": 3,
              "reused_controls": 3, "same_targets_and_origins": True, "saved_metrics_recomputed": True,
              "original_artifacts_unchanged": True, "unweighted_checkpoint_selection_verified": True,
              "rendering_source_sha256": sha256(Path("src/bekantor_energy/forecasting/weighted_plots.py")),
              "artifact_sha256": hashes}
    write_json(run_dir / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir, args.model_dir)
