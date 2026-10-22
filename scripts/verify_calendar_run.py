"""Check saved validation predictions/aggregates and original artifact preservation.

No model inference and no held-out data access. Run from the project root.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.forecasting.calendar_experiment import aggregate_metrics, validation_rows
from bekantor_energy.forecasting.windows import WindowSet


def verify(run_dir, model_dir, reference_run):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))["thresholds_kwh"]
    rows, first_actual, first_origins, hashes = [], None, None, {}
    for seed in config["seeds"]:
        for variant in ("reference", "calendar"):
            directory = model_dir / f"seed_{seed}" / variant
            with np.load(directory / "validation_predictions.npz", allow_pickle=False) as saved:
                actual, predicted, origins = saved["actual_kwh"], saved["predicted_kwh"], saved["origins"]
            if first_actual is None:
                first_actual, first_origins = actual, origins
            np.testing.assert_array_equal(actual, first_actual)
            np.testing.assert_array_equal(origins, first_origins)
            windows = WindowSet(np.empty((len(actual), 168)), actual, pd.DatetimeIndex(origins), {})
            rows.extend(validation_rows(windows, predicted, thresholds, seed, variant))
            for name in ("best.keras", "training.json", "epochs.csv", "validation_predictions.npz",
                         "bundle/model.keras", "bundle/predictor.json", "bundle/normalization.json"):
                hashes[str(directory / name)] = sha256(directory / name)
    frame, aggregate, paired = aggregate_metrics(rows)
    for filename, expected in (("metrics_by_seed.csv", frame), ("metrics_aggregate.csv", aggregate),
                               ("paired_deltas.csv", paired)):
        actual_frame = pd.read_csv(run_dir / filename, dtype={"group": str})
        # CSV does not preserve the pivot's column-axis name; values/labels must match.
        pd.testing.assert_frame_equal(actual_frame, expected, check_dtype=False, check_names=False,
                                      rtol=1e-12, atol=1e-12)
    preserved = json.loads((run_dir / "preservation.json").read_text(encoding="utf-8"))
    assert preserved["verified"]
    for path, expected_hash in preserved["original_artifacts_before"].items():
        assert sha256(Path(path)) == expected_hash, f"Original artifact changed: {path}"
    # The first seed repeats the original training, without reopening its held-out results.
    differences = None
    if 2026 in config["seeds"]:
        selection = json.loads((reference_run / "selection.json").read_text(encoding="utf-8"))
        original_scores = selection["validation_comparison"]["selected_mlp"]
        repeated = frame[(frame.seed == 2026) & (frame.variant == "reference") & (frame.scope == "global")]
        differences = {metric: float(repeated.loc[repeated.metric == metric, "value"].iloc[0] - original_scores[metric])
                       for metric in ("mae_kwh", "rmse_kwh", "bias_kwh")}
        assert all(abs(value) < 1e-6 for value in differences.values()), differences
    for path in sorted((run_dir / "plots").glob("*.png")):
        hashes[str(path)] = sha256(path)
    for name in ("config.json", "thresholds.json", "normalization.json", "windows.json", "fits.csv",
                 "metrics_by_seed.csv", "metrics_aggregate.csv", "paired_deltas.csv", "reproducibility.json"):
        hashes[str(run_dir / name)] = sha256(run_dir / name)
    result = {"status": "passed", "validation_only": True, "paired_fits": len(config["seeds"]) * 2,
              "identical_targets_and_origins": True, "recomputed_saved_metrics": True,
              "original_artifacts_unchanged": True, "repeated_seed_2026_minus_original_validation": differences,
              "artifact_sha256": hashes}
    write_json(run_dir / "verification.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, default=Path("experiments/phase2/reference-20261022-cpu"))
    args = parser.parse_args()
    verify(args.run_dir, args.model_dir, args.reference_run)
