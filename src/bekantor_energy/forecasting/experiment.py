"""Two-stage experiment: train/select with validation, then one locked test pass."""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from ..data.acquisition import sha256, write_json
from .baselines import seasonal_persistence
from .config import ExperimentConfig
from .metrics import demand_metrics
from .mlp import MLPDemandPredictor, configure_tensorflow, predict_windows, train_one
from .normalization import DemandNormalizer
from .plots import comparison_figures, forecast_example, training_figures
from .windows import make_windows, read_hourly


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _checked_partition(directory: Path, name: str, manifest: dict):
    path = directory / f"{name}.csv"
    if sha256(path) != manifest["partitions"][name]["sha256"]:
        raise ValueError(f"Phase 1 {name} hash mismatch; regenerate or preserve the original splits")
    frame = read_hourly(path)
    expected = manifest["partitions"][name]
    if len(frame) != expected["hours"] or frame.index[0].isoformat() != expected["start"] or frame.index[-1].isoformat() != expected["end"]:
        raise ValueError(f"{name} partition bounds differ from Phase 1 manifest")
    return frame


def _baseline_predictions(windows, horizon):
    return {"seasonal_24h": seasonal_persistence(windows.x, lag=24, horizon=horizon),
            "seasonal_168h": seasonal_persistence(windows.x, lag=168, horizon=horizon)}


def _save_predictions(path, windows, predictions):
    starts = windows.origins.to_numpy().astype("datetime64[ns]")
    target_times = starts[:, None] + np.arange(windows.y.shape[1])[None, :] * np.timedelta64(1, "h")
    frame = pd.DataFrame({"origin": np.repeat(starts, windows.y.shape[1]),
                          "target_timestamp": target_times.ravel(),
                          "horizon_hour": np.tile(np.arange(1, windows.y.shape[1] + 1), len(starts)),
                          "actual_kwh": windows.y.ravel(),
                          **{name + "_kwh": values.ravel() for name, values in predictions.items()}})
    frame.to_csv(path, index=False)


def train_experiment(data_dir: Path, config: ExperimentConfig, run_dir: Path, model_dir: Path) -> dict:
    data_dir, run_dir, model_dir = Path(data_dir), Path(run_dir), Path(model_dir)
    if run_dir.exists() or model_dir.exists():
        raise ValueError("Run/model directory already exists; use new paths to preserve experiment history")
    manifest = _json(data_dir / "manifest.json")
    # Explicitly no test.csv loading or inference in this stage.
    train = _checked_partition(data_dir, "train", manifest)
    validation = _checked_partition(data_dir, "validation", manifest)
    kwargs = {"lookback": config.lookback, "horizon": config.horizon, "stride": config.stride}
    windows = {"train": make_windows(train, **kwargs),
               "validation": make_windows(validation, history=train, **kwargs)}
    if any(not len(w.x) for w in windows.values()):
        raise ValueError("No eligible train/validation windows; inspect quality and window settings")
    normalizer = DemandNormalizer.fit(train)
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "data_manifest.json", manifest)
    normalizer.save(run_dir / "normalization.json")
    write_json(run_dir / "windows.json", {name: w.report for name, w in windows.items()})
    tf = configure_tensorflow(config.seed, config.threads)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
                "logical_cpu_count": os.cpu_count(), "devices": [d.name for d in tf.config.list_logical_devices()],
                "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "determinism": {"seed": config.seed, "oneDNN": False, "op_determinism": True,
                                "intra_threads": config.threads, "inter_threads": 1,
                                "training_shuffle": "seeded within training windows only"},
                "source_hashes": {str(p.relative_to(Path(__file__).resolve().parents[1])): sha256(p)
                                   for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py"))},
                "selection_rule": "minimum validation RMSE in kWh; tie: fewer parameters, then name",
                "test_policy": "test.csv not read until selected bundle is frozen"}
    write_json(run_dir / "reproducibility.json", metadata)
    reports = {}
    for architecture in config.architectures:
        print(f"Training {architecture.name}: {architecture.hidden_units}", flush=True)
        model, report = train_one(windows["train"], windows["validation"], normalizer,
                                   config, architecture, model_dir / architecture.name)
        reports[architecture.name] = report
        del model
        print(f"Validation {architecture.name}: RMSE={report['validation']['rmse_kwh']:.6f} kWh", flush=True)
    selected_name = min(reports, key=lambda name: (reports[name]["validation"]["rmse_kwh"],
                                                   reports[name]["parameters"], name))
    architecture = next(a for a in config.architectures if a.name == selected_name)
    predictor = MLPDemandPredictor(config, architecture, model_dir / selected_name,
                                   model_id=f"{run_dir.name}:{selected_name}")
    predictor.model = tf.keras.models.load_model(model_dir / selected_name / "best.keras", compile=False)
    predictor.normalizer = normalizer
    bundle = model_dir / "selected"
    predictor.save(bundle)
    predictions = {"selected_mlp": predict_windows(predictor.model, windows["validation"].x, normalizer),
                   **_baseline_predictions(windows["validation"], config.horizon)}
    scores = {name: demand_metrics(windows["validation"].y, values) for name, values in predictions.items()}
    selection = {"selected_architecture": selected_name, "selection_metric": "validation.rmse_kwh",
                 "config_sha256": sha256(run_dir / "config.json"),
                 "selected_bundle": os.path.relpath(bundle.resolve(), run_dir.resolve()),
                 "bundle_sha256": {name: sha256(bundle / name) for name in ("model.keras", "normalization.json", "predictor.json")},
                 "architectures": {name: {key: r[key] for key in ("architecture", "parameters", "epochs_run",
                                                               "best_epoch", "train", "validation", "training_seconds")}
                                   for name, r in reports.items()}, "validation_comparison": scores}
    training_figures(reports, normalizer.scale_kwh, run_dir / "plots")
    comparison_figures(scores, run_dir / "plots", "validation")
    forecast_example(windows["validation"].origins, windows["validation"].y, predictions,
                      run_dir / "plots", "validation")
    _save_predictions(run_dir / "validation_predictions.csv", windows["validation"], predictions)
    # Freeze selection last, after complete validation outputs.
    write_json(run_dir / "selection.json", selection)
    print(f"Selected {selected_name}; test has not been read.", flush=True)
    return selection


def evaluate_test_once(data_dir: Path, run_dir: Path) -> dict:
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    selection = _json(run_dir / "selection.json")
    if sha256(run_dir / "config.json") != selection["config_sha256"]:
        raise ValueError("Experiment configuration changed after validation selection")
    config = ExperimentConfig.load(run_dir / "config.json")
    bundle = (run_dir / selection["selected_bundle"]).resolve()
    for name, expected in selection["bundle_sha256"].items():
        if sha256(bundle / name) != expected:
            raise ValueError("Selected model/normalizer metadata changed after validation selection")
    original = _json(run_dir / "data_manifest.json")
    if _json(data_dir / "manifest.json") != original:
        raise ValueError("Phase 1 manifest changed after model selection")
    # Exclusive marker is acquired BEFORE test file access. Never removed on failure.
    marker = run_dir / "test_evaluation.json"
    with marker.open("x", encoding="utf-8") as output:
        json.dump({"status": "started", "started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "selection_sha256": sha256(run_dir / "selection.json")}, output, indent=2)
    try:
        predictor = MLPDemandPredictor.load(bundle)
        train = _checked_partition(data_dir, "train", original)
        validation = _checked_partition(data_dir, "validation", original)
        test = _checked_partition(data_dir, "test", original)
        windows = make_windows(test, history=pd.concat([train, validation]),
                               lookback=config.lookback, horizon=config.horizon, stride=config.stride)
        if not len(windows.x):
            raise ValueError("No eligible test windows")
        predictions = {"selected_mlp": predict_windows(predictor.model, windows.x, predictor.normalizer),
                       **_baseline_predictions(windows, config.horizon)}
        scores = {name: demand_metrics(windows.y, values) for name, values in predictions.items()}
        results = {"selected_architecture": selection["selected_architecture"], "window_report": windows.report,
                   "metrics": scores,
                   "protocol": "one rolling-origin test pass; same origin/horizon pairs for MLP and both baselines",
                   "selection_sha256": sha256(run_dir / "selection.json")}
        # Preserve predictions and metrics before plotting so rendering never requires a new test pass.
        _save_predictions(run_dir / "test_predictions.csv", windows, predictions)
        write_json(run_dir / "test_metrics.json", results)
        write_json(marker, {"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                            "selection_sha256": results["selection_sha256"],
                            "test_sha256": sha256(data_dir / "test.csv")})
    except Exception as error:
        write_json(marker, {"status": "failed", "error": str(error),
                            "policy": "Do not automatically reopen test; inspect preserved artifacts first"})
        raise
    comparison_figures(scores, run_dir / "plots", "test")
    forecast_example(windows.origins, windows.y, predictions, run_dir / "plots", "test")
    return results


def replot_experiment(run_dir: Path) -> None:
    """Rebuild figures from saved reports/predictions; never read source test or run inference."""
    run_dir = Path(run_dir)
    selection = _json(run_dir / "selection.json")
    model_root = (run_dir / selection["selected_bundle"]).resolve().parent
    reports = {name: _json(model_root / name / "training.json") for name in selection["architectures"]}
    scale = DemandNormalizer.load(run_dir / "normalization.json").scale_kwh
    training_figures(reports, scale, run_dir / "plots")
    horizon = ExperimentConfig.load(run_dir / "config.json").horizon
    for split in ("validation", "test"):
        path = run_dir / f"{split}_predictions.csv"
        if not path.exists():
            continue
        saved = pd.read_csv(path, parse_dates=["origin", "target_timestamp"])
        if not len(saved) or len(saved) % horizon or not np.array_equal(
                saved.horizon_hour.to_numpy(), np.tile(np.arange(1, horizon + 1), len(saved) // horizon)):
            raise ValueError("Saved prediction rows are not ordered complete horizons")
        actual = saved.actual_kwh.to_numpy().reshape(-1, horizon)
        predictions = {name.removesuffix("_kwh"): saved[name].to_numpy().reshape(-1, horizon)
                       for name in saved.columns if name.endswith("_kwh") and name != "actual_kwh"}
        origins = pd.DatetimeIndex(saved.origin.iloc[::horizon])
        scores = (selection["validation_comparison"] if split == "validation"
                  else _json(run_dir / "test_metrics.json")["metrics"])
        comparison_figures(scores, run_dir / "plots", split)
        forecast_example(origins, actual, predictions, run_dir / "plots", split)
