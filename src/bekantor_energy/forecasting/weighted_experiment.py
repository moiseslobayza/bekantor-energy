"""One weighted loss, three paired seeds, preserved calendar-MSE control."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from ..data.acquisition import sha256, write_json
from .calendar import CalendarInputNormalizer, with_calendar
from .calendar_experiment import (CalendarComparisonConfig, aggregate_metrics, metric_values,
                                  peak_thresholds, validation_rows)
from .experiment import _checked_partition
from .mlp import MLPDemandPredictor, configure_tensorflow, predict_windows, train_one
from .normalization import DemandNormalizer
from .weighted_loss import make_peak_weighted_mse
from .windows import make_windows

VARIANTS = ("mse", "weighted_mse")


@dataclass(frozen=True)
class WeightedComparisonConfig:
    normal_weight: float = 1.0
    peak_weight: float = 2.0
    peak_quantile: float = 0.9
    low_quantile: float = 0.25
    global_error_tolerance_pct: float = 1.0
    weight_reduction: str = "mean_over_all_targets"

    def __post_init__(self):
        if (self.normal_weight, self.peak_weight, self.peak_quantile, self.low_quantile,
            self.weight_reduction) != (1., 2., .9, .25, "mean_over_all_targets"):
            raise ValueError("Experiment 2 fixes weights 1/2, train P90 and low consumption <= train P25")
        if not np.isfinite(self.global_error_tolerance_pct) or self.global_error_tolerance_pct < 0:
            raise ValueError("Nonnegative finite operational global-error tolerance required")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def weighted_validation_rows(windows, predictions, thresholds, seed, variant):
    rows = validation_rows(windows, predictions, {q: thresholds[q] for q in ("q90", "q95")}, seed, variant)
    for scope, group, mask in (("low_consumption", "q25", windows.y.astype(float) <= thresholds["q25"]),
                                ("normal_consumption", "leq90", windows.y.astype(float) <= thresholds["q90"])):
        if not mask.any():
            continue
        scores = metric_values(windows.y[mask], predictions[mask])
        scores["overestimate_pct"] = float(100 * np.mean(predictions[mask] > windows.y[mask]))
        for metric, value in scores.items():
            rows.append({"seed": seed, "variant": variant, "scope": scope, "group": group,
                         "metric": metric, "value": value, "n_pairs": int(mask.sum())})
    return rows


def previous_snapshot(reference_run, reference_models, seeds):
    """Explicit filenames only; no glob or source reader can reach held-out data."""
    reference_run, reference_models = Path(reference_run), Path(reference_models)
    names = ("config.json", "normalization.json", "thresholds.json", "windows.json", "fits.csv",
             "metrics_by_seed.csv", "metrics_aggregate.csv", "paired_deltas.csv", "reproducibility.json",
             "preservation.json", "completed.json")
    paths = [reference_run / name for name in names]
    if (reference_run / "verification.json").exists():
        paths.append(reference_run / "verification.json")
    for seed in seeds:
        for variant in ("reference", "calendar"):
            paths.extend(reference_models / f"seed_{seed}" / variant / name for name in
                         ("best.keras", "epochs.csv", "training.json", "validation_predictions.npz",
                          "bundle/model.keras", "bundle/predictor.json", "bundle/normalization.json"))
    paths.extend((reference_run / "plots").glob("*.png"))
    originals = json.loads((reference_run / "preservation.json").read_text(encoding="utf-8"))["original_artifacts_before"]
    for path, expected in originals.items():
        # The inherited allowlist never contains evaluation files.
        if Path(path).name.startswith("test"):
            raise ValueError("Held-out artifact in preservation allowlist")
        if sha256(Path(path)) != expected:
            raise ValueError("Earlier Phase 2 artifact differs from preserved hash")
        paths.append(Path(path))
    return {str(path.resolve()): sha256(path) for path in paths}


def _check_data_and_control(reference_run, train, validation, training, validating, normalizer, manifest):
    reference_run = Path(reference_run)
    if normalizer != DemandNormalizer.load(reference_run / "normalization.json"):
        raise ValueError("Normalization differs from calendar reference")
    reports = json.loads((reference_run / "windows.json").read_text(encoding="utf-8"))
    if reports["train"] != training.report or reports["validation"] != validating.report:
        raise ValueError("Window protocol/support differs from calendar reference")
    data = json.loads((reference_run / "reproducibility.json").read_text(encoding="utf-8"))["data"]
    if any(data[name] != manifest["partitions"][name] for name in ("train", "validation")):
        raise ValueError("Data partitions differ from calendar reference")
    if train.index[-1] >= validation.index[0]:
        raise ValueError("Training must precede validation")


def run_weighted_comparison(data_dir, config, run_dir, model_dir, reference_run, reference_models):
    data_dir, run_dir, model_dir, reference_run, reference_models = map(
        Path, (data_dir, run_dir, model_dir, reference_run, reference_models))
    if run_dir.exists() or model_dir.exists():
        raise ValueError("New run/model directories required; previous experiments are immutable")
    for output in (run_dir, model_dir):
        for protected in (data_dir, reference_run, reference_models):
            if output.resolve().is_relative_to(protected.resolve()) or protected.resolve().is_relative_to(output.resolve()):
                raise ValueError("Outputs must be disjoint from input data and previous artifacts")
    if run_dir.resolve().is_relative_to(model_dir.resolve()) or model_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Run/model roots must be disjoint")
    control = CalendarComparisonConfig.load(reference_run / "config.json")
    if control.seeds != (2026, 2027, 2028):
        raise ValueError("Experiment 2 must reuse seeds 2026/2027/2028")
    before = previous_snapshot(reference_run, reference_models, control.seeds)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train = _checked_partition(data_dir, "train", manifest)
    validation = _checked_partition(data_dir, "validation", manifest)
    c = control.training
    kwargs = {"lookback": c.lookback, "horizon": c.horizon, "stride": c.stride}
    training, validating = make_windows(train, **kwargs), make_windows(validation, history=train, **kwargs)
    if not len(training.x) or not len(validating.x):
        raise ValueError("Eligible train/validation windows required")
    normalizer = DemandNormalizer.fit(train)
    _check_data_and_control(reference_run, train, validation, training, validating, normalizer, manifest)
    thresholds = peak_thresholds(train, (.25, .9, .95))
    scaled_threshold = np.float32(thresholds["q90"] / normalizer.scale_kwh)
    counts = {}
    for name, windows in (("train", training), ("validation", validating)):
        physical = windows.y.astype(float) > thresholds["q90"]
        scaled = normalizer.targets(windows.y) > scaled_threshold
        if not np.array_equal(physical, scaled):
            raise ValueError("Peak classification changes after normalization; inspect numeric boundary")
        counts[name] = {"targets": int(windows.y.size), "high_targets": int(physical.sum())}
    tw, vw = with_calendar(training), with_calendar(validating)
    adapter = CalendarInputNormalizer(normalizer, c.lookback, c.horizon)
    # Verify all existing control predictions before fitting or creating outputs.
    sources, baseline = {}, {}
    for seed in control.seeds:
        directory = reference_models / f"seed_{seed}" / "calendar"
        metadata = json.loads((directory / "bundle" / "predictor.json").read_text(encoding="utf-8"))
        if (metadata["feature_mode"] != "future_calendar" or
            metadata["config"] != json.loads(json.dumps(replace(c, seed=seed).to_dict()))):
            raise ValueError("Existing MSE calendar predictor configuration differs")
        with np.load(directory / "validation_predictions.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["actual_kwh"], validating.y)
            np.testing.assert_array_equal(saved["origins"], validating.origins.to_numpy())
            baseline[seed] = saved["predicted_kwh"].copy()
        sources[str(seed)] = str((directory / "validation_predictions.npz").resolve())
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", {"loss_experiment": asdict(config), "control": control.to_dict(),
               "checkpoint_monitor": "unweighted val_rmse", "trainings": 3,
               "control_policy": "reuse frozen MSE predictions/checkpoints from Experiment 1, paired by seed"})
    write_json(run_dir / "thresholds.json", {"thresholds_kwh": thresholds, "p90_scaled": float(scaled_threshold),
               "p90_scaled_float64": thresholds["q90"] / normalizer.scale_kwh,
               "fit_partition": "train", "fit_observed_unique_hours": normalizer.fitted_hours,
               "high_rule": "actual > train P90/P95", "low_rule": "actual <= train P25",
               "normalization": "target = kWh / scale; no centering", "peak_masks_equal_after_scaling": True,
               "counts": counts, "loss_reduction": "mean(w*error_squared) over batch and horizon; raw weights 1/2"})
    write_json(run_dir / "windows.json", {"train": training.report, "validation": validating.report})
    write_json(run_dir / "baseline_sources.json", sources)
    normalizer.save(run_dir / "normalization.json")
    tf = configure_tensorflow(control.seeds[0], c.threads)
    write_json(run_dir / "reproducibility.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(),
               "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
               "devices": [d.name for d in tf.config.list_logical_devices()],
               "source_hashes": {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "data": {name: manifest["partitions"][name] for name in ("train", "validation")},
               "original_artifacts_before": before,
               "protocol": "Same 312 inputs, 64-32, Adam/LR, seeded order, callbacks, CPU deterministic ops; only loss changes",
               "selection": "minimum unweighted val_rmse per fit; no seed selection, no refit",
               "interpretation": "1% global MAE/RMSE engineering tolerance, not a statistical significance threshold"})
    rows, fits = [], []
    for seed in control.seeds:
        seeded = replace(c, seed=seed)
        rows.extend(weighted_validation_rows(validating, baseline[seed], thresholds, seed, "mse"))
        old_report = json.loads((reference_models / f"seed_{seed}" / "calendar" / "training.json").read_text(encoding="utf-8"))
        fits.append({"seed": seed, "variant": "mse", "trained_in_this_experiment": False,
                     **{key: old_report[key] for key in ("parameters", "input_size", "epochs_run", "best_epoch", "training_seconds")}})
        directory = model_dir / f"seed_{seed}" / "weighted_mse"
        loss = make_peak_weighted_mse(thresholds["q90"], normalizer.scale_kwh)
        print(f"\nSeed {seed}: weighted MSE; train P90={thresholds['q90']:.8f} kWh, scaled={float(scaled_threshold):.8f}", flush=True)
        model, report = train_one(tw, vw, adapter, seeded, seeded.architectures[0], directory,
                                  input_size=312, score_training=False, loss=loss)
        predicted = predict_windows(model, vw.x, adapter)
        rows.extend(weighted_validation_rows(validating, predicted, thresholds, seed, "weighted_mse"))
        np.savez_compressed(directory / "validation_predictions.npz", actual_kwh=validating.y,
                            predicted_kwh=predicted, origins=validating.origins.to_numpy())
        predictor = MLPDemandPredictor(seeded, seeded.architectures[0], directory,
                    model_id=f"{run_dir.name}:{seed}:weighted_mse", feature_mode="future_calendar")
        predictor.model, predictor.normalizer = model, normalizer
        predictor.save(directory / "bundle")
        # Inference bundle is uncompiled; persist objective provenance alongside it.
        write_json(directory / "bundle" / "training_objective.json", {
                   "loss": report["loss"], "threshold_kwh": thresholds["q90"], "scale_kwh": normalizer.scale_kwh})
        fits.append({"seed": seed, "variant": "weighted_mse", "trained_in_this_experiment": True,
                     **{key: report[key] for key in ("parameters", "input_size", "epochs_run", "best_epoch", "training_seconds")}})
        print(f"Validation unweighted RMSE={report['validation']['rmse_kwh']:.6f} kWh; best epoch={report['best_epoch']}", flush=True)
        del predictor, model
    frame, aggregate, paired = aggregate_metrics(rows, VARIANTS)
    frame.to_csv(run_dir / "metrics_by_seed.csv", index=False)
    aggregate.to_csv(run_dir / "metrics_aggregate.csv", index=False)
    paired.to_csv(run_dir / "paired_deltas.csv", index=False)
    pd.DataFrame(fits).to_csv(run_dir / "fits.csv", index=False)
    tradeoffs = paired[(paired.scope == "global") & paired.metric.isin(("mae_kwh", "rmse_kwh"))].copy()
    tradeoffs["relative_change_pct"] = 100 * tradeoffs.delta_weighted_mse_minus_mse / tradeoffs.mse
    tradeoffs["within_operational_tolerance"] = tradeoffs.relative_change_pct <= config.global_error_tolerance_pct
    tradeoffs.to_csv(run_dir / "global_tradeoffs.csv", index=False)
    global_changes = {}
    for metric in ("mae_kwh", "rmse_kwh"):
        mean = aggregate[(aggregate.scope == "global") & (aggregate.metric == metric)].set_index("variant")["mean"]
        global_changes[metric] = float(100 * (mean["weighted_mse"] - mean["mse"]) / mean["mse"])
    write_json(run_dir / "global_guardrail.json", {"tolerance_pct": config.global_error_tolerance_pct,
               "meaning": "engineering interpretation only, not statistical significance",
               "change_ratio_of_means_pct": global_changes,
               "mean_global_errors_within_tolerance": all(v <= config.global_error_tolerance_pct for v in global_changes.values()),
               "all_seed_global_errors_within_tolerance": bool(tradeoffs.within_operational_tolerance.all())})
    after = previous_snapshot(reference_run, reference_models, control.seeds)
    write_json(run_dir / "preservation.json", {"verified": before == after,
               "original_artifacts_before": before, "original_artifacts_after": after})
    if before != after:
        raise ValueError("Previous artifacts changed during weighted experiment")
    from .weighted_plots import weighted_figures
    weighted_figures(run_dir, model_dir, reference_models)
    write_json(run_dir / "completed.json", {"status": "completed", "new_fits": 3,
               "reused_mse_controls": 3, "evaluation_partition": "validation",
               "validation_windows": len(validating.x), "pairs_per_model": int(validating.y.size),
               "original_artifacts_unchanged": True})
    return aggregate
