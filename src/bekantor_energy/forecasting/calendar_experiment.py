"""Paired, bounded calendar ablation. Only train and validation are ever loaded."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd

from ..data.acquisition import sha256, write_json
from .calendar import CALENDAR_FIELDS, CalendarInputNormalizer, with_calendar
from .config import ExperimentConfig
from .experiment import _checked_partition
from .mlp import MLPDemandPredictor, configure_tensorflow, predict_windows, train_one
from .normalization import DemandNormalizer
from .windows import make_windows

VARIANTS = {"reference": "consumption_only", "calendar": "future_calendar"}
METRICS = ("mae_kwh", "rmse_kwh", "bias_kwh", "underestimate_pct")


@dataclass(frozen=True)
class CalendarComparisonConfig:
    training: ExperimentConfig
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    peak_quantiles: tuple[float, ...] = (0.9, 0.95)

    def __post_init__(self):
        object.__setattr__(self, "seeds", tuple(self.seeds))
        object.__setattr__(self, "peak_quantiles", tuple(self.peak_quantiles))
        if len(set(self.seeds)) < 3 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("At least three distinct paired seeds required")
        for seed in self.seeds:
            replace(self.training, seed=seed)  # Reuse seed validation.
        c = self.training
        if (c.lookback, c.horizon, c.stride) != (168, 24, 1):
            raise ValueError("Calendar ablation fixes lookback/horizon/stride to 168/24/1")
        if len(c.architectures) != 1:
            raise ValueError("Exactly one fixed hidden architecture required")
        a = c.architectures[0]
        if (a.hidden_units, a.activation, a.dropout, a.l2) != ((64, 32), "relu", 0.0, 0.0):
            raise ValueError("Calendar ablation fixes 64-32 ReLU, without dropout or L2")
        if len(set(self.peak_quantiles)) != len(self.peak_quantiles) or not self.peak_quantiles:
            raise ValueError("Distinct peak quantiles required")
        if any(not np.isfinite(q) or not 0 < q < 1 for q in self.peak_quantiles):
            raise ValueError("Peak quantiles must lie strictly between zero and one")
        if 0.9 not in self.peak_quantiles:
            raise ValueError("Primary predeclared high-consumption threshold is train P90")

    def to_dict(self):
        return {"training": self.training.to_dict(), "seeds": list(self.seeds),
                "peak_quantiles": list(self.peak_quantiles)}

    @classmethod
    def load(cls, path):
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(training=ExperimentConfig(**values["training"]),
                   seeds=values["seeds"], peak_quantiles=values["peak_quantiles"])


def peak_thresholds(train, quantiles):
    """Fit thresholds to unique, originally observed train hours, before windows."""
    observed = train.loc[train.is_observed_complete.astype(bool), "load_kwh"].to_numpy(dtype=float)
    if not len(observed) or not np.isfinite(observed).all():
        raise ValueError("Finite observed train demand required")
    return {f"q{100 * q:g}": float(np.quantile(observed, q, method="linear")) for q in quantiles}


def metric_values(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Nonempty aligned actual/prediction arrays required")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Nonfinite validation values")
    error = predicted - actual
    return {"mae_kwh": float(np.abs(error).mean()), "rmse_kwh": float(np.sqrt(np.mean(error ** 2))),
            "bias_kwh": float(error.mean()), "underestimate_pct": float(100 * np.mean(error < 0))}


def validation_rows(windows, predictions, thresholds, seed, variant):
    """One row per metric/scope; every forecast-origin/target pair has equal weight."""
    actual = windows.y
    if predictions.shape != actual.shape:
        raise ValueError("Prediction and validation window shapes differ")
    times = windows.origins.to_numpy()[:, None] + np.arange(actual.shape[1])[None, :] * np.timedelta64(1, "h")
    hours = pd.DatetimeIndex(times.ravel()).hour.to_numpy().reshape(actual.shape)
    horizons = np.broadcast_to(np.arange(1, actual.shape[1] + 1), actual.shape)
    rows = []

    def append(scope, group, mask):
        count = int(mask.sum())
        if not count:
            return  # An empty high-consumption stratum is absent, never assigned zero error.
        for metric, value in metric_values(actual[mask], predictions[mask]).items():
            rows.append({"seed": seed, "variant": variant, "scope": scope, "group": str(group),
                         "metric": metric, "value": value, "n_pairs": count})

    append("global", "all", np.ones(actual.shape, dtype=bool))
    for label, threshold in thresholds.items():
        append("high_consumption", label, actual.astype(float) > threshold)
    for hour in range(24):
        append("hour", hour, hours == hour)
    for horizon in range(1, actual.shape[1] + 1):
        append("horizon", horizon, horizons == horizon)
    return rows


def aggregate_metrics(rows, variants=("reference", "calendar")):
    frame = pd.DataFrame(rows)
    keys = ["variant", "scope", "group", "metric"]
    aggregate = frame.groupby(keys, sort=False).agg(
        mean=("value", "mean"), std=("value", "std"), minimum=("value", "min"),
        maximum=("value", "max"), n_seeds=("seed", "nunique"), n_pairs=("n_pairs", "first")).reset_index()
    if (frame.groupby(keys).n_pairs.nunique() != 1).any():
        raise ValueError("Seeds do not share validation support")
    paired = frame.pivot(index=["seed", "scope", "group", "metric"], columns="variant", values="value").reset_index()
    first, second = variants
    if not set(variants) <= set(paired.columns) or paired[list(variants)].isna().any().any():
        raise ValueError("Comparison is not paired")
    paired[f"delta_{second}_minus_{first}"] = paired[second] - paired[first]
    return frame, aggregate, paired


def original_snapshot(run_dir, model_dir):
    """Explicit allowlist. Never hash, read, or glob held-out evaluation artifacts."""
    paths = [Path(run_dir) / name for name in
             ("config.json", "normalization.json", "selection.json", "validation_predictions.csv")]
    for architecture in ("small", "medium", "large"):
        paths.extend(Path(model_dir) / architecture / name for name in ("best.keras", "training.json", "epochs.csv"))
    paths.extend(Path(model_dir) / "selected" / name for name in ("model.keras", "normalization.json", "predictor.json"))
    return {str(p.resolve()): sha256(p) for p in paths}


def run_calendar_comparison(data_dir, config, run_dir, model_dir, *, reference_run=None, reference_models=None):
    data_dir, run_dir, model_dir = map(Path, (data_dir, run_dir, model_dir))
    if run_dir.exists() or model_dir.exists():
        raise ValueError("Use new experiment/model paths; existing artifacts are preserved")
    if (reference_run is None) != (reference_models is None):
        raise ValueError("Both original artifact roots are required for preservation verification")
    protected = [data_dir] + ([Path(reference_run), Path(reference_models)] if reference_run else [])
    for output in (run_dir, model_dir):
        for source in protected:
            if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
                raise ValueError("Output paths must be disjoint from original data/artifacts")
    if run_dir.resolve().is_relative_to(model_dir.resolve()) or model_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Run/model paths must be disjoint")
    original = original_snapshot(reference_run, reference_models) if reference_run else {}
    if reference_run:
        reference = ExperimentConfig.load(Path(reference_run) / "config.json")
        for field in ("lookback", "horizon", "stride", "epochs", "batch_size", "learning_rate", "patience", "threads"):
            if getattr(config.training, field) != getattr(reference, field):
                raise ValueError(f"Training protocol differs from original reference: {field}")
        if config.training.architectures[0] not in reference.architectures:
            raise ValueError("Hidden architecture/regularization differs from original reference")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    # Only these two names reach the source reader. No hourly.csv or held-out file access.
    train = _checked_partition(data_dir, "train", manifest)
    validation = _checked_partition(data_dir, "validation", manifest)
    if train.index[-1] >= validation.index[0]:
        raise ValueError("Train must precede validation")
    c = config.training
    kwargs = {"lookback": c.lookback, "horizon": c.horizon, "stride": c.stride}
    training = make_windows(train, **kwargs)
    validating = make_windows(validation, history=train, **kwargs)
    if not len(training.x) or not len(validating.x):
        raise ValueError("Eligible train and validation windows required")
    calendar_train, calendar_validation = with_calendar(training), with_calendar(validating)
    normalizer = DemandNormalizer.fit(train)
    thresholds = peak_thresholds(train, config.peak_quantiles)
    run_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", config.to_dict())
    normalizer.save(run_dir / "normalization.json")
    write_json(run_dir / "thresholds.json", {"thresholds_kwh": thresholds, "fit_partition": "train",
               "fit_hours": normalizer.fitted_hours, "method": "linear quantile, unique observed train hours",
               "high_consumption_rule": "actual_kwh > threshold; strict, ex-post stratum"})
    write_json(run_dir / "windows.json", {"train": training.report, "validation": validating.report,
               "same_support_for_all_variants_and_seeds": True})
    tf = configure_tensorflow(config.seeds[0], c.threads)
    write_json(run_dir / "reproducibility.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "platform": platform.platform(), "cpu": platform.processor(),
        "devices": [d.name for d in tf.config.list_logical_devices()],
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "data": {name: manifest["partitions"][name] for name in ("train", "validation")},
        "source_hashes": {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "original_artifacts_before": original,
        "protocol": {"evaluation_partition": "validation only", "paired_seeds": list(config.seeds),
                     "calendar_fields": list(CALENDAR_FIELDS), "source_clock": "UCI naive, no timezone invented",
                     "selection": "minimum val_rmse checkpoint per fit; no seed selection or ensemble",
                     "dispersion": "sample std (ddof=1) over seeds, not confidence/prediction intervals",
                     "determinism": "CPU; oneDNN disabled; deterministic ops; seeded shuffled train only",
                     "early_stopping": "patience fixed, no train+validation refit",
                     "weighting": "equal origin/horizon pairs; overlapping windows retained"}})
    rows, fits = [], []
    for seed in config.seeds:
        seeded = replace(c, seed=seed)
        architecture = seeded.architectures[0]
        for variant, feature_mode in VARIANTS.items():
            tw, vw = (training, validating) if variant == "reference" else (calendar_train, calendar_validation)
            adapter = normalizer if variant == "reference" else CalendarInputNormalizer(normalizer, c.lookback, c.horizon)
            directory = model_dir / f"seed_{seed}" / variant
            print(f"\nSeed {seed}: {variant}, hidden 64-32, input={tw.x.shape[1]}", flush=True)
            model, report = train_one(tw, vw, adapter, seeded, architecture, directory,
                                      input_size=tw.x.shape[1], score_training=False)
            predicted = predict_windows(model, vw.x, adapter)
            rows.extend(validation_rows(validating, predicted, thresholds, seed, variant))
            np.savez_compressed(directory / "validation_predictions.npz", actual_kwh=validating.y,
                                predicted_kwh=predicted, origins=validating.origins.to_numpy())
            predictor = MLPDemandPredictor(seeded, architecture, directory,
                       model_id=f"{run_dir.name}:{seed}:{variant}", feature_mode=feature_mode)
            predictor.model, predictor.normalizer = model, normalizer
            predictor.save(directory / "bundle")
            fits.append({"seed": seed, "variant": variant, **{key: report[key] for key in
                         ("parameters", "input_size", "epochs_run", "best_epoch", "training_seconds")}})
            print(f"Validation RMSE={report['validation']['rmse_kwh']:.6f} kWh; best epoch={report['best_epoch']}", flush=True)
            del predictor, model
    frame, aggregate, paired = aggregate_metrics(rows)
    frame.to_csv(run_dir / "metrics_by_seed.csv", index=False)
    aggregate.to_csv(run_dir / "metrics_aggregate.csv", index=False)
    paired.to_csv(run_dir / "paired_deltas.csv", index=False)
    pd.DataFrame(fits).to_csv(run_dir / "fits.csv", index=False)
    after = original_snapshot(reference_run, reference_models) if reference_run else {}
    write_json(run_dir / "preservation.json", {"verified": original == after,
               "original_artifacts_before": original, "original_artifacts_after": after})
    if original != after:
        raise ValueError("Original artifacts changed during the experiment")
    from .calendar_plots import calendar_figures
    calendar_figures(run_dir, model_dir)
    write_json(run_dir / "completed.json", {"status": "completed", "fits": len(fits),
               "validation_windows": len(validating.x), "validation_pairs_per_fit": int(validating.y.size),
               "original_artifacts_unchanged": original == after, "evaluation_partition": "validation"})
    return aggregate
