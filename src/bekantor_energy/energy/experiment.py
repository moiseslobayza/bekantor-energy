"""Reproducible validation demonstration; no neural fitting or held-out access."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from ..config import HomeConfig
from ..data.acquisition import sha256, write_json
from ..forecasting.experiment import _checked_partition
from ..forecasting.mlp import MLPDemandPredictor
from ..forecasting.normalization import DemandNormalizer
from .evaluation import evaluate_receding
from .planner import PlannerConfig
from .scenarios import first_complete_midnight, prepare_solar_scenario


class CausalForecastProvider:
    def __init__(self, historical_data, predictor, horizon=24):
        self.historical_data, self.predictor, self.horizon = historical_data, predictor, horizon
        self.audit = []

    def __call__(self, origin):
        # Only completed hours strictly before issue time reach the neural port.
        history = self.historical_data.loc[self.historical_data.index < origin]
        if history.empty or history.index[-1] >= origin:
            raise ValueError("Causal history unavailable")
        starts = pd.date_range(origin, periods=self.horizon, freq="h").to_pydatetime()
        result = self.predictor.predict(history, starts)
        self.audit.append({"origin": origin.isoformat(), "last_history_timestamp": history.index[-1].isoformat(),
                           "history_hours": len(history), "model_id": result.model_id})
        return result


def preservation_snapshot(previous_verification):
    """Reuse explicit non-test allowlists from completed predictive experiments."""
    verification = Path(previous_verification)
    previous = json.loads(verification.read_text(encoding="utf-8"))
    preserved_path = verification.parent / "preservation.json"
    preserved = json.loads(preserved_path.read_text(encoding="utf-8"))
    expected = {**previous["artifact_sha256"], **preserved["original_artifacts_before"]}
    snapshot = {}
    for value, checksum in expected.items():
        path = Path(value)
        if path.name.startswith("test"):
            raise ValueError("Held-out path in preservation allowlist")
        actual = sha256(path)
        if actual != checksum:
            raise ValueError(f"Previous artifact changed: {path}")
        snapshot[str(path.resolve())] = actual
    snapshot[str(verification.resolve())] = sha256(verification)
    snapshot[str(preserved_path.resolve())] = sha256(preserved_path)
    return snapshot


def run_validation_demo(data_dir, model_bundle, solar_raw, home_config, planner_config, run_dir,
                        scenario_dir, *, previous_verification):
    data_dir, model_bundle, solar_raw, run_dir, scenario_dir = map(
        Path, (data_dir, model_bundle, solar_raw, run_dir, scenario_dir))
    for output in (run_dir, scenario_dir):
        if output.exists():
            raise ValueError("New result/scenario directories required")
        for source in (data_dir, model_bundle.parent, solar_raw.parent, Path(previous_verification).parent):
            if output.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(output.resolve()):
                raise ValueError("Output directories must be disjoint from source artifacts")
    if run_dir.resolve().is_relative_to(scenario_dir.resolve()) or scenario_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Output and scenario roots must be disjoint")
    home = HomeConfig.load(home_config)
    settings = PlannerConfig(**json.loads(Path(planner_config).read_text(encoding="utf-8")))
    if home.horizon_steps != 24 or home.step_hours != 1 or any(home.export_prices):
        raise ValueError("Validation demo requires 24 hourly steps, no export revenue")
    if settings.terminal_min_soc not in (None, home.min_soc):
        raise ValueError("Comparison demo uses the same reserve-only terminal constraint for both strategies")
    before = preservation_snapshot(previous_verification)
    before.update({str((model_bundle / name).resolve()): sha256(model_bundle / name)
                   for name in ("model.keras", "normalization.json", "predictor.json")})
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train = _checked_partition(data_dir, "train", manifest)
    validation = _checked_partition(data_dir, "validation", manifest)
    if train.index[-1] >= validation.index[0]:
        raise ValueError("Train must precede validation")
    if DemandNormalizer.load(model_bundle / "normalization.json") != DemandNormalizer.fit(train):
        raise ValueError("Model normalization does not match observed training data")
    with np.load(model_bundle.parent / "validation_predictions.npz", allow_pickle=False) as saved:
        origins = pd.DatetimeIndex(saved["origins"])
    origin = first_complete_midnight(origins, validation)
    times = pd.date_range(origin, periods=24, freq="h")
    solar, solar_metadata = prepare_solar_scenario(solar_raw, home, scenario_dir)
    predictor = MLPDemandPredictor.load(model_bundle)  # No .fit, .compile or weight mutation.
    if predictor.feature_mode != "future_calendar" or predictor.config.lookback != 168 or predictor.config.horizon != 24:
        raise ValueError("Expected the existing 168->24 calendar MLP bundle")
    provider = CausalForecastProvider(pd.concat([train, validation]), predictor)
    result = evaluate_receding(validation.loc[times, "load_kwh"].to_numpy(), times, solar.solar_kwh.to_numpy(),
                               provider, home, settings, scenario_id="cordoba_pvgis_independent_uci_validation")
    run_dir.mkdir(parents=True)
    write_json(run_dir / "config.json", {"home": asdict(home), "planner": asdict(settings),
               "model_bundle": str(model_bundle.resolve()), "model_id": predictor.model_id})
    write_json(run_dir / "metrics.json", result["metrics"])
    for name in ("ledger", "planning", "first_plan", "forecasts"):
        result[name].to_csv(run_dir / f"{name}.csv", index=False)
    pd.DataFrame.from_dict(result["metrics"], orient="index").rename_axis("strategy").to_csv(run_dir / "comparison.csv")
    write_json(run_dir / "causality_audit.json", provider.audit)
    write_json(run_dir / "reproducibility.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "platform": platform.platform(), "evaluation_partition": "validation only",
               "validation_period": [times[0].isoformat(), times[-1].isoformat()],
               "selection": "first midnight with 24 consecutive existing eligible forecast origins; no cost selection",
               "data": {name: manifest["partitions"][name] for name in ("train", "validation")},
               "solar": solar_metadata, "scenario_data_dir": str(scenario_dir.resolve()),
               "model_sha256": {name: sha256(model_bundle / name) for name in ("model.keras", "normalization.json", "predictor.json")},
               "source_sha256": {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "evaluation_seconds": result["elapsed_seconds"],
               "assumptions": ["no outages in this initial demonstration", "perfect modeled solar path",
                               "illustrative SIM tariffs, no claims of real savings",
                               "same actual demand/PV/tariffs/initial battery/reserve for both strategies",
                               "baseline has ideal instantaneous feedback, represented by hourly averages",
                               "A* optimal only on discretized forecast graph, not on realized MPC replay",
                               "no terminal battery value; final SOC reported explicitly"]})
    after = {path: sha256(Path(path)) for path in before}
    write_json(run_dir / "preservation.json", {"verified": before == after,
               "original_artifacts_before": before, "original_artifacts_after": after})
    if before != after:
        raise ValueError("Previous prediction artifacts changed")
    from .plots import energy_figures
    energy_figures(run_dir)
    write_json(run_dir / "completed.json", {"status": "completed", "hours": 24, "replans": 24,
               "evaluation_partition": "validation", "neural_training_performed": False,
               "original_artifacts_unchanged": True})
    print(json.dumps(result["metrics"], indent=2), flush=True)
    return result
