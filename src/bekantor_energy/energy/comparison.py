"""Four-way paired scenario evaluation with frozen forecast replay and an oracle.

No neural fitting. Only the oracle provider receives realized future demand.
All strategies share scenario conditions, original objective and physical engine.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from ..config import HomeConfig
from ..contracts import EnergyState, LoadForecast, PlanningInputs
from ..data.acquisition import sha256, write_json
from ..forecasting.experiment import _checked_partition
from ..forecasting.metrics import demand_metrics
from ..forecasting.mlp import MLPDemandPredictor
from ..forecasting.windows import make_windows
from .evaluation import realized_metrics
from .experiment import preservation_snapshot
from .planner import AStarPlanner, PlannerConfig
from .scenarios import prepare_solar_scenario
from .simulator import EPS, PhysicalSimulator, SelfConsumptionStrategy

STRATEGIES = ("reactive", "mlp_mse", "mlp_weighted", "oracle")
METRIC_COLUMNS = ("import_cost_sim", "grid_import_kwh", "solar_utilization_pct", "min_soc", "final_soc",
                  "unserved_kwh", "constraint_violations", "charge_kwh", "discharge_kwh",
                  "planning_seconds", "expanded_nodes", "conversion_loss_kwh", "adjusted_actions")


@dataclass(frozen=True)
class ComparisonConfig:
    days_per_demand_stratum: int = 7
    solar_factors: dict | None = None
    initial_soc_levels: tuple = (.2, .8)
    tariff_modes: tuple = ("flat", "time_of_use")
    flat_tariff_sim_per_kwh: float = .15
    seed: int = 2026
    workers: int = 2
    description: str = "Fixed factorial validation design"

    def __post_init__(self):
        object.__setattr__(self, "solar_factors", {"high": 1., "low": .25} if self.solar_factors is None else dict(self.solar_factors))
        object.__setattr__(self, "initial_soc_levels", tuple(self.initial_soc_levels))
        object.__setattr__(self, "tariff_modes", tuple(self.tariff_modes))
        if type(self.days_per_demand_stratum) is not int or self.days_per_demand_stratum < 1:
            raise ValueError("Positive integer days per demand stratum required")
        if set(self.solar_factors) != {"high", "low"} or not 0 < self.solar_factors["low"] < self.solar_factors["high"] <= 1:
            raise ValueError("Solar factors must satisfy 0 < low < high <= 1")
        if not self.initial_soc_levels or len(set(self.initial_soc_levels)) != len(self.initial_soc_levels):
            raise ValueError("Distinct SOC levels required")
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in self.initial_soc_levels):
            raise ValueError("SOC levels must lie in [0,1]")
        if set(self.tariff_modes) != {"flat", "time_of_use"} or len(self.tariff_modes) != 2:
            raise ValueError("Exactly flat and time_of_use tariff modes required")
        if not np.isfinite(self.flat_tariff_sim_per_kwh) or self.flat_tariff_sim_per_kwh < 0:
            raise ValueError("Nonnegative finite flat tariff required")
        if self.seed != 2026:
            raise ValueError("This comparison freezes the predefined seed-2026 pair, without seed selection")
        if type(self.workers) is not int or not 1 <= self.workers <= 4:
            raise ValueError("Use one to four local worker processes")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def select_episodes(origins, validation, per_stratum):
    """Offline design, not an input available to a production controller."""
    candidates, origin_set = [], set(pd.DatetimeIndex(origins))
    for day in sorted(t for t in origin_set if t.hour == 0):
        issue_times = pd.date_range(day, periods=24, freq="h")
        targets = validation.reindex(pd.date_range(day, periods=47, freq="h"))
        if all(t in origin_set for t in issue_times) and targets.is_observed_complete.fillna(False).all() and np.isfinite(targets.load_kwh).all():
            candidates.append({"start": day, "demand_kwh": float(targets.load_kwh.iloc[:24].sum()),
                               "day_of_week": day.dayofweek, "month": day.month})
    frame = pd.DataFrame(candidates)
    if frame.empty:
        raise ValueError("No eligible validation days")
    low, high = frame.demand_kwh.quantile([1 / 3, 2 / 3]).to_numpy()
    selected, used = [], set()
    for label, mask in (("low", frame.demand_kwh <= low), ("high", frame.demand_kwh >= high)):
        pool = frame[mask & ~frame.start.isin(used)].sort_values("start")
        if len(pool) < per_stratum:
            raise ValueError(f"Insufficient distinct complete days in {label} demand stratum")
        positions = np.linspace(0, len(pool) - 1, per_stratum).round().astype(int)
        chosen = pool.iloc[positions].copy()
        chosen["demand_stratum"] = label
        selected.append(chosen)
        used.update(chosen.start)
    selected = pd.concat(selected).sort_values("start").reset_index(drop=True)
    selected["episode_id"] = [f"day_{i + 1:02d}" for i in range(len(selected))]
    if selected.start.duplicated().any() or (selected.start.diff().dropna() < pd.Timedelta(hours=24)).any():
        raise ValueError("Executed 24h episodes overlap")
    metadata = {"eligible_days": len(frame), "low_tercile_max_kwh": float(low),
                "high_tercile_min_kwh": float(high), "selection": f"{per_stratum} chronological quantile positions in each demand tercile; no cost-based selection",
                "retrospective_design": True, "executed_hours_nonoverlapping": True,
                "forecast_tails_may_overlap": True}
    return selected, frame, metadata


class FrozenForecastProvider:
    """No actual demand argument or oracle reference exists in this provider."""
    def __init__(self, origins, predictions, model_id):
        self.origins = pd.DatetimeIndex(origins)
        self.values = np.asarray(predictions, dtype=float)
        if self.values.shape != (len(self.origins), 24) or not np.isfinite(self.values).all() or (self.values < 0).any():
            raise ValueError("Expected finite nonnegative frozen 24h predictions")
        self.model_id = model_id
        self.lookup = {t: i for i, t in enumerate(self.origins)}
        if len(self.lookup) != len(self.origins):
            raise ValueError("Duplicate forecast origins")

    def __call__(self, origin):
        origin = pd.Timestamp(origin)
        values = self.values[self.lookup[origin]]
        return LoadForecast(origin.to_pydatetime(), tuple(pd.date_range(origin, periods=24, freq="h").to_pydatetime()),
                            tuple(float(v) for v in values), self.model_id)


class OracleForecastProvider(FrozenForecastProvider):
    """Explicit perfect-future-demand reference, unavailable in production."""
    pass


def comparison_metrics(rows, home):
    metrics = realized_metrics(rows, home)
    frame = pd.DataFrame(rows)
    return {**metrics, "charge_kwh": float(frame.charge_kwh.sum()),
            "discharge_kwh": float(frame.discharge_kwh.sum()), "final_soc": float(frame.soc.iloc[-1]),
            "import_to_demand_pct": 100 * metrics["grid_import_kwh"] / metrics["demand_kwh"] if metrics["demand_kwh"] else None}


def run_paired_episode(case, output_dir, *, planner_factory=AStarPlanner):
    """One 24h case, all strategies, full predicted-plan feasibility audit.

    Failure is a failed strategy, with partial traces; no silent fallback policy.
    A failure never masquerades as a successfully simulated complete episode.
    """
    home = HomeConfig(**case["home"])
    settings = PlannerConfig(**case["planner"])
    times = pd.date_range(case["start"], periods=24, freq="h")
    demand, solar = np.asarray(case["actual_demand"], float), np.asarray(case["solar"], float)
    if demand.shape != (24,) or solar.shape != (48,) or not np.isfinite(demand).all() or (demand < 0).any() or not np.isfinite(solar).all() or (solar < 0).any():
        raise ValueError("Expected 24 realized demand values and 48 independent modeled solar values")
    providers = {name: FrozenForecastProvider(times, case["predictions"][name], case["model_ids"][name])
                 for name in ("mlp_mse", "mlp_weighted")}
    providers["oracle"] = OracleForecastProvider(times, case["oracle_predictions"], "ORACLE_PERFECT_FUTURE_NOT_PRODUCTION")
    planner, sim, policy = planner_factory(settings), PhysicalSimulator(), SelfConsumptionStrategy()
    states = {name: EnergyState(0, home.initial_soc * home.battery_capacity_kwh) for name in STRATEGIES}
    failures, ledger, planned, forecast_rows = {}, [], [], []
    plan_audit = {name: {"checked_plans": 0, "checked_transitions": 0, "max_balance_residual": 0., "max_storage_residual": 0.} for name in STRATEGIES[1:]}
    for t, now in enumerate(times):
        prices = tuple(home.import_prices[dt.hour] for dt in pd.date_range(now, periods=24, freq="h"))
        shifted = replace(home, import_tariff_per_kwh=prices)
        for name in STRATEGIES:
            if name in failures:
                continue
            start = perf_counter()
            try:
                state = states[name]
                plan = None
                if name == "reactive":
                    action = policy.action(demand[t], solar[t], home)  # Ideal within-hour feedback only.
                else:
                    forecast = providers[name](now)  # No current/future actual demand passed to neural providers.
                    horizon_solar = tuple(float(v) for v in solar[t:t + 24])
                    inputs = PlanningInputs(forecast, horizon_solar, shifted, case["case_id"])
                    plan = planner.plan(inputs, EnergyState(0, state.stored_kwh))
                    replay = EnergyState(0, state.stored_kwh)
                    predicted_cost = 0.
                    for h, planned_action in enumerate(plan.actions):
                        r = sim.step(replay, planned_action, forecast.demand_kwh[h], horizon_solar[h], shifted)
                        if abs(r.state.stored_kwh - plan.states[h + 1].stored_kwh) > EPS:
                            raise ArithmeticError("Planned states do not match exact physical replay")
                        predicted_cost += r.cost
                        replay = r.state
                        audit = plan_audit[name]
                        audit["checked_transitions"] += 1
                        audit["max_balance_residual"] = max(audit["max_balance_residual"], abs(r.balance_residual_kwh))
                        audit["max_storage_residual"] = max(audit["max_storage_residual"], abs(r.storage_residual_kwh))
                    if len(plan.actions) != 24 or abs(predicted_cost - plan.predicted_cost) > 1e-9:
                        raise ArithmeticError("Incomplete or cost-inconsistent 24h plan")
                    plan_audit[name]["checked_plans"] += 1
                    action = plan.actions[0]
                    planned.append({"strategy": name, "origin": now, "predicted_cost_24h": plan.predicted_cost,
                                    "seconds": plan.elapsed_seconds, "expanded_nodes": plan.expanded_nodes,
                                    "generated_nodes": plan.generated_nodes, "initial_kwh": state.stored_kwh})
                    forecast_rows.extend({"strategy": name, "origin": now, "target_timestamp": target,
                                          "predicted_kwh": value, "is_oracle": name == "oracle"}
                                         for target, value in zip(forecast.interval_starts, forecast.demand_kwh))
                result = sim.step(state, action, float(demand[t]), float(solar[t]), home,
                                  repair=True, import_price=prices[0])
                states[name] = result.state
                record = asdict(result)
                record.pop("state")
                ledger.append({"strategy": name, "timestamp": now, **record, "stored_kwh": result.state.stored_kwh,
                               "soc": result.state.stored_kwh / home.battery_capacity_kwh,
                               "requested_charge_kw": action.charge_kw, "requested_discharge_kw": action.discharge_kw,
                               "tariff_sim_per_kwh": prices[0], "grid_available": True,
                               "planning_seconds": plan.elapsed_seconds if plan else 0.,
                               "expanded_nodes": plan.expanded_nodes if plan else 0,
                               "generated_nodes": plan.generated_nodes if plan else 0})
            except Exception as error:
                failures[name] = {"step": t, "exception_type": type(error).__name__, "message": str(error),
                                  "elapsed_failed_call_seconds": perf_counter() - start}
    results = []
    for name in STRATEGIES:
        rows = [row for row in ledger if row["strategy"] == name]
        status = "failed" if name in failures else "completed"
        metrics = comparison_metrics(rows, home) if status == "completed" and len(rows) == 24 else {}
        results.append({"strategy": name, "status": status, "hours_completed": len(rows),
                        "is_oracle": name == "oracle", **metrics})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(ledger).to_csv(output_dir / "ledger.csv", index=False)
    pd.DataFrame(planned).to_csv(output_dir / "planning.csv", index=False)
    pd.DataFrame(forecast_rows).to_csv(output_dir / "forecasts.csv.gz", index=False, compression="gzip")
    write_json(output_dir / "plan_audit.json", plan_audit)
    write_json(output_dir / "failures.json", failures)
    write_json(output_dir / "case.json", {k: v for k, v in case.items() if k not in ("predictions", "oracle_predictions")})
    write_json(output_dir / "metrics.json", results)
    return {"case_id": case["case_id"], "metrics": results, "failures": failures}


def summarize_results(results, output):
    """Primary aggregates use complete four-way cases; failures remain visible."""
    output = Path(output)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "results_by_case.csv", index=False)
    status = frame.groupby("case_id").agg(completed=("status", lambda s: int((s == "completed").sum())),
                                          strategies=("strategy", "nunique"))
    good_ids = status.index[(status.completed == 4) & (status.strategies == 4)]
    frame["four_way_complete"] = frame.case_id.isin(good_ids)
    frame.to_csv(output / "results_by_case.csv", index=False)
    complete = frame[frame.four_way_complete].copy()
    status.to_csv(output / "case_coverage.csv")
    if complete.empty:
        write_json(output / "coverage.json", {"total_cases": len(status), "four_way_complete": 0,
                   "failed_cases": len(status), "warning": "No complete paired cases; no comparison possible"})
        return
    baselines = complete[complete.strategy == "reactive"].set_index("case_id")
    weighted_reference = complete[complete.strategy == "mlp_mse"].set_index("case_id")
    for reference_name, reference in (("reactive", baselines), ("mlp_mse", weighted_reference)):
        for metric in ("import_cost_sim", "grid_import_kwh"):
            ref = complete.case_id.map(reference[metric])
            complete[f"delta_{metric}_vs_{reference_name}"] = complete[metric] - ref
            # Undefined if zero denominator. Never invent a percentage or infinity.
            complete[f"improvement_pct_{metric}_vs_{reference_name}"] = 100 * (ref - complete[metric]) / ref.where(ref.abs() > 1e-12)
    complete["cost_down_grid_up_vs_reactive"] = ((complete.delta_import_cost_sim_vs_reactive < -1e-9) &
                                                   (complete.delta_grid_import_kwh_vs_reactive > 1e-9))
    complete.to_csv(output / "paired_results.csv", index=False)
    metrics = [c for c in METRIC_COLUMNS if c in complete]
    aggregate = complete.groupby("strategy")[metrics].agg(["mean", "std", "median", "min", "max", "sum"])
    aggregate.columns = [f"{metric}__{stat}" for metric, stat in aggregate.columns]
    aggregate.to_csv(output / "aggregate.csv")
    for name, groups in (("by_scenario", ["solar_level", "tariff_mode", "initial_soc", "strategy"]),
                         ("by_demand", ["demand_stratum", "strategy"])):
        scoped = complete.groupby(groups)[metrics].agg(["mean", "std", "min", "max"])
        scoped.columns = [f"{metric}__{stat}" for metric, stat in scoped.columns]
        scoped.to_csv(output / f"{name}.csv")
    comparisons = []
    for name, group in complete.groupby("strategy"):
        baseline_cost = group.case_id.map(baselines.import_cost_sim)
        cost_percent = group.improvement_pct_import_cost_sim_vs_reactive
        comparisons.append({"strategy": name, "n_cases": len(group),
            "cost_improvement_ratio_of_sums_pct": 100 * (1 - group.import_cost_sim.sum() / baseline_cost.sum()) if baseline_cost.sum() else None,
            "cost_improvement_mean_episode_pct": cost_percent.mean(), "cost_improvement_std_episode_pp": cost_percent.std(),
            "cost_improvement_min_episode_pct": cost_percent.min(), "cost_improvement_max_episode_pct": cost_percent.max(),
            "defined_cost_percentages": int(cost_percent.notna().sum()),
            "cost_wins_vs_reactive": int((group.delta_import_cost_sim_vs_reactive < -1e-9).sum()),
            "cost_losses_vs_reactive": int((group.delta_import_cost_sim_vs_reactive > 1e-9).sum()),
            "cost_down_grid_up_cases": int(group.cost_down_grid_up_vs_reactive.sum())})
    pd.DataFrame(comparisons).to_csv(output / "improvements.csv", index=False)
    complete[complete.strategy != "reactive"].sort_values("delta_import_cost_sim_vs_reactive", ascending=False).to_csv(output / "unfavorable_cases.csv", index=False)
    complete[complete.strategy == "mlp_weighted"].to_csv(output / "weighted_vs_mse.csv", index=False)
    write_json(output / "coverage.json", {"total_cases": len(status), "four_way_complete": len(good_ids),
               "failed_cases": len(status) - len(good_ids), "primary_policy": "complete four-way pairs only; raw partial/failure records retained",
               "statistical_unit": "14 demand days crossed with repeated controlled scenario conditions, not 112 independent days"})


def run_comparison(data_dir, mse_bundle, weighted_bundle, solar_raw, home_config, planner_config,
                   config, run_dir, scenario_dir, previous_verification, *, spot_check=True):
    data_dir, mse_bundle, weighted_bundle, solar_raw, run_dir, scenario_dir = map(
        Path, (data_dir, mse_bundle, weighted_bundle, solar_raw, run_dir, scenario_dir))
    if run_dir.exists() or scenario_dir.exists():
        raise ValueError("New result/scenario directories required")
    protected = (data_dir, mse_bundle.parent, weighted_bundle.parent, solar_raw.parent, Path(previous_verification).parent)
    for output in (run_dir, scenario_dir):
        if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
            raise ValueError("Outputs must be disjoint from input artifacts")
    if run_dir.resolve().is_relative_to(scenario_dir.resolve()) or scenario_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Separate result and scenario roots required")
    home = HomeConfig.load(home_config)
    settings = PlannerConfig(**json.loads(Path(planner_config).read_text(encoding="utf-8")))
    if home.horizon_steps != 24 or home.step_hours != 1 or any(home.export_prices) or settings.terminal_min_soc not in (None, home.min_soc):
        raise ValueError("Keep the Phase 3 hourly 24h, nonexport, reserve-only planning objective")
    if any(v < home.min_soc for v in config.initial_soc_levels):
        raise ValueError("Scenario initial SOC cannot violate reserve")
    before = preservation_snapshot(previous_verification)
    before.update({str(p.resolve()): sha256(p) for p in (Path(__file__).parent / "planner.py", Path(__file__).parent / "simulator.py")})
    before.update({str(p.resolve()): sha256(p) for p in (Path(home_config), Path(planner_config), solar_raw, solar_raw.with_suffix(".metadata.json"))})
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train = _checked_partition(data_dir, "train", manifest)
    validation = _checked_partition(data_dir, "validation", manifest)
    windows = make_windows(validation, history=train)
    records, predictions, model_ids = {}, {}, {}
    for name, bundle in (("mlp_mse", mse_bundle), ("mlp_weighted", weighted_bundle)):
        metadata = json.loads((bundle / "predictor.json").read_text(encoding="utf-8"))
        if metadata["feature_mode"] != "future_calendar" or metadata["config"]["seed"] != config.seed:
            raise ValueError("Expected the frozen calendar pair with predefined seed 2026")
        model_ids[name] = metadata["model_id"]
        path = bundle.parent / "validation_predictions.npz"
        if str(path.resolve()) not in before:
            raise ValueError("Frozen validation forecasts lack verified experiment provenance")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["origins"], windows.origins.to_numpy())
            np.testing.assert_array_equal(saved["actual_kwh"], windows.y)
            predictions[name] = saved["predicted_kwh"].copy()
        records[name] = {"bundle": str(bundle.resolve()), "predictions_path": str(path.resolve()),
                         "model_id": metadata["model_id"], "forecast_origin_policy": "causal frozen Phase 2 validation predictions, no realized targets exposed to provider"}
    episodes, candidates, selection = select_episodes(windows.origins, validation, config.days_per_demand_stratum)
    spot_checks = {}
    if spot_check:
        t = episodes.start.iloc[0]
        history = pd.concat([train, validation.loc[validation.index < t]])
        expected = pd.date_range(t, periods=24, freq="h").to_pydatetime()
        index = windows.origins.get_loc(t)
        for name, bundle in (("mlp_mse", mse_bundle), ("mlp_weighted", weighted_bundle)):
            predictor = MLPDemandPredictor.load(bundle)
            predicted = np.asarray(predictor.predict(history, expected).demand_kwh)
            np.testing.assert_allclose(predicted, predictions[name][index], atol=2e-6, rtol=1e-6)
            spot_checks[name] = {"origin": t.isoformat(), "max_abs_difference_kwh": float(np.abs(predicted - predictions[name][index]).max()),
                                 "last_history_timestamp": history.index[-1].isoformat()}
            del predictor
    solar, solar_metadata = prepare_solar_scenario(solar_raw, home, scenario_dir)
    run_dir.mkdir(parents=True)
    episodes.to_csv(run_dir / "episodes.csv", index=False)
    candidates.to_csv(run_dir / "eligible_days.csv", index=False)
    write_json(run_dir / "design.json", {"comparison": asdict(config), "home": asdict(home), "planner": asdict(settings),
               "selection": selection, "solar": solar_metadata, "models": records,
               "solar_scenarios": "high=100% and low=25% of the same independent Cordoba PVGIS 48h path; synthetic attenuation, not separate weather observations",
               "oracle": "perfect realized 24h demand at every origin; rolling reference, not a production policy or a guaranteed 24h realized-cost lower bound",
               "causality": "future actual demand reaches oracle only; neural forecasts replay prior causal issuance; reactive uses ideal instantaneous current imbalance",
               "aggregation": "equal case weights; scenarios reused over 14 unique demand days; no population-weighted savings claim",
               "spot_checks": spot_checks})
    write_json(run_dir / "reproducibility.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
               "data": {name: manifest["partitions"][name] for name in ("train", "validation")},
               "source_sha256": {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
               "oracle_values_only_in_oracle_provider": True, "new_neural_training": False,
               "search_limits": asdict(settings), "workers": config.workers})
    tasks, case_manifest, predictive_metrics = [], [], []
    for row in episodes.itertuples():
        times = pd.date_range(row.start, periods=24, freq="h")
        indices = windows.origins.get_indexer(times)
        actual = validation.loc[times, "load_kwh"].to_numpy()
        oracle = np.array([validation.loc[pd.date_range(t, periods=24, freq="h"), "load_kwh"].to_numpy() for t in times])
        for name in ("mlp_mse", "mlp_weighted"):
            scores = demand_metrics(oracle, predictions[name][indices])
            predictive_metrics.append({"episode_id": row.episode_id, "strategy": name,
                **{k: scores[k] for k in ("mae_kwh", "rmse_kwh", "bias_kwh")},
                "one_step_mae_kwh": float(np.abs(actual - predictions[name][indices, 0]).mean())})
        for solar_level, tariff, soc in product(config.solar_factors, config.tariff_modes, config.initial_soc_levels):
            case_id = f"{row.episode_id}_{solar_level}_{tariff}_soc{int(round(100 * soc))}"
            scenario_home = replace(home, initial_soc=soc, import_tariff_per_kwh=(config.flat_tariff_sim_per_kwh,) if tariff == "flat" else home.import_tariff_per_kwh)
            case = {"case_id": case_id, "episode_id": row.episode_id, "start": row.start.isoformat(),
                    "demand_stratum": row.demand_stratum, "solar_level": solar_level, "tariff_mode": tariff, "initial_soc": soc,
                    "home": asdict(scenario_home), "planner": asdict(settings), "actual_demand": actual.tolist(),
                    "solar": (solar.solar_kwh.to_numpy()[:48] * config.solar_factors[solar_level]).tolist(),
                    "predictions": {name: predictions[name][indices].tolist() for name in predictions},
                    "oracle_predictions": oracle.tolist(), "model_ids": model_ids}
            tasks.append(case)
            case_manifest.append({k: case[k] for k in ("case_id", "episode_id", "start", "demand_stratum", "solar_level", "tariff_mode", "initial_soc")})
    pd.DataFrame(case_manifest).to_csv(run_dir / "cases.csv", index=False)
    pd.DataFrame(predictive_metrics).to_csv(run_dir / "predictive_metrics.csv", index=False)
    results, by_id = [], {row["case_id"]: row for row in case_manifest}
    started = perf_counter()

    def record(result):
        results.extend({**by_id[result["case_id"]], **metric} for metric in result["metrics"])
        completed_count = len(results) // 4
        write_json(run_dir / "progress.json", {"completed_cases": completed_count, "total_cases": len(tasks),
                   "elapsed_seconds": perf_counter() - started,
                   "failed_strategy_runs": sum(r["status"] != "completed" for r in results)})
        if completed_count % 8 == 0 or completed_count == len(tasks):
            print(f"Completed {completed_count}/{len(tasks)} cases; failed strategy runs={sum(r['status'] != 'completed' for r in results)}", flush=True)

    if config.workers == 1:
        for case in tasks:
            record(run_paired_episode(case, run_dir / "cases" / case["case_id"]))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = {pool.submit(run_paired_episode, case, run_dir / "cases" / case["case_id"]): case for case in tasks}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failure_dir = run_dir / "cases" / case["case_id"]
                    failure_dir.mkdir(parents=True, exist_ok=True)
                    failure = {"exception_type": type(error).__name__, "message": str(error)}
                    write_json(failure_dir / "worker_failure.json", failure)
                    result = {"case_id": case["case_id"], "metrics": [{"strategy": name, "status": "worker_failed", "hours_completed": 0,
                                "is_oracle": name == "oracle"} for name in STRATEGIES]}
                record(result)
    results.sort(key=lambda row: (row["case_id"], STRATEGIES.index(row["strategy"])))
    summarize_results(results, run_dir)
    after = {path: sha256(Path(path)) for path in before}
    write_json(run_dir / "preservation.json", {"verified": before == after,
               "original_artifacts_before": before, "original_artifacts_after": after})
    if before != after:
        raise ValueError("Frozen previous artifacts or fixed configurations changed")
    write_json(run_dir / "completed.json", {"status": "completed", "cases": len(tasks), "strategy_runs": len(results),
               "unique_days": len(episodes), "new_neural_training": False, "elapsed_seconds": perf_counter() - started,
               "failures": sum(row["status"] != "completed" for row in results), "evaluation_partition": "validation"})
    from .comparison_plots import comparison_figures
    comparison_figures(run_dir)
    return pd.DataFrame(results)
