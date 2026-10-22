"""Continuous paired MPC replay with explicit terminal accounting.

No midnight resets, oracle, neural training or terminal monetary search edges.
The optional terminal condition uses the existing constrained A* graph.
"""
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..config import HomeConfig
from ..contracts import EnergyState, PlanningInputs
from ..data.acquisition import write_json
from .comparison import FrozenForecastProvider, comparison_metrics
from .planner import AStarPlanner, PlannerConfig
from .simulator import EPS, PhysicalSimulator, SelfConsumptionStrategy

STRATEGIES = ("reactive", "mlp_mse", "mlp_mse_terminal")


@dataclass(frozen=True)
class ContinuousConfig:
    days_per_block: int = 7
    initial_soc: float = .5
    terminal_min_soc: float = .5
    flat_tariff_sim_per_kwh: float = .15
    valuation_sim_per_stored_kwh: float = .1425
    valuation_sensitivity: tuple = (0., .076, .1425, .15 / .95, .38)
    solar_sources: dict | None = None
    seed: int = 2026
    workers: int = 2

    def __post_init__(self):
        object.__setattr__(self, "valuation_sensitivity", tuple(self.valuation_sensitivity))
        object.__setattr__(self, "solar_sources", dict(self.solar_sources or {}))
        if type(self.days_per_block) is not int or self.days_per_block < 7:
            raise ValueError("At least seven consecutive days per block required")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (self.initial_soc, self.terminal_min_soc)):
            raise ValueError("Finite initial and terminal SOC in [0,1] required")
        values = (self.flat_tariff_sim_per_kwh, self.valuation_sim_per_stored_kwh, *self.valuation_sensitivity)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Finite nonnegative prices/valuations required")
        if not self.valuation_sensitivity or self.valuation_sim_per_stored_kwh not in self.valuation_sensitivity:
            raise ValueError("Sensitivity must include the primary fixed valuation")
        if len(set(self.valuation_sensitivity)) != len(self.valuation_sensitivity):
            raise ValueError("Duplicate sensitivity values")
        if len(self.solar_sources) != 2:
            raise ValueError("Exactly two independently dated chronological solar sources required")
        if self.seed != 2026 or type(self.workers) is not int or not 1 <= self.workers <= 4:
            raise ValueError("Frozen seed 2026 and one to four workers required")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def adjusted_cost(gross_cost, initial_kwh, final_kwh, value_per_stored_kwh):
    """Resource accounting, not a bill: C_adj = C_grid + v * (E0 - ET).

The same fixed value is used for every policy, date and tariff. Physical stored
kWh already include charge losses; converting its value to AC uses eta_d once.
"""
    if not math.isfinite(gross_cost):
        raise ValueError("Finite cost required")
    if any(not math.isfinite(v) or v < 0 for v in (initial_kwh, final_kwh, value_per_stored_kwh)):
        raise ValueError("Nonnegative finite energy and fixed valuation required")
    return gross_cost + value_per_stored_kwh * (initial_kwh - final_kwh)


def select_continuous_blocks(origins, validation, days=7):
    """First eligible midnight in each chronological validation half, no cost filter.

    Future observed completeness is an offline eligibility check, never an input
    to a controller. Both execution and the last forecast tail stay in validation.
    """
    hours = days * 24
    origins = pd.DatetimeIndex(origins)
    origin_set = set(origins)
    midpoint = validation.index.min() + (validation.index.max() - validation.index.min()) / 2
    candidates = []
    for start in sorted(t for t in origins if t.hour == 0):
        times = pd.date_range(start, periods=hours, freq="h")
        tail = pd.date_range(start, periods=hours + 23, freq="h")
        part = validation.reindex(tail)
        good = all(t in origin_set for t in times) and part.is_observed_complete.fillna(False).all() and np.isfinite(part.load_kwh).all()
        half = "early" if tail[-1] < midpoint else "late" if start >= midpoint else "crosses_midpoint"
        candidates.append({"start": start, "end_exclusive": start + pd.Timedelta(hours=hours),
                           "forecast_tail_end": tail[-1], "half": half, "eligible": bool(good)})
    frame = pd.DataFrame(candidates)
    selected = []
    for half in ("early", "late"):
        pool = frame[(frame.half == half) & frame.eligible]
        if pool.empty:
            raise ValueError(f"No complete consecutive {days}-day block in {half} validation half")
        selected.append({"block_id": half, **pool.iloc[0].to_dict()})
    blocks = pd.DataFrame(selected)
    if blocks.start.iloc[1] < blocks.forecast_tail_end.iloc[0] + pd.Timedelta(hours=1):
        raise ValueError("Selected validation blocks/tails overlap")
    return blocks, frame


def validate_case(case):
    home = HomeConfig(**case["home"])
    settings = PlannerConfig(**case["planner"])
    demand = np.asarray(case["actual_demand"], float)
    solar = np.asarray(case["solar"], float)
    times = pd.date_range(case["start"], periods=len(demand), freq="h")
    if home.step_hours != 1 or home.horizon_steps != 24 or any(home.export_prices):
        raise ValueError("Hourly 24h nonexport case required")
    if settings.terminal_min_soc not in (None, home.min_soc):
        raise ValueError("Reference planner must retain its original reserve-only goal")
    terminal = case["terminal_min_soc"]
    if not math.isfinite(terminal) or not home.min_soc <= terminal <= 1:
        raise ValueError("Terminal target outside physical SOC bounds")
    if demand.ndim != 1 or not len(demand) or solar.shape != (len(demand) + 23,):
        raise ValueError("One continuous realized series and its 23-hour solar tail required")
    if not np.isfinite(demand).all() or (demand < 0).any() or not np.isfinite(solar).all() or (solar < 0).any():
        raise ValueError("Finite nonnegative physical energy required")
    provider = FrozenForecastProvider(times, case["predictions"], case["model_id"])
    return home, settings, demand, solar, times, provider


def run_continuous_case(case, output_dir, *, planner_factory=AStarPlanner, planning_policies=None):
    home, settings, demand, solar, times, provider = validate_case(case)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "case.json", {k: v for k, v in case.items() if k != "predictions"})
    planners = ({"mlp_mse": planner_factory(settings),
                 "mlp_mse_terminal": planner_factory(replace(settings, terminal_min_soc=case["terminal_min_soc"]))}
                if planning_policies is None else dict(planning_policies))
    if not planners or "reactive" in planners:
        raise ValueError("Provide named planning policies separately from the reactive reference")
    strategies = ("reactive", *planners)
    forecast_policy = next(iter(planners))
    states = {name: EnergyState(0, home.initial_soc * home.battery_capacity_kwh) for name in strategies}
    sim, reactive = PhysicalSimulator(), SelfConsumptionStrategy()
    ledger, searches, plans, forecasts, failures = [], [], [], [], {}
    started = perf_counter()
    for t, now in enumerate(times):
        target_times = pd.date_range(now, periods=24, freq="h")
        prices = tuple(home.import_prices[dt.hour] for dt in target_times)
        shifted = replace(home, import_tariff_per_kwh=prices)
        for name in strategies:
            if name in failures:
                continue
            attempt_started = perf_counter()
            try:
                state, plan = states[name], None
                if name == "reactive":
                    action = reactive.action(demand[t], solar[t], home)
                else:
                    forecast = provider(now)  # Provider has no realized demand argument.
                    inputs = PlanningInputs(forecast, tuple(solar[t:t + 24]), shifted, case["case_id"])
                    plan = planners[name].plan(inputs, EnergyState(0, state.stored_kwh))
                    if len(plan.actions) != 24 or len(plan.states) != 25:
                        raise ArithmeticError("Incomplete predicted plan")
                    replay, total = EnergyState(0, state.stored_kwh), 0.
                    plan_rows = []
                    for h, command in enumerate(plan.actions):
                        result = sim.step(replay, command, forecast.demand_kwh[h], solar[t + h], shifted)
                        if abs(result.state.stored_kwh - plan.states[h + 1].stored_kwh) > EPS:
                            raise ArithmeticError("Predicted plan states fail physical replay")
                        plan_rows.append({"strategy": name, "origin": now, "horizon": h + 1,
                            "target_timestamp": target_times[h], "initial_kwh": replay.stored_kwh,
                            "stored_kwh": result.state.stored_kwh, "charge_kw": command.charge_kw,
                            "discharge_kw": command.discharge_kw, "predicted_demand_kwh": forecast.demand_kwh[h],
                            "solar_kwh": solar[t + h], "tariff_sim_per_kwh": prices[h], "cost": result.cost,
                            "grid_import_kwh": result.grid_import_kwh})
                        total += result.cost
                        replay = result.state
                    policy_settings = getattr(planners[name], "settings", settings)
                    terminal = max(home.min_soc, policy_settings.terminal_min_soc or home.min_soc) * home.battery_capacity_kwh
                    if replay.stored_kwh < terminal - EPS or abs(total - plan.predicted_cost) > EPS:
                        raise ArithmeticError("Plan goal/cost failed strict replay")
                    plans.extend(plan_rows)
                    searches.append({"strategy": name, "origin": now, "initial_kwh": state.stored_kwh,
                        "predicted_terminal_kwh": replay.stored_kwh, "required_terminal_kwh": terminal,
                        "predicted_cost": plan.predicted_cost, "seconds": plan.elapsed_seconds,
                        "expanded_nodes": plan.expanded_nodes, "generated_nodes": plan.generated_nodes,
                        "objective_mode": getattr(plan, "objective_mode", "economic"),
                        "objective_primary": getattr(plan, "objective_primary", plan.predicted_cost),
                        "objective_secondary": getattr(plan, "objective_secondary", None),
                        "objective_units": json.dumps(getattr(plan, "objective_units", ("SIM",))),
                        "predicted_grid_import_kwh": sum(r["grid_import_kwh"] for r in plan_rows)})
                    if name == forecast_policy:
                        forecasts.extend({"origin": now, "target_timestamp": dt, "horizon": h + 1,
                                          "predicted_demand_kwh": value, "last_available_history_hour": now - pd.Timedelta(hours=1)}
                                         for h, (dt, value) in enumerate(zip(target_times, forecast.demand_kwh)))
                    action = plan.actions[0]
                # The chronological step/SOC persist through midnight. Only the
                # search uses a relative step zero, without resetting stored energy.
                result = sim.step(state, action, demand[t], solar[t], home, repair=True, import_price=prices[0])
                states[name] = result.state
                values = asdict(result)
                values.pop("state")
                ledger.append({"timestamp": now, "strategy": name, "step": t, **values,
                    "initial_kwh": state.stored_kwh, "stored_kwh": result.state.stored_kwh,
                    "soc": result.state.stored_kwh / home.battery_capacity_kwh,
                    "requested_charge_kw": action.charge_kw, "requested_discharge_kw": action.discharge_kw,
                    "tariff_sim_per_kwh": prices[0], "planning_seconds": plan.elapsed_seconds if plan else 0.,
                    "expanded_nodes": plan.expanded_nodes if plan else 0,
                    "generated_nodes": plan.generated_nodes if plan else 0})
            except Exception as error:
                failures[name] = {"step": t, "timestamp": now.isoformat(), "exception_type": type(error).__name__,
                                  "message": str(error), "attempt_seconds": perf_counter() - attempt_started}
    for filename, rows in (("ledger.csv", ledger), ("searches.csv", searches),
                            ("plans.csv.gz", plans), ("forecasts.csv.gz", forecasts)):
        pd.DataFrame(rows).to_csv(output / filename, index=False)
    metrics = []
    for name in strategies:
        rows = [r for r in ledger if r["strategy"] == name]
        metric = {"strategy": name, "status": "failed" if name in failures else "completed", "hours_completed": len(rows)}
        if rows:
            metric.update(comparison_metrics(rows, home))
            metric["mean_soc"] = float(np.mean([(r["initial_kwh"] + r["stored_kwh"]) / (2 * home.battery_capacity_kwh) for r in rows]))
            metric["adjusted_cost_sim"] = adjusted_cost(metric["import_cost_sim"], metric["battery_initial_kwh"],
                    metric["battery_final_kwh"], case["valuation_sim_per_stored_kwh"])
            metric["terminal_accounting_sim"] = metric["adjusted_cost_sim"] - metric["import_cost_sim"]
        metrics.append(metric)
    write_json(output / "metrics.json", metrics)
    write_json(output / "failures.json", failures)
    return {"case_id": case["case_id"], "metrics": metrics, "failures": failures, "elapsed_seconds": perf_counter() - started}


def summarize_continuous(results, output, valuations, *, strategies=STRATEGIES, reference_strategy="mlp_mse"):
    """Complete paired cases only; all failures remain in raw results/coverage."""
    output = Path(output)
    frame = pd.DataFrame(results)
    frame.to_csv(output / "results_by_case.csv", index=False)
    status = frame.pivot(index="case_id", columns="strategy", values="status").reindex(columns=strategies)
    good_ids = status.index[(status == "completed").all(axis=1)]
    complete = frame[frame.case_id.isin(good_ids)].copy()
    coverage_key = "complete_three_way_cases" if len(strategies) == 3 else "complete_four_way_cases"
    write_json(output / "coverage.json", {"cases": len(status), coverage_key: len(good_ids),
        "complete_paired_cases": len(good_ids), "paired_strategy_count": len(strategies),
        "failed_cases": len(status) - len(good_ids), "failed_strategy_runs": int((frame.status != "completed").sum()),
        "policy": "primary aggregates require every paired strategy complete; no fallback or hidden failures"})
    if complete.empty:
        complete.to_csv(output / "paired_results.csv", index=False)
        return
    metrics = [c for c in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh", "solar_utilization_pct",
        "battery_initial_kwh", "battery_final_kwh", "min_soc", "mean_soc", "final_soc", "curtailed_kwh", "charge_kwh", "discharge_kwh",
        "unserved_kwh", "constraint_violations", "planning_seconds", "expanded_nodes") if c in complete]
    for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
        base = complete[complete.strategy == "reactive"].set_index("case_id")[metric]
        original = complete[complete.strategy == reference_strategy].set_index("case_id")[metric]
        reference_label = "original_mse" if reference_strategy == "mlp_mse" else reference_strategy
        for label, series in (("reactive", base), (reference_label, original)):
            ref = complete.case_id.map(series)
            complete[f"delta_{metric}_vs_{label}"] = complete[metric] - ref
            complete[f"improvement_pct_{metric}_vs_{label}"] = 100 * (ref - complete[metric]) / ref.where(ref.abs() > EPS)
    complete.to_csv(output / "paired_results.csv", index=False)
    for filename, grouping in (("aggregate.csv", ["strategy"]), ("by_scenario.csv", ["solar_profile", "tariff_mode", "strategy"]),
                                ("by_block.csv", ["block_id", "tariff_mode", "strategy"])):
        aggregate = complete.groupby(grouping)[metrics].agg(["count", "mean", "std", "min", "max", "sum"])
        aggregate.columns = ["__".join(v) for v in aggregate.columns]
        aggregate.reset_index().to_csv(output / filename, index=False)
    sensitivity = []
    for value in valuations:
        current = complete[["case_id", "block_id", "solar_profile", "tariff_mode", "strategy", "import_cost_sim",
                             "battery_initial_kwh", "battery_final_kwh"]].copy()
        current["valuation_sim_per_stored_kwh"] = value
        current["adjusted_cost_sim"] = current.import_cost_sim + value * (current.battery_initial_kwh - current.battery_final_kwh)
        base = current[current.strategy == "reactive"].set_index("case_id").adjusted_cost_sim
        current["delta_vs_reactive_sim"] = current.adjusted_cost_sim - current.case_id.map(base)
        sensitivity.append(current)
    pd.concat(sensitivity).to_csv(output / "sensitivity.csv", index=False)
    complete[(complete.strategy != "reactive") & (complete.delta_adjusted_cost_sim_vs_reactive > EPS)].to_csv(
        output / "unfavorable_cases.csv", index=False)
