"""Causal hourly MPC replay; predicted costs and realized costs stay distinct."""
from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
import pandas as pd

from ..contracts import EnergyState, PlanningInputs
from .planner import AStarPlanner, PlannerConfig
from .simulator import PhysicalSimulator, SelfConsumptionStrategy


def realized_metrics(rows, config):
    frame = pd.DataFrame(rows)
    pv = float(frame.solar_kwh.sum())
    used = float((frame.pv_direct_kwh + frame.pv_charge_kwh).sum())
    return {"import_cost_sim": float(frame.cost.sum()), "grid_import_kwh": float(frame.grid_import_kwh.sum()),
            "demand_kwh": float(frame.demand_kwh.sum()), "served_kwh": float(frame.served_kwh.sum()),
            "unserved_kwh": float(frame.unserved_kwh.sum()), "solar_available_kwh": pv,
            "solar_used_on_site_ac_kwh": used, "solar_utilization_pct": 100 * used / pv if pv else None,
            "curtailed_kwh": float(frame.curtailed_kwh.sum()),
            "conversion_loss_kwh": float(frame.conversion_loss_kwh.sum()),
            "battery_initial_kwh": config.initial_soc * config.battery_capacity_kwh,
            "battery_final_kwh": float(frame.stored_kwh.iloc[-1]),
            "min_soc": float(min(config.initial_soc, frame.soc.min())),
            "max_soc": float(max(config.initial_soc, frame.soc.max())),
            "constraint_violations": int(frame.constraint_violations.sum()),
            "adjusted_actions": int(frame.action_adjusted.sum()),
            "max_abs_ac_balance_residual_kwh": float(frame.balance_residual_kwh.abs().max()),
            "max_abs_total_balance_residual_kwh": float(frame.storage_residual_kwh.abs().max()),
            "planning_seconds": float(frame.planning_seconds.sum()),
            "expanded_nodes": int(frame.expanded_nodes.sum()), "generated_nodes": int(frame.generated_nodes.sum())}


def evaluate_receding(actual_demand, timestamps, solar_scenario, forecast_provider, config,
                      settings=None, *, grid_available=None, scenario_id="independent-solar"):
    """Provider sees an origin only; actual demand is used AFTER its decision.

    Conventional feedback is an ideal within-hour self-consumption controller.
    Realized solar is this explicitly known modeled scenario (perfect forecast).
    Optional outages are unmodeled replay disturbances, not part of cost search.
    """
    if config.step_hours != 1.:
        raise ValueError("This evaluator uses hourly UCI data; simulator/search also support other dt")
    demand, solar = np.asarray(actual_demand, float), np.asarray(solar_scenario, float)
    times = pd.DatetimeIndex(timestamps)
    if demand.ndim != 1 or len(demand) != len(times) or not len(demand):
        raise ValueError("Nonempty aligned actual demand and timestamps required")
    if not times.equals(pd.date_range(times[0], periods=len(times), freq="h")):
        raise ValueError("Consecutive hourly replay required")
    if solar.ndim != 1 or len(solar) < len(demand) + config.horizon_steps - 1:
        raise ValueError("Solar path must cover every rolling horizon")
    if not np.isfinite(demand).all() or not np.isfinite(solar).all() or (demand < 0).any() or (solar < 0).any():
        raise ValueError("Finite nonnegative scenario energy required")
    available = (True,) * len(demand) if grid_available is None else tuple(grid_available)
    if len(available) != len(demand) or any(type(v) is not bool for v in available):
        raise ValueError("One actual grid-availability boolean per replay hour required")
    planner, simulator, conventional = AStarPlanner(settings), PhysicalSimulator(), SelfConsumptionStrategy()
    states = {name: EnergyState(0, config.initial_soc * config.battery_capacity_kwh)
              for name in ("self_consumption", "astar_mlp")}
    rows, planning_rows, first_plan_rows, forecasts = [], [], [], []
    start = perf_counter()
    for t, timestamp in enumerate(times):
        # No current realized demand passed into forecasting or planning.
        forecast = forecast_provider(timestamp)
        expected = tuple(pd.date_range(timestamp, periods=config.horizon_steps, freq="h").to_pydatetime())
        if forecast.interval_starts != expected or forecast.issued_at != timestamp.to_pydatetime():
            raise ValueError("Forecast issuance/horizon differs from replay origin")
        prices = tuple(config.import_prices[dt.hour] for dt in pd.date_range(timestamp, periods=config.horizon_steps, freq="h")) if config.horizon_steps == 24 else config.import_prices
        horizon_config = replace(config, import_tariff_per_kwh=prices)
        inputs = PlanningInputs(forecast, tuple(solar[t:t + config.horizon_steps]), horizon_config, scenario_id)
        plan = planner.plan(inputs, EnergyState(0, states["astar_mlp"].stored_kwh))
        planning_rows.append({"timestamp": timestamp, "predicted_cost_next_24h": plan.predicted_cost,
                              "expanded_nodes": plan.expanded_nodes, "generated_nodes": plan.generated_nodes,
                              "seconds": plan.elapsed_seconds, "initial_kwh": states["astar_mlp"].stored_kwh,
                              "model_id": forecast.model_id})
        forecasts.extend({"origin": timestamp, "target_timestamp": dt, "predicted_demand_kwh": value,
                          "known_solar_kwh": solar[t + h], "tariff_sim_per_kwh": prices[h]}
                         for h, (dt, value) in enumerate(zip(forecast.interval_starts, forecast.demand_kwh)))
        if t == 0:
            predicted_state = EnergyState(0, states["astar_mlp"].stored_kwh)
            for h, action in enumerate(plan.actions):
                result = simulator.step(predicted_state, action, forecast.demand_kwh[h], solar[h], horizon_config)
                first_plan_rows.append({"timestamp": forecast.interval_starts[h], "predicted_demand_kwh": forecast.demand_kwh[h],
                                        "solar_kwh": solar[h], "charge_kw": action.charge_kw,
                                        "discharge_kw": action.discharge_kw, "stored_kwh": result.state.stored_kwh,
                                        "grid_import_kwh": result.grid_import_kwh, "predicted_cost": result.cost})
                predicted_state = result.state
        actions = {"astar_mlp": plan.actions[0],
                   "self_consumption": conventional.action(demand[t], solar[t], config)}
        for name, state in states.items():
            result = simulator.step(state, actions[name], demand[t], solar[t], config, repair=True,
                                     grid_available=available[t], import_price=prices[0])
            states[name] = result.state
            ledger = asdict(result)
            ledger.pop("state")
            rows.append({"timestamp": timestamp, "strategy": name, **ledger,
                         "stored_kwh": result.state.stored_kwh,
                         "soc": result.state.stored_kwh / config.battery_capacity_kwh,
                         "requested_charge_kw": actions[name].charge_kw,
                         "requested_discharge_kw": actions[name].discharge_kw,
                         "actual_charge_kw": result.charge_kwh / config.step_hours,
                         "actual_discharge_kw": result.discharge_kwh / config.step_hours,
                         "grid_available": available[t], "tariff_sim_per_kwh": prices[0],
                         "planning_seconds": plan.elapsed_seconds if name == "astar_mlp" else 0.,
                         "expanded_nodes": plan.expanded_nodes if name == "astar_mlp" else 0,
                         "generated_nodes": plan.generated_nodes if name == "astar_mlp" else 0})
    metrics = {name: realized_metrics([row for row in rows if row["strategy"] == name], config) for name in states}
    return {"metrics": metrics, "ledger": pd.DataFrame(rows), "planning": pd.DataFrame(planning_rows),
            "first_plan": pd.DataFrame(first_plan_rows), "forecasts": pd.DataFrame(forecasts),
            "elapsed_seconds": perf_counter() - start}
