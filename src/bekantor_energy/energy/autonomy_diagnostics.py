"""Independent physical import bound and explicit validation-only diagnostics.

No changes to production objectives, forecasts, search or realization policy.
"""
from collections import deque
from dataclasses import asdict, replace
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import HomeConfig
from ..data.acquisition import write_json
from .continuous import run_continuous_case
from .modes import ModePlanner, ObjectiveConfig
from .planner import PlannerConfig


def import_lower_bound(demand, solar, home):
    """Time-expanded max-flow relaxation, certified by residual min-cut.

    Energies are expressed as deliverable AC kWh. Grid charging cannot improve
    total import with an unlimited always-up grid and round-trip efficiency<=1.
    Unused sources may be discarded in the relaxation. A feasible replay that
    matches the bound therefore proves optimality for this finite scenario.
    """
    demand, solar = np.asarray(demand, float), np.asarray(solar, float)
    if demand.ndim != 1 or not len(demand) or solar.shape != demand.shape:
        raise ValueError("Equal nonempty one-dimensional physical series required")
    if not np.isfinite(demand).all() or not np.isfinite(solar).all() or (demand < 0).any() or (solar < 0).any():
        raise ValueError("Finite nonnegative energy required")
    n = len(demand)
    source, sink = n + 1, n + 2
    graph = [[] for _ in range(n + 3)]
    original = []

    def edge(a, b, capacity):
        graph[a].append([b, float(capacity), len(graph[b])])
        graph[b].append([a, 0., len(graph[a]) - 1])
        original.append((a, b, float(capacity)))

    capacity = home.discharge_efficiency * home.battery_capacity_kwh * (1 - home.min_soc)
    initial = home.discharge_efficiency * home.battery_capacity_kwh * (home.initial_soc - home.min_soc)
    if initial < -1e-9:
        raise ValueError("Initial SOC below reserve")
    edge(source, 0, max(0., initial))
    deficits = np.maximum(demand - solar, 0.)
    for t in range(n):
        surplus = max(0., solar[t] - demand[t])
        edge(source, t, home.charge_efficiency * home.discharge_efficiency * min(surplus, home.max_charge_kw * home.step_hours))
        edge(t, sink, min(deficits[t], home.max_discharge_kw * home.step_hours))
        edge(t, t + 1, capacity)
    total, augmentations = 0., 0
    while True:
        parent = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            node = queue.popleft()
            for index, (other, residual, _) in enumerate(graph[node]):
                if residual > 1e-11 and other not in parent:
                    parent[other] = (node, index)
                    queue.append(other)
        if sink not in parent:
            break
        amount, node = math.inf, sink
        while node != source:
            previous, index = parent[node]
            amount = min(amount, graph[previous][index][1])
            node = previous
        node = sink
        while node != source:
            previous, index = parent[node]
            target, _, reverse = graph[previous][index]
            graph[previous][index][1] -= amount
            graph[target][reverse][1] += amount
            node = previous
        total += amount
        augmentations += 1
    cut = sum(c for a, b, c in original if a in parent and b not in parent)
    if abs(cut - total) > 1e-8:
        raise ArithmeticError("Max-flow/min-cut certificate failed")
    return {"import_lower_bound_kwh": max(0., float(deficits.sum()) - total),
            "max_deliverable_kwh": total, "min_cut_kwh": cut,
            "flow_cut_residual_kwh": total - cut, "augmentations": augmentations,
            "time_nodes": n + 1, "terminal_stock_constraint": "reserve_only"}


def oracle_windows(validation, start, hours):
    """Explicit diagnostic look-ahead; caller must supply validation, never test."""
    times = pd.date_range(start, periods=hours + 23, freq="h")
    selected = validation.reindex(times)
    if not selected.is_observed_complete.fillna(False).all():
        raise ValueError("Oracle requires fully observed validation including 23h tail")
    values = selected.load_kwh.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Oracle targets must be known nonnegative observations")
    return np.stack([values[t:t + 24] for t in range(hours)])


def stock_adjusted_import(import_kwh, initial_kwh, final_kwh, value):
    """Accounting equivalence only; never used as a planning objective."""
    if any(not math.isfinite(v) or v < 0 for v in (import_kwh, initial_kwh, final_kwh, value)):
        raise ValueError("Finite nonnegative stock/accounting values required")
    return import_kwh + value * (initial_kwh - final_kwh)


def ledger_diagnostics(frame, home):
    """Opportunity measures use realized imbalance; they never drive actions."""
    deficit = np.maximum(frame.demand_kwh - frame.solar_kwh, 0.)
    surplus = np.maximum(frame.solar_kwh - frame.demand_kwh, 0.)
    reserve = home.min_soc * home.battery_capacity_kwh
    available = np.minimum(np.minimum(deficit, home.max_discharge_kw * home.step_hours),
                           np.maximum(0., frame.initial_kwh - reserve) * home.discharge_efficiency)
    room = np.minimum(np.minimum(surplus, home.max_charge_kw * home.step_hours),
                      np.maximum(0., home.battery_capacity_kwh - frame.initial_kwh) / home.charge_efficiency)
    return {"grid_charge_kwh": float(frame.grid_charge_kwh.sum()),
            "pv_charge_kwh": float(frame.pv_charge_kwh.sum()),
            "unused_instant_discharge_kwh": float(np.maximum(0., available - frame.discharge_kwh).sum()),
            "unused_instant_pv_charge_kwh": float(np.maximum(0., room - frame.pv_charge_kwh).sum()),
            "conversion_loss_kwh": float(frame.conversion_loss_kwh.sum()),
            "curtailed_kwh": float(frame.curtailed_kwh.sum()),
            "action_adjustments": int(frame.action_adjusted.sum()),
            "max_ac_residual_kwh": float(frame.balance_residual_kwh.abs().max()),
            "max_storage_residual_kwh": float(frame.storage_residual_kwh.abs().max())}


def replay_diagnostic(case, forecasts, step_kwh, method, output_dir):
    """Inject one explicitly labelled forecast source into the existing replay."""
    if method not in ("mlp", "oracle"):
        raise ValueError("Explicit mlp or diagnostic oracle label required")
    modified = dict(case)
    settings = replace(PlannerConfig(**case["planner"]), storage_step_kwh=step_kwh)
    modified.update(predictions=np.asarray(forecasts).tolist(), planner=asdict(settings),
                    model_id="ORACLE_DIAGNOSTIC_NOT_PRODUCTION" if method == "oracle" else case["model_id"])
    result = run_continuous_case(modified, output_dir,
            planning_policies={"autonomy": ModePlanner(settings, ObjectiveConfig("autonomy"))})
    write_json(Path(output_dir) / "diagnostic.json", {"forecast": method,
        "storage_step_kwh": step_kwh, "production_applicable": method != "oracle",
        "objective_unchanged": True, "original_model_id": case["model_id"]})
    return result
