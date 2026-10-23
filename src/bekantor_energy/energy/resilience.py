"""Experimental outage readiness, separate from the three production objectives.

Only the evaluator observes realized grid availability. Search assumes available
grid, and scores hypothetical four-hour interruptions without their real dates.
"""
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import math

import numpy as np
import pandas as pd

from ..contracts import EnergyAction, EnergyState
from .continuous import adjusted_cost
from .modes import ModeEnergyPlan, ModeSearchProblem, ObjectiveConfig
from .planner import PlannerConfig, PlanningLimitError
from .search import LexicographicArithmetic, astar_search
from .simulator import EPS, PhysicalSimulator, SelfConsumptionStrategy

POLICIES = ("reactive", "fixed_reserve", "adaptive_mlp", "astar_persistence", "astar_mlp", "astar_oracle")


def q(value):
    return Fraction.from_float(float(value))


@dataclass(frozen=True)
class ResilienceConfig:
    critical_fraction: float = .4
    operational_soc: float = .5
    risk_hours: int = 4
    ens_tolerance_kwh: float = 1e-6
    terminal_value: float = .1425

    def __post_init__(self):
        if not math.isfinite(self.critical_fraction) or not 0 < self.critical_fraction <= 1:
            raise ValueError("Critical fraction must be in (0,1]")
        if not math.isfinite(self.operational_soc) or not 0 <= self.operational_soc <= 1:
            raise ValueError("Operational reserve must lie in [0,1]")
        if type(self.risk_hours) is not int or self.risk_hours < 1:
            raise ValueError("Positive integer hypothetical duration required")
        if any(not math.isfinite(v) or v < 0 for v in (self.ens_tolerance_kwh, self.terminal_value)):
            raise ValueError("Nonnegative finite accounting parameters required")


class ReadinessRisk:
    """Exact rational emergency replay; Phi_h(E) in critical AC kWh.

    No actual outage argument exists. Indices 1..H-K are equally weighted;
    current availability is already known, so hypothetical start zero is omitted.
    """
    def __init__(self, demand, solar, home, settings=ResilienceConfig()):
        if len(demand) != len(solar) or len(demand) <= settings.risk_hours:
            raise ValueError("Forecast must cover future starts plus entire hypothetical outage")
        if any(not math.isfinite(float(v)) or v < 0 for v in (*demand, *solar)):
            raise ValueError("Finite nonnegative forecast/solar required")
        self.home, self.settings = home, settings
        self.net = tuple(q(d) * q(settings.critical_fraction) - q(s) for d, s in zip(demand, solar))
        self.reserve = q(home.min_soc * home.battery_capacity_kwh)
        self.capacity = q(home.battery_capacity_kwh)
        self.eta_c, self.eta_d = q(home.charge_efficiency), q(home.discharge_efficiency)
        self.pc, self.pd = q(home.max_charge_kw * home.step_hours), q(home.max_discharge_kw * home.step_hours)
        self.starts = len(demand) - settings.risk_hours
        self.cache = {}

    def phi(self, start, stored):
        if type(start) is not int or not 1 <= start <= self.starts:
            raise ValueError("Hypothetical start must be future and within forecast")
        energy = q(stored)
        if not self.reserve <= energy <= self.capacity:
            raise ValueError("Physical storage required")
        key = (start, energy)
        if key in self.cache:
            return self.cache[key]
        missing = Fraction(0)
        for net in self.net[start:start + self.settings.risk_hours]:
            if net >= 0:
                delivered = min(net, self.pd, (energy - self.reserve) * self.eta_d)
                missing += net - delivered
                energy -= delivered / self.eta_d
            else:
                charged = min(-net, self.pc, (self.capacity - energy) / self.eta_c)
                energy += self.eta_c * charged
        self.cache[key] = missing
        return missing

    def target(self):
        """Minimum energy at next boundary that makes Phi_1 zero, or flag impossible."""
        needed, impossible = self.reserve, False
        for net in reversed(self.net[1:1 + self.settings.risk_hours]):
            if net >= 0:
                impossible |= net > self.pd
                needed += net / self.eta_d
                impossible |= needed > self.capacity
            else:
                needed = max(self.reserve, needed - self.eta_c * min(-net, self.pc))
        return (float(self.capacity), False) if impossible else (float(min(self.capacity, needed)), True)


class ResilienceProblem(ModeSearchProblem):
    def __init__(self, inputs, initial, planner_settings, resilience_settings):
        super().__init__(inputs, initial, planner_settings, ObjectiveConfig("autonomy"))
        self.risk = ReadinessRisk(inputs.forecast.demand_kwh, inputs.solar_kwh, inputs.config, resilience_settings)

    def successors(self, state):
        return [(action, successor,
                 (self.risk.phi(successor.step, successor.stored_kwh) / self.risk.starts
                  if successor.step <= self.risk.starts else Fraction(0), monetary))
                for action, successor, (_, monetary) in super().successors(state)]

    def heuristic(self, state):
        return LexicographicArithmetic.zero


class ResiliencePlanner:
    def __init__(self, settings=None, resilience=None):
        self.settings = replace(settings or PlannerConfig(), heuristic="zero")
        self.resilience = resilience or ResilienceConfig()

    def plan(self, inputs, initial):
        problem = ResilienceProblem(inputs, initial, self.settings, self.resilience)
        result = astar_search(problem, initial, self.settings, PlanningLimitError, LexicographicArithmetic())
        state, monetary, imported = initial, 0., 0.
        for h, action in enumerate(result.actions):
            step = problem.simulator.step(state, action, inputs.forecast.demand_kwh[h], inputs.solar_kwh[h], inputs.config)
            state = step.state
            monetary += step.cost
            imported += step.grid_import_kwh
        return ModeEnergyPlan(result.actions, monetary, result.expanded_nodes, inputs.forecast.model_id,
            result.generated_nodes, result.elapsed_seconds, "zero", result.states,
            "resilience_experimental", float(result.value[0]), float(result.value[1]),
            ("critical_kWh_readiness", "SIM"), imported)


def reserve_action(stored, target, demand, solar, home):
    """Solar autoconsumption with operational reserve; never changes physical floor."""
    reserve = home.min_soc * home.battery_capacity_kwh
    if not reserve - EPS <= stored <= home.battery_capacity_kwh + EPS or not reserve <= target <= home.battery_capacity_kwh:
        raise ValueError("Storage/operational target outside physical bounds")
    if any(not math.isfinite(v) or v < 0 for v in (demand, solar)):
        raise ValueError("Nonnegative finite realized imbalance required")
    surplus = max(0., solar - demand)
    if stored < target - EPS or surplus > 0:
        charge = max(surplus, max(0., target - stored) / home.charge_efficiency)
        return EnergyAction(charge_kw=charge / home.step_hours)
    return EnergyAction(discharge_kw=min(max(0., demand - solar), max(0., stored - target) * home.discharge_efficiency) / home.step_hours)


def realized_step(state, requested, demand, solar, home, grid_up, price, settings):
    """All policies share emergency feedback; only evaluator supplies grid_up."""
    critical = settings.critical_fraction * demand
    noncritical = demand - critical
    dispatch = demand if grid_up else critical
    action = requested if grid_up else SelfConsumptionStrategy().action(critical, solar, home)
    step = PhysicalSimulator().step(state, action, dispatch, solar, home,
            grid_available=grid_up, repair=True, import_price=price)
    ens = 0. if grid_up else step.unserved_kwh
    served_critical = critical - ens
    served_noncritical = noncritical if grid_up else 0.
    sacrificed = 0. if grid_up else noncritical
    service_residual = demand - (served_critical + ens + served_noncritical + sacrificed)
    if abs(service_residual) > EPS or (not grid_up and step.grid_import_kwh > EPS):
        raise ArithmeticError("Critical/noncritical service or blackout import invariant failed")
    values = asdict(step)
    values.pop("state")
    return step, {**values, "initial_kwh": state.stored_kwh, "stored_kwh": step.state.stored_kwh,
        "soc": step.state.stored_kwh / home.battery_capacity_kwh,
        "original_demand_kwh": demand, "critical_required_kwh": critical,
        "noncritical_required_kwh": noncritical, "critical_served_kwh": served_critical,
        "critical_ens_kwh": ens, "noncritical_served_kwh": served_noncritical,
        "noncritical_shed_kwh": sacrificed, "total_unserved_kwh": ens + sacrificed,
        "service_residual_kwh": service_residual, "grid_available": grid_up,
        "requested_charge_kw": action.charge_kw, "requested_discharge_kw": action.discharge_kw}


def resilience_metrics(rows, home, settings, hours_expected):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    cut = frame[~frame.grid_available]
    ens, required = float(cut.critical_ens_kwh.sum()), float(cut.critical_required_kwh.sum())
    deficit = cut.critical_ens_kwh.to_numpy() > settings.ens_tolerance_kwh
    first = int(np.argmax(deficit)) if deficit.any() else len(cut)
    complete = len(frame) == hours_expected
    gross, final = float(frame.cost.sum()), float(frame.stored_kwh.iloc[-1])
    initial = home.initial_soc * home.battery_capacity_kwh
    solar = float(frame.solar_kwh.sum())
    used = float((frame.pv_direct_kwh + frame.pv_charge_kwh).sum())
    return {"hours_completed": len(frame), "outage_hours_observed": len(cut),
        "critical_ens_kwh": ens, "critical_ens_fraction": ens / required if required else None,
        "critical_outage_required_kwh": required, "critical_full_hours": int((~deficit).sum()) if len(cut) else None,
        "initial_autonomy_hours": first if len(cut) else None,
        "autonomy_right_censored": bool(not deficit.any()) if len(cut) else None,
        "no_critical_deficit": bool(complete and ens <= settings.ens_tolerance_kwh) if len(cut) else None,
        "noncritical_shed_kwh": float(frame.noncritical_shed_kwh.sum()),
        "grid_import_kwh": float(frame.grid_import_kwh.sum()), "gross_cost_sim": gross,
        "adjusted_cost_sim": adjusted_cost(gross, initial, final, settings.terminal_value),
        "initial_kwh": initial, "final_kwh": final, "final_soc": final / home.battery_capacity_kwh,
        "outage_initial_kwh": float(cut.initial_kwh.iloc[0]) if len(cut) else None,
        "outage_final_kwh": float(cut.stored_kwh.iloc[-1]) if len(cut) else None,
        "outage_min_soc": float(min(cut.initial_kwh.min(), cut.stored_kwh.min()) / home.battery_capacity_kwh) if len(cut) else None,
        "min_soc": float(min(initial, frame.stored_kwh.min()) / home.battery_capacity_kwh),
        "mean_soc": float(((frame.initial_kwh + frame.stored_kwh) / (2 * home.battery_capacity_kwh)).mean()),
        "solar_used_kwh": used, "solar_utilization_pct": 100 * used / solar if solar else None,
        "curtailed_kwh": float(frame.curtailed_kwh.sum()), "loss_kwh": float(frame.conversion_loss_kwh.sum()),
        "grid_charge_kwh": float(frame.grid_charge_kwh.sum()), "pv_charge_kwh": float(frame.pv_charge_kwh.sum()),
        "discharge_kwh": float(frame.discharge_kwh.sum()),
        "violations": int(frame.constraint_violations.sum()),
        "max_physical_residual_kwh": float(max(frame.balance_residual_kwh.abs().max(), frame.storage_residual_kwh.abs().max(), frame.service_residual_kwh.abs().max()))}
