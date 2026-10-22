"""Configurable objectives over the existing physical graph and single A* loop."""
from dataclasses import dataclass, replace
from fractions import Fraction
import math

from ..contracts import EnergyPlan
from .planner import BatterySearchProblem, PlannerConfig, PlanningLimitError
from .search import LexicographicArithmetic, ScalarArithmetic, astar_search


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: str = "economic"
    cost_weight: float = .5
    cost_scale_sim: float = 1.
    grid_scale_kwh: float = 1.

    def __post_init__(self):
        if self.mode not in ("economic", "autonomy", "balanced"):
            raise ValueError("Use economic, autonomy or balanced")
        if not math.isfinite(self.cost_weight) or not 0 <= self.cost_weight <= 1:
            raise ValueError("Dimensionless cost_weight must lie in [0,1]")
        if any(not math.isfinite(v) or v <= 0 for v in (self.cost_scale_sim, self.grid_scale_kwh)):
            raise ValueError("Positive finite monetary and energy normalization scales required")

    @property
    def units(self):
        return ("kWh", "SIM") if self.mode == "autonomy" else ("dimensionless",) if self.mode == "balanced" else ("SIM",)

    def edge_value(self, grid_kwh, cost_sim):
        if any(not math.isfinite(v) or v < 0 for v in (grid_kwh, cost_sim)):
            raise ValueError("Finite nonnegative import and cost required")
        if self.mode == "autonomy":
            return (Fraction.from_float(float(grid_kwh)), Fraction.from_float(float(cost_sim)))
        if self.mode == "economic":
            return cost_sim
        return self.cost_weight * cost_sim / self.cost_scale_sim + (1 - self.cost_weight) * grid_kwh / self.grid_scale_kwh


class ModeSearchProblem(BatterySearchProblem):
    def __init__(self, inputs, initial, settings, objective):
        super().__init__(inputs, initial, settings)
        self.objective = objective
        if objective.mode == "autonomy":
            # Build exact objective equations BEFORE rounding physical import
            # edges. Summing rounded imports exactly can manufacture unequal
            # primary labels for physically tied discharge schedules.
            q = lambda v: Fraction.from_float(float(v))
            self._net_q = tuple(q(d) - q(s) for d, s in zip(inputs.forecast.demand_kwh, inputs.solar_kwh))
            self._price_q = tuple(q(p) for p in inputs.config.import_prices)
            self._charge_eta_q = q(inputs.config.charge_efficiency)
            self._discharge_eta_q = q(inputs.config.discharge_efficiency)
            self._energy_q = {level: q(level) for level in self.levels}

    def successors(self, state):
        edges = self.physical_successors(state)
        if self.objective.mode != "autonomy":
            return [(action, successor, self.objective.edge_value(result.grid_import_kwh, result.cost))
                    for action, successor, result in edges]
        values = []
        for action, successor, _ in edges:
            delta = self._energy_q[successor.stored_kwh] - self._energy_q[state.stored_kwh]
            ac_change = delta / self._charge_eta_q if delta >= 0 else delta * self._discharge_eta_q
            imported = max(Fraction(0), self._net_q[state.step] + ac_change)
            values.append((action, successor, (imported, imported * self._price_q[state.step])))
        return values

    def heuristic(self, state):
        if self.objective.mode == "autonomy":
            return LexicographicArithmetic.zero  # Conservative exact lexicographic UCS.
        if self.settings.heuristic == "zero" or state.step == self.horizon:
            return 0.
        if self.objective.mode == "economic":
            return super().heuristic(state)
        grid = self.grid_lower_bound(state)
        cost = min(self.inputs.config.import_prices[state.step:]) * grid
        return self.objective.edge_value(grid, cost)


@dataclass(frozen=True)
class ModeEnergyPlan(EnergyPlan):
    objective_mode: str = "economic"
    objective_primary: float = 0.
    objective_secondary: float | None = None
    objective_units: tuple = ("SIM",)
    predicted_grid_import_kwh: float = 0.


class ModePlanner:
    """One planner, selected objective. predicted_cost always remains money."""
    def __init__(self, settings=None, objective=None):
        self.objective = objective or ObjectiveConfig()
        self.settings = settings or PlannerConfig()
        if self.objective.mode == "autonomy":
            self.settings = replace(self.settings, heuristic="zero")

    def plan(self, inputs, initial_state):
        problem = ModeSearchProblem(inputs, initial_state, self.settings, self.objective)
        arithmetic = LexicographicArithmetic() if self.objective.mode == "autonomy" else ScalarArithmetic()
        result = astar_search(problem, initial_state, self.settings, PlanningLimitError, arithmetic)
        state, costs, imports = initial_state, [], []
        for h, action in enumerate(result.actions):
            step = problem.simulator.step(state, action, inputs.forecast.demand_kwh[h], inputs.solar_kwh[h], inputs.config)
            state = step.state
            costs.append(step.cost)
            imports.append(step.grid_import_kwh)
        primary = float(result.value[0]) if self.objective.mode == "autonomy" else result.value
        secondary = float(result.value[1]) if self.objective.mode == "autonomy" else None
        return ModeEnergyPlan(result.actions, sum(costs), result.expanded_nodes, inputs.forecast.model_id,
            result.generated_nodes, result.elapsed_seconds, self.settings.heuristic, result.states,
            self.objective.mode, primary, secondary, self.objective.units, sum(imports))
