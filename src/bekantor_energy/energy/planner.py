"""A* on (time, exact storage-grid point); h=0 is uniform-cost reference."""
from dataclasses import dataclass
import math

from ..contracts import EnergyAction, EnergyPlan, EnergyState, PlanningInputs
from .simulator import EPS, PhysicalSimulator, finite_nonnegative
from .search import astar_search


class PlanningLimitError(RuntimeError):
    """No optimal plan returned when a resource limit interrupts the search."""


@dataclass(frozen=True)
class PlannerConfig:
    storage_step_kwh: float = .25
    heuristic: str = "relaxed"
    max_expanded: int = 100000
    max_seconds: float = 10.
    terminal_min_soc: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.storage_step_kwh) or self.storage_step_kwh <= 0:
            raise ValueError("Positive finite storage-grid spacing required")
        if self.heuristic not in ("zero", "relaxed"):
            raise ValueError("Use zero or relaxed admissible heuristic")
        if type(self.max_expanded) is not int or self.max_expanded < 1:
            raise ValueError("Positive integer node budget required")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("Positive finite time budget required")
        if self.terminal_min_soc is not None and not 0 <= self.terminal_min_soc <= 1:
            raise ValueError("Terminal SOC must lie in [0,1]")


class BatterySearchProblem:
    def __init__(self, inputs: PlanningInputs, initial: EnergyState, settings: PlannerConfig):
        self.inputs, self.initial, self.settings = inputs, initial, settings
        c, forecast = inputs.config, inputs.forecast
        self.horizon = len(forecast.demand_kwh)
        if self.horizon != c.horizon_steps or len(inputs.solar_kwh) != self.horizon:
            raise ValueError("Forecast, solar and configured horizon must have equal lengths")
        if not self.horizon or len(forecast.interval_starts) != self.horizon:
            raise ValueError("Nonempty complete forecast timestamps required")
        if forecast.issued_at != forecast.interval_starts[0]:
            raise ValueError("Forecast must be issued at the first target boundary")
        for a, b in zip(forecast.interval_starts, forecast.interval_starts[1:]):
            if abs((b - a).total_seconds() / 3600 - c.step_hours) > EPS:
                raise ValueError("Forecast intervals must be consecutive physical time steps")
        if initial.step != 0:
            raise ValueError("A planning horizon starts at relative step zero")
        if any(c.export_prices):
            raise ValueError("A* Phase 3 forbids export revenue; nonnegative edges are required")
        for value in (*forecast.demand_kwh, *inputs.solar_kwh):
            finite_nonnegative(value, "forecast/solar energy")
        self.available = inputs.grid_available or (True,) * self.horizon
        if len(self.available) != self.horizon or any(type(v) is not bool for v in self.available):
            raise ValueError("Grid availability must be one boolean per interval")
        if not all(self.available):
            raise ValueError("This cost-only planner assumes available grid; unannounced outages are handled in replay")
        self.reserve = c.min_soc * c.battery_capacity_kwh
        if not math.isfinite(initial.stored_kwh) or not self.reserve <= initial.stored_kwh <= c.battery_capacity_kwh:
            raise ValueError("Initial energy must respect capacity and reserve")
        self.terminal = max(c.min_soc, settings.terminal_min_soc or c.min_soc) * c.battery_capacity_kwh
        steps = math.floor((c.battery_capacity_kwh - self.reserve) / settings.storage_step_kwh)
        if steps > 10000:
            raise ValueError("Storage grid too large; choose a coarser discretization")
        # Always include physical boundaries, exact initial energy, and optional terminal target.
        self.levels = tuple(sorted(set([self.reserve + i * settings.storage_step_kwh for i in range(steps + 1)]
                           + [c.battery_capacity_kwh, initial.stored_kwh, self.terminal])))
        self.simulator = PhysicalSimulator()

    def physical_successors(self, state):
        """Shared graph edges and physical ledgers, independent of objective."""
        if state.step >= self.horizon:
            return []
        c, t = self.inputs.config, state.step
        successors = []
        for target in self.levels:
            delta = target - state.stored_kwh
            action = (EnergyAction(charge_kw=delta / (c.charge_efficiency * c.step_hours)) if delta > 0
                      else EnergyAction(discharge_kw=-delta * c.discharge_efficiency / c.step_hours))
            try:
                result = self.simulator.step(state, action, self.inputs.forecast.demand_kwh[t],
                                             self.inputs.solar_kwh[t], c)
            except ValueError:
                continue  # Infeasible physical edge is absent, not rounded/clipped into existence.
            if abs(result.state.stored_kwh - target) > EPS or result.cost < -EPS:
                raise ArithmeticError("Invalid state-grid transition or negative edge")
            successors.append((action, EnergyState(t + 1, target), result))
        return successors

    def successors(self, state):
        return [(action, successor, result.cost)
                for action, successor, result in self.physical_successors(state)]

    def heuristic(self, state):
        if self.settings.heuristic == "zero" or state.step == self.horizon:
            return 0.
        return min(self.inputs.config.import_prices[state.step:]) * self.grid_lower_bound(state)

    def grid_lower_bound(self, state):
        """Continuous optimistic AC import bound; ignores timing/power/terminal."""
        if state.step == self.horizon:
            return 0.
        c = self.inputs.config
        demand = self.inputs.forecast.demand_kwh[state.step:]
        solar = self.inputs.solar_kwh[state.step:]
        deficits = sum(max(d - s, 0.) for d, s in zip(demand, solar))
        surplus = sum(max(s - d, 0.) for d, s in zip(demand, solar))
        usable = (state.stored_kwh - self.reserve) * c.discharge_efficiency
        optimistic_stored_solar = surplus * c.charge_efficiency * c.discharge_efficiency
        grid_lower_bound = max(0., deficits - usable - optimistic_stored_solar)
        # Ignores capacity, power, chronology and terminal replenishment, and buys
        # all remaining required energy at its cheapest tariff. Thus optimistic.
        return grid_lower_bound

    def is_goal(self, state):
        return state.step == self.horizon and state.stored_kwh >= self.terminal - EPS


class AStarPlanner:
    def __init__(self, settings: PlannerConfig | None = None):
        self.settings = settings or PlannerConfig()

    def plan(self, inputs, initial_state):
        problem = BatterySearchProblem(inputs, initial_state, self.settings)
        return self.search(problem, initial_state)

    def search(self, problem, initial_state):
        """Reopen improved states; correct also for admissible inconsistent h."""
        result = astar_search(problem, initial_state, self.settings, PlanningLimitError)
        return EnergyPlan(result.actions, result.value, result.expanded_nodes,
                          problem.inputs.forecast.model_id, result.generated_nodes,
                          result.elapsed_seconds, self.settings.heuristic, result.states)
