"""Ports for prediction, physical energy dispatch and classical search.

All per-step arrays refer to the same ordered scenario intervals. A forecast
is issued at the first target interval's boundary and only uses history available
at that issue time. Measured future demand is reserved for evaluation.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .config import HomeConfig


@dataclass(frozen=True)
class LoadForecast:
    issued_at: datetime
    interval_starts: tuple[datetime, ...]
    demand_kwh: tuple[float, ...]
    model_id: str


class DemandPredictor(Protocol):
    """Keras MLP port; preprocessing fit on training data only."""

    def fit(self, train: object, validation: object) -> Mapping[str, float]: ...
    def predict(self, history: object, interval_starts: Sequence[datetime]) -> LoadForecast: ...
    def save(self, path: Path) -> None: ...


@dataclass(frozen=True)
class EnergyState:
    step: int
    stored_kwh: float


@dataclass(frozen=True)
class EnergyAction:
    charge_kw: float = 0.0
    discharge_kw: float = 0.0


@dataclass(frozen=True)
class PlanningInputs:
    forecast: LoadForecast
    solar_kwh: tuple[float, ...]
    config: HomeConfig
    scenario_id: str
    grid_available: tuple[bool, ...] = ()  # Empty means available throughout the forecast.


@dataclass(frozen=True)
class StepResult:
    state: EnergyState
    grid_import_kwh: float
    grid_export_kwh: float
    curtailed_kwh: float
    cost: float
    demand_kwh: float = 0.0
    solar_kwh: float = 0.0
    served_kwh: float = 0.0
    unserved_kwh: float = 0.0
    charge_kwh: float = 0.0  # AC bus, not stored energy.
    discharge_kwh: float = 0.0  # AC bus, after discharge losses.
    pv_direct_kwh: float = 0.0
    pv_charge_kwh: float = 0.0
    grid_charge_kwh: float = 0.0
    conversion_loss_kwh: float = 0.0
    balance_residual_kwh: float = 0.0
    storage_residual_kwh: float = 0.0
    action_adjusted: bool = False
    constraint_violations: int = 0


@dataclass(frozen=True)
class EnergyPlan:
    actions: tuple[EnergyAction, ...]
    predicted_cost: float
    expanded_nodes: int
    forecast_model_id: str
    generated_nodes: int = 0
    elapsed_seconds: float = 0.0
    heuristic_name: str = "zero"
    states: tuple[EnergyState, ...] = ()


class EnergySimulator(Protocol):
    def step(self, state: EnergyState, action: EnergyAction, demand_kwh: float,
             solar_kwh: float, config: HomeConfig) -> StepResult: ...


class SearchProblem(Protocol):
    """Future A*: f(n)=g(n)+h(n); costs use PlanningInputs.forecast."""

    def successors(self, state: EnergyState) -> Sequence[tuple[EnergyAction, EnergyState, float]]: ...
    def heuristic(self, state: EnergyState) -> float: ...
    def is_goal(self, state: EnergyState) -> bool: ...


class EnergyPlanner(Protocol):
    def plan(self, inputs: PlanningInputs, initial_state: EnergyState) -> EnergyPlan: ...


class StrategyEvaluator(Protocol):
    """Replay identical measured demand/solar for all strategies, outside search."""

    def evaluate(self, plans: Mapping[str, EnergyPlan], measured_demand_kwh: Sequence[float],
                 solar_kwh: Sequence[float], config: HomeConfig) -> Mapping[str, Mapping[str, float]]: ...
