"""Small, explainable forward-chaining filter for proposed battery actions.

Facts describe one candidate transition. The physical simulator remains the
authority: every accepted edge is still executed with repair=False.
"""
from dataclasses import dataclass
import math

from ..config import HomeConfig
from ..contracts import EnergyAction, EnergyState
from .simulator import EPS


@dataclass(frozen=True)
class Rule:
    name: str
    premises: frozenset[str]
    conclusion: str
    explanation: str


@dataclass(frozen=True)
class FiredRule:
    rule: Rule
    supporting_facts: frozenset[str]


@dataclass(frozen=True)
class InferenceResult:
    initial_facts: frozenset[str]
    facts: frozenset[str]
    fired: tuple[FiredRule, ...]

    @property
    def admissible(self) -> bool:
        # Rejection has priority if contradictory facts are ever supplied.
        return "discard_from_search" not in self.facts


RULES = (
    Rule("R1", frozenset({"discharge_requested", "reserve_would_be_breached"}),
         "energy_constraint_violated", "La descarga superaría la energía disponible sobre la reserva."),
    Rule("R2", frozenset({"charge_requested", "capacity_would_be_exceeded"}),
         "energy_constraint_violated", "La carga superaría la capacidad de almacenamiento."),
    Rule("R3c", frozenset({"charge_power_exceeded"}), "power_constraint_violated",
         "La potencia de carga supera el máximo configurado."),
    Rule("R3d", frozenset({"discharge_power_exceeded"}), "power_constraint_violated",
         "La potencia de descarga supera el máximo configurado."),
    Rule("R4", frozenset({"charge_requested", "discharge_requested"}),
         "operation_constraint_violated", "No se permite cargar y descargar simultáneamente."),
    Rule("R5", frozenset({"discharge_exceeds_deficit"}), "energy_constraint_violated",
         "La descarga excede la demanda restante tras usar energía solar."),
    Rule("R6", frozenset({"grid_unavailable", "charge_exceeds_solar_surplus"}),
         "energy_constraint_violated", "Sin red sólo puede cargarse con excedente solar."),
    Rule("R7e", frozenset({"energy_constraint_violated"}), "action_forbidden",
         "Una restricción energética prohíbe la acción."),
    Rule("R7p", frozenset({"power_constraint_violated"}), "action_forbidden",
         "Una restricción de potencia prohíbe la acción."),
    Rule("R7o", frozenset({"operation_constraint_violated"}), "action_forbidden",
         "Una restricción de operación prohíbe la acción."),
    Rule("R8", frozenset({"action_forbidden"}), "discard_from_search",
         "A* no genera la transición prohibida."),
)


def forward_chain(initial_facts: frozenset[str] | set[str], rules: tuple[Rule, ...] = RULES) -> InferenceResult:
    """Apply rules to a fixed point; record the premises available when each fired."""
    known = set(initial_facts)
    fired = []
    used = set()
    changed = True
    while changed:
        changed = False
        for rule in rules:
            if rule.name not in used and rule.premises <= known:
                used.add(rule.name)
                fired.append(FiredRule(rule, rule.premises))
                if rule.conclusion not in known:
                    known.add(rule.conclusion)
                    changed = True
    return InferenceResult(frozenset(initial_facts), frozenset(known), tuple(fired))


def action_facts(state: EnergyState, action: EnergyAction, demand_kwh: float,
                 solar_kwh: float, config: HomeConfig, *, grid_available: bool = True) -> frozenset[str]:
    """Convert the simulator's AC limits into facts, in kWh per time step.

    Input validity is the simulator's responsibility; this filter receives
    validated planner candidates and does not repair or round an action.
    """
    if (not all(math.isfinite(x) and x >= 0 for x in
                (action.charge_kw, action.discharge_kw, demand_kwh, solar_kwh, state.stored_kwh))
            or type(grid_available) is not bool):
        raise ValueError("Finite nonnegative action facts and boolean grid availability required")
    capacity = config.battery_capacity_kwh
    reserve = config.min_soc * capacity
    if type(state.step) is not int or state.step < 0 or not reserve - EPS <= state.stored_kwh <= capacity + EPS:
        raise ValueError("State must respect the simulator's step, capacity and reserve limits")
    # The simulator clamps only boundary noise before computing its limits.
    stored = min(capacity, max(reserve, state.stored_kwh))
    charge = action.charge_kw * config.step_hours
    discharge = action.discharge_kw * config.step_hours
    surplus = max(0., solar_kwh - demand_kwh)
    deficit = max(0., demand_kwh - solar_kwh)
    facts = {"candidate_action"}
    if action.charge_kw > 0:
        facts.add("charge_requested")
    if action.discharge_kw > 0:
        facts.add("discharge_requested")
    if not grid_available:
        facts.add("grid_unavailable")
    if charge > (capacity - stored) / config.charge_efficiency + EPS:
        facts.add("capacity_would_be_exceeded")
    if discharge > max(0., stored - reserve) * config.discharge_efficiency + EPS:
        facts.add("reserve_would_be_breached")
    if charge > config.max_charge_kw * config.step_hours + EPS:
        facts.add("charge_power_exceeded")
    if discharge > config.max_discharge_kw * config.step_hours + EPS:
        facts.add("discharge_power_exceeded")
    if discharge > deficit + EPS:
        facts.add("discharge_exceeds_deficit")
    if not grid_available and charge > surplus + EPS:
        facts.add("charge_exceeds_solar_surplus")
    return frozenset(facts)


def evaluate_action(state: EnergyState, action: EnergyAction, demand_kwh: float,
                    solar_kwh: float, config: HomeConfig, *, grid_available: bool = True) -> InferenceResult:
    return forward_chain(action_facts(state, action, demand_kwh, solar_kwh, config,
                                      grid_available=grid_available))
