"""Hourly AC-bus simulator with an auditable energy ledger and hard reserve."""
import math

from ..config import HomeConfig
from ..contracts import EnergyAction, EnergyState, StepResult

EPS = 1e-9


def finite_nonnegative(value, name):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


class PhysicalSimulator:
    """No simultaneous charge/discharge, no battery export, unlimited grid when up.

    `repair=False` rejects infeasible commands (used inside search). `repair=True`
    is the realization controller: cap commands, serve load first, and never
    borrow energy from below reserve, even during an outage. It does not create
    an unrequested discharge to hide forecast errors.
    """

    def step(self, state: EnergyState, action: EnergyAction, demand_kwh: float,
             solar_kwh: float, config: HomeConfig, *, grid_available: bool = True,
             repair: bool = False, import_price: float | None = None,
             allow_export: bool = False) -> StepResult:
        if type(state.step) is not int or state.step < 0:
            raise ValueError("State step must be a nonnegative integer")
        if type(grid_available) is not bool or type(allow_export) is not bool:
            raise ValueError("Grid/export availability must be boolean")
        for name, value in (("demand", demand_kwh), ("solar", solar_kwh),
                            ("charge_kw", action.charge_kw), ("discharge_kw", action.discharge_kw)):
            finite_nonnegative(value, name)
        if action.charge_kw > 0 and action.discharge_kw > 0:
            raise ValueError("Simultaneous battery charge/discharge is forbidden")
        capacity, reserve, dt = config.battery_capacity_kwh, config.min_soc * config.battery_capacity_kwh, config.step_hours
        stored = state.stored_kwh
        if not math.isfinite(stored) or stored < reserve - EPS or stored > capacity + EPS:
            raise ValueError("Stored energy outside capacity/reserve")
        # Only numerical boundary tolerance; no SOC discretization/rounding here.
        stored = min(capacity, max(reserve, stored))
        price = config.import_prices[state.step % config.horizon_steps] if import_price is None else import_price
        finite_nonnegative(price, "import price")
        requested_charge = action.charge_kw * dt
        requested_discharge = action.discharge_kw * dt
        direct = min(demand_kwh, solar_kwh)
        surplus = solar_kwh - direct
        deficit = demand_kwh - direct
        charge_limit = min(config.max_charge_kw * dt, (capacity - stored) / config.charge_efficiency)
        if not grid_available:
            charge_limit = min(charge_limit, surplus)
        discharge_limit = min(config.max_discharge_kw * dt,
                              max(0., stored - reserve) * config.discharge_efficiency, deficit)
        charge = min(requested_charge, max(0., charge_limit))
        discharge = min(requested_discharge, max(0., discharge_limit))
        adjusted = abs(charge - requested_charge) > EPS or abs(discharge - requested_discharge) > EPS
        if adjusted and not repair:
            raise ValueError("Requested action exceeds power, storage, load or grid availability")
        pv_charge = min(surplus, charge)
        grid_charge = charge - pv_charge
        grid_load = max(0., deficit - discharge) if grid_available else 0.
        imported = grid_load + grid_charge
        unserved = max(0., deficit - discharge - grid_load)
        remainder = max(0., surplus - pv_charge)
        exported = remainder if allow_export and grid_available else 0.
        curtailed = remainder - exported
        next_stored = stored + config.charge_efficiency * charge - discharge / config.discharge_efficiency
        charge_loss = charge * (1 - config.charge_efficiency)
        discharge_loss = discharge * (1 / config.discharge_efficiency - 1)
        loss = charge_loss + discharge_loss
        served = demand_kwh - unserved
        ac_residual = solar_kwh + imported + discharge - (served + charge + exported + curtailed)
        storage_residual = stored + solar_kwh + imported - (next_stored + served + exported + curtailed + loss)
        violations = sum((next_stored < reserve - EPS, next_stored > capacity + EPS,
                          charge > config.max_charge_kw * dt + EPS,
                          discharge > config.max_discharge_kw * dt + EPS,
                          abs(ac_residual) > EPS, abs(storage_residual) > EPS,
                          charge > EPS and discharge > EPS, imported > EPS and exported > EPS))
        if violations:
            raise ArithmeticError("Physical energy invariant failed")
        next_stored = min(capacity, max(reserve, next_stored))  # Floating-point boundary noise only.
        cost = imported * price - exported * config.export_prices[state.step % config.horizon_steps]
        return StepResult(EnergyState(state.step + 1, next_stored), imported, exported, curtailed, cost,
                          demand_kwh, solar_kwh, served, unserved, charge, discharge, direct,
                          pv_charge, grid_charge, loss, ac_residual, storage_residual, adjusted, violations)


class SelfConsumptionStrategy:
    """Ideal instantaneous feedback represented by constant within-hour averages.

    Uses only present imbalance, never forecasts, future tariffs or future load.
    Requests PV-surplus charging or deficit discharging; the same realization
    controller enforces all limits. It never intentionally charges from grid.
    """

    def action(self, demand_kwh, solar_kwh, config):
        finite_nonnegative(demand_kwh, "demand")
        finite_nonnegative(solar_kwh, "solar")
        if solar_kwh > demand_kwh:
            return EnergyAction(charge_kw=(solar_kwh - demand_kwh) / config.step_hours)
        return EnergyAction(discharge_kw=(demand_kwh - solar_kwh) / config.step_hours)
