"""Explicit physical conversions; power (kW) is not energy (kWh)."""

import math


def _nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("A finite nonnegative value is required")
    return value


def watts_to_kw(watts: float) -> float:
    return _nonnegative(watts) / 1000


def wh_to_kwh(wh: float) -> float:
    return _nonnegative(wh) / 1000


def energy_kwh(power_kw: float, duration_hours: float) -> float:
    return _nonnegative(power_kw) * _nonnegative(duration_hours)


def soc_from_energy(stored_kwh: float, capacity_kwh: float) -> float:
    _nonnegative(stored_kwh)
    if not math.isfinite(capacity_kwh) or capacity_kwh <= 0:
        raise ValueError("Capacity must be finite and > 0")
    if stored_kwh > capacity_kwh:
        raise ValueError("Stored energy exceeds capacity")
    return stored_kwh / capacity_kwh
