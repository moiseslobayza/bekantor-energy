"""Validated scenario assumptions. Tariffs are illustrative, never live prices."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class HomeConfig:
    pv_peak_kwp: float = 3.0
    pv_system_efficiency: float = 0.86
    battery_capacity_kwh: float = 5.0
    max_charge_kw: float = 2.5
    max_discharge_kw: float = 2.5
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    min_soc: float = 0.2
    initial_soc: float = 0.5
    import_tariff_per_kwh: tuple[float, ...] = (0.15,)
    export_tariff_per_kwh: tuple[float, ...] = (0.0,)
    currency: str = "SIM"
    horizon_steps: int = 24
    step_hours: float = 1.0

    def __post_init__(self) -> None:
        for name in ("pv_peak_kwp", "battery_capacity_kwh", "max_charge_kw",
                     "max_discharge_kw", "step_hours"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("charge_efficiency", "discharge_efficiency", "pv_system_efficiency"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 <= self.min_soc <= self.initial_soc <= 1:
            raise ValueError("Require 0 <= min_soc <= initial_soc <= 1")
        if type(self.horizon_steps) is not int or self.horizon_steps <= 0:
            raise ValueError("horizon_steps must be a positive integer")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ValueError("currency is required")
        for name in ("import_tariff_per_kwh", "export_tariff_per_kwh"):
            values = tuple(getattr(self, name))
            if len(values) not in (1, self.horizon_steps):
                raise ValueError(f"{name}: provide one value or horizon_steps values")
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f"{name}: finite, nonnegative tariffs required")
            object.__setattr__(self, name, values)
        if any(sell > buy for buy, sell in zip(self.import_prices, self.export_prices)):
            raise ValueError("Export tariff cannot exceed import tariff in phase 1")

    @property
    def import_prices(self) -> tuple[float, ...]:
        values = self.import_tariff_per_kwh
        return values * self.horizon_steps if len(values) == 1 else values

    @property
    def export_prices(self) -> tuple[float, ...]:
        values = self.export_tariff_per_kwh
        return values * self.horizon_steps if len(values) == 1 else values

    @classmethod
    def load(cls, path: str | Path) -> "HomeConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
