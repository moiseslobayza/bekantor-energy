"""PVGIS v6 adapter, checked against OpenAPI and live data on 2026-10-22.

This source models solar generation; it does not measure the UCI household.
The v6 `unit` field describes ANGLES, not the power array. `power` is W.
Hourly energy is a rectangle-rule scenario estimate, not measured kWh.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .acquisition import download, provenance, sha256, write_json

PVGIS_ENDPOINT = "https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6/power/broadband"


@dataclass(frozen=True)
class PVGISQuery:
    latitude: float = -34.6037
    longitude: float = -58.3816
    start: str = "2023-01-01T00:00:00"
    end: str = "2023-01-02T23:00:00"
    peak_kwp: float = 3.0
    system_efficiency: float = 0.86
    tilt_deg: float = 30.0
    orientation_deg: float = 0.0  # v6: north=0, east=90, south=180, west=270
    irradiance_source: str = "ERA5"

    def __post_init__(self) -> None:
        bounds = {"latitude": (-90, 90), "longitude": (-180, 180),
                  "tilt_deg": (0, 180), "orientation_deg": (0, 360)}
        for name, (lo, hi) in bounds.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"Invalid {name}")
        if not math.isfinite(self.peak_kwp) or self.peak_kwp <= 0:
            raise ValueError("peak_kwp must be finite and > 0")
        if not 0 < self.system_efficiency <= 1:
            raise ValueError("system_efficiency must be in (0, 1]")
        if self.irradiance_source not in ("ERA5", "SARAH-3"):
            raise ValueError("Use an explicitly supported historical irradiance source")
        start, end = self.bounds
        if start > end or start.year < 2014 or end.year > 2025:
            raise ValueError("Use ordered dates within documented v6 coverage 2014–2025")
        if start != start.floor("h") or end != end.floor("h"):
            raise ValueError("PVGIS adapter requires whole-hour bounds")

    @property
    def bounds(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        return pd.to_datetime(self.start, utc=True), pd.to_datetime(self.end, utc=True)

    @property
    def params(self) -> dict:
        start, end = self.bounds
        return {"latitude": self.latitude, "longitude": self.longitude,
                "start_time": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "end_time": end.strftime("%Y-%m-%dT%H:%M:%S"),
                "frequency": "Hourly", "timezone": "UTC", "irradiance_source": self.irradiance_source,
                "peak-power": self.peak_kwp, "system_efficiency": self.system_efficiency,
                "surface_tilt": self.tilt_deg, "surface_orientation": self.orientation_deg,
                "photovoltaic_module": "cSi:Free standing 2025", "verbose": 1}

    @property
    def url(self) -> str:
        return PVGIS_ENDPOINT + "?" + urlencode(self.params)


def parse_pvgis(payload: dict, query: PVGISQuery) -> pd.DataFrame:
    if not {"power", "timestamps", "peak_power_unit", "peak_power"} <= payload.keys():
        raise ValueError("Unexpected PVGIS v6 schema: required fields absent")
    if payload["peak_power_unit"] != "kWp" or payload["peak_power"] != query.peak_kwp:
        raise ValueError("Unexpected PVGIS peak power metadata")
    expected = pd.date_range(*query.bounds, freq="h", name="timestamp")
    actual = pd.DatetimeIndex(pd.to_datetime(payload["timestamps"], utc=True, errors="raise"),
                              name="timestamp")
    if not actual.equals(expected):
        raise ValueError("PVGIS timestamps differ from requested UTC hourly intervals")
    power = np.asarray(payload["power"], dtype=float)
    if power.ndim != 1 or len(power) != len(expected) or not np.isfinite(power).all() or (power < 0).any():
        raise ValueError("PVGIS power must contain one finite nonnegative W value per hour")
    return pd.DataFrame({"pv_kw": power / 1000, "pv_kwh_estimate": power / 1000,
                         "source_kind": "pvgis_modeled_generation"}, index=actual)


def acquire_pvgis(query: PVGISQuery, destination: Path, *, force: bool = False) -> Path:
    destination = Path(destination)
    receipt = destination.with_suffix(".metadata.json")
    if destination.exists() and not force:
        if not receipt.exists():
            raise ValueError("PVGIS cache lacks metadata; choose a new path or --force")
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        if meta["url"] != query.url or meta["sha256"] != sha256(destination):
            raise ValueError("PVGIS cached query/checksum mismatch; choose a new path or --force")
        parse_pvgis(json.loads(destination.read_text(encoding="utf-8")), query)
        return destination
    download(query.url, destination, max_bytes=64 * 1024**2)
    parse_pvgis(json.loads(destination.read_text(encoding="utf-8")), query)
    write_json(receipt, provenance(destination, query.url, "pvgis_modeled_generation",
                                  api_version="6", query=asdict(query), timezone="UTC",
                                  power_unit="W", interval_hours=1,
                                  energy_assumption="hourly rectangular integration; not measured energy"))
    return destination
