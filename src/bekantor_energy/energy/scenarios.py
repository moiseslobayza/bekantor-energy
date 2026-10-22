"""Explicit scenario mapping; independent UCI and PVGIS clocks are never joined."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.acquisition import sha256, write_json
from ..data.pvgis import PVGISQuery, parse_pvgis


def prepare_solar_scenario(raw_path, config, output_dir=None):
    raw_path = Path(raw_path)
    receipt = json.loads(raw_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if receipt["sha256"] != sha256(raw_path):
        raise ValueError("Solar source checksum mismatch")
    query = PVGISQuery(**receipt["query"])
    if query.peak_kwp != config.pv_peak_kwp or query.system_efficiency != config.pv_system_efficiency:
        raise ValueError("PVGIS sizing/efficiency must match home; reacquire rather than double-scaling")
    source = parse_pvgis(json.loads(raw_path.read_text(encoding="utf-8")), query)
    local = source.index.tz_convert("America/Argentina/Cordoba")
    if local[0].hour != 0 or len(local) < 47:
        raise ValueError("Provide at least 47 hourly solar samples starting at Cordoba local midnight")
    mapped = pd.DataFrame({"scenario_step": np.arange(len(source)), "pvgis_timestamp_utc": source.index,
                           "pvgis_timestamp_cordoba": local,
                           "solar_kwh": source.pv_kwh_estimate.to_numpy(),
                           "source_kind": "pvgis_modeled_independent_scenario"})
    metadata = {"source": receipt, "query": asdict(query), "source_local_timezone": "America/Argentina/Cordoba",
                "local_period": [local[0].isoformat(), local[-1].isoformat()],
                "mapping": "ordinal scenario hour from PVGIS local midnight to selected naive UCI validation midnight",
                "synchronization": "independent households/locations/years; not synchronized measurements",
                "solar_forecast_assumption": "perfect knowledge of this modeled scenario path, not a weather forecast",
                "energy_conversion": "W / 1000 * 1h; PVGIS system efficiency already included"}
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=False)
        mapped.to_csv(output_dir / "solar_scenario.csv", index=False)
        write_json(output_dir / "solar_scenario.json", metadata)
    return mapped, metadata


def first_complete_midnight(origins, validation, hours=24):
    """Predeclared ex-post completeness filter; never select on cost or load size."""
    origin_set = set(origins)
    for origin in origins:
        if origin.hour != 0:
            continue
        times = pd.date_range(origin, periods=hours, freq="h")
        if all(t in origin_set for t in times):
            part = validation.reindex(times)
            if part.is_observed_complete.fillna(False).all() and np.isfinite(part.load_kwh).all():
                return origin
    raise ValueError("No complete validation midnight with consecutive forecast origins")
