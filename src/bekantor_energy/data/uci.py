"""Minute consumption validation and causal, quality-aware hourly aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = ("Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
                   "Sub_metering_1", "Sub_metering_2", "Sub_metering_3")
REQUIRED_COLUMNS = ("Date", "Time", *NUMERIC_COLUMNS)


def load_uci(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", na_values=["?", "", "NA"], dtype="str")
    missing = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required UCI columns: {missing}")
    if raw.empty:
        raise ValueError("Empty UCI dataset")
    timestamps = pd.to_datetime(raw["Date"] + " " + raw["Time"],
                                format="%d/%m/%Y %H:%M:%S", errors="raise")
    if timestamps.isna().any() or (timestamps != timestamps.dt.floor("min")).any():
        raise ValueError("UCI requires valid timestamps at whole-minute resolution")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate UCI timestamps: source timezone/DST ambiguity must be reviewed")
    frame = raw[list(NUMERIC_COLUMNS)].apply(pd.to_numeric, errors="raise").astype(float)
    if np.isinf(frame.to_numpy()).any() or (frame < 0).any().any():
        raise ValueError("UCI electrical measurements must be nonnegative and finite or missing")
    frame.index = pd.DatetimeIndex(timestamps, name="timestamp")
    return frame.sort_index()


def hourly_consumption(minutes: pd.DataFrame, *, max_ffill_minutes: int = 5) -> tuple[pd.DataFrame, dict]:
    if type(max_ffill_minutes) is not int or max_ffill_minutes < 0:
        raise ValueError("max_ffill_minutes must be a nonnegative integer")
    if minutes.empty or not isinstance(minutes.index, pd.DatetimeIndex):
        raise ValueError("Nonempty datetime-indexed measurements required")
    if minutes.index.has_duplicates or not minutes.index.is_monotonic_increasing:
        raise ValueError("Minute index must be sorted and unique")
    if not (minutes.index == minutes.index.floor("min")).all():
        raise ValueError("Minute-aligned measurements required")
    if set(NUMERIC_COLUMNS) - set(minutes.columns):
        raise ValueError("Missing electrical columns")
    if np.isinf(minutes.to_numpy(dtype=float)).any() or (minutes < 0).any().any():
        raise ValueError("Invalid electrical measurements")
    grid = pd.date_range(minutes.index.min(), minutes.index.max(), freq="min", name="timestamp")
    regular = minutes.reindex(grid)
    # Never interpolate backward, fit statistics, or use a future observation.
    cleaned = regular.ffill(limit=max_ffill_minutes) if max_ffill_minutes else regular.copy()
    power = cleaned["Global_active_power"]
    available = power.resample("h").count()
    observed = regular["Global_active_power"].resample("h").count()
    imputed = (regular["Global_active_power"].isna() & power.notna()).resample("h").sum()
    hourly = pd.DataFrame({
        "load_kw": power.resample("h").mean().where(available == 60),
        "load_kwh": power.resample("h").sum(min_count=60) / 60,
        "observed_minutes": observed,
        "imputed_minutes": imputed,
        "available_minutes": available,
        "observed_coverage": observed / 60,
        "is_complete": available == 60,
        "is_observed_complete": observed == 60,
    })
    for i in range(1, 4):
        hourly[f"sub_metering_{i}_kwh"] = cleaned[f"Sub_metering_{i}"].resample("h").sum(min_count=60) / 1000
    quality = {"input_rows": len(minutes), "expected_minutes_in_source_span": len(grid),
               "missing_timestamp_minutes": len(grid) - len(minutes),
               "missing_before": regular.isna().sum().astype(int).to_dict(),
               "missing_after": cleaned.isna().sum().astype(int).to_dict(),
               "imputed_values": (regular.isna() & cleaned.notna()).sum().astype(int).to_dict(),
               "hourly_rows": len(hourly), "complete_hours": int(hourly.is_complete.sum()),
               "fully_observed_hours": int(hourly.is_observed_complete.sum()),
               "incomplete_hours": int((~hourly.is_complete).sum()),
               "max_ffill_minutes": max_ffill_minutes,
               "timezone": "unspecified UCI wall-clock; no UTC conversion",
               "policy": "causal bounded forward-fill; incomplete hourly energy remains missing"}
    return hourly, quality
