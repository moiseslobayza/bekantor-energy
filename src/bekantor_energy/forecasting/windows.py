"""Causal rolling origins; all target labels stay in their original partition."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def validate_hourly(frame: pd.DataFrame) -> None:
    required = {"load_kwh", "is_observed_complete"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("Nonempty hourly data with load_kwh/is_observed_complete required")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans or index.has_duplicates:
        raise ValueError("Valid unique datetime index required")
    if not index.is_monotonic_increasing or not (index == index.floor("h")).all():
        raise ValueError("Hourly aligned, chronological timestamps required")
    if frame["is_observed_complete"].isna().any() or not frame["is_observed_complete"].isin([True, False]).all():
        raise ValueError("is_observed_complete must contain actual booleans")
    values = frame["load_kwh"].to_numpy(dtype=float)
    if np.isinf(values).any() or (values < 0).any():
        raise ValueError("Demand must be nonnegative, finite or missing")
    if (frame["is_observed_complete"].to_numpy(dtype=bool) & ~np.isfinite(values)).any():
        raise ValueError("Observed-complete targets cannot have unknown energy")


def read_hourly(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"],
                        dtype={"is_observed_complete": "boolean"})
    validate_hourly(frame)
    return frame


@dataclass(frozen=True)
class WindowSet:
    x: np.ndarray
    y: np.ndarray
    origins: pd.DatetimeIndex
    report: dict


def make_windows(partition: pd.DataFrame, *, history: pd.DataFrame | None = None,
                 lookback: int = 168, horizon: int = 24, stride: int = 1) -> WindowSet:
    for name, value in (("lookback", lookback), ("horizon", horizon), ("stride", stride)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    validate_hourly(partition)
    if history is not None:
        validate_hourly(history)
        if history.index.max() >= partition.index.min():
            raise ValueError("History must strictly precede the target partition")
        if str(history.index.tz) != str(partition.index.tz):
            raise ValueError("History and target timezones must agree")
        combined = pd.concat([history.tail(lookback), partition])
    else:
        combined = partition
    # Reindex, never drop missing hours and accidentally join nonadjacent times.
    grid = pd.date_range(combined.index.min(), partition.index.max(), freq="h", name="timestamp")
    regular = combined.reindex(grid)
    values = regular.load_kwh.to_numpy(dtype=float)
    observed = regular.is_observed_complete.fillna(False).to_numpy(dtype=bool)
    start = grid.get_loc(partition.index.min())
    counts = {"candidates": 0, "insufficient_history": 0, "unknown_inputs": 0,
              "unknown_targets": 0, "accepted": 0}
    xs, ys, origins = [], [], []
    for origin in range(start, len(grid) - horizon + 1, stride):
        counts["candidates"] += 1
        if origin < lookback:
            counts["insufficient_history"] += 1
            continue
        x = values[origin - lookback:origin]
        y = values[origin:origin + horizon]
        if not np.isfinite(x).all():
            counts["unknown_inputs"] += 1
            continue
        if not observed[origin:origin + horizon].all() or not np.isfinite(y).all():
            counts["unknown_targets"] += 1
            continue
        xs.append(x)
        ys.append(y)
        origins.append(grid[origin])
    counts["accepted"] = len(xs)
    counts.update({"lookback": lookback, "horizon": horizon, "stride": stride,
                   "partition_start": partition.index.min().isoformat(),
                   "partition_end": partition.index.max().isoformat(),
                   "input_policy": "all hours known; causal phase1 imputations allowed",
                   "target_policy": "all hours observed-complete; no imputed targets"})
    return WindowSet(np.asarray(xs, dtype=np.float32).reshape(-1, lookback),
                     np.asarray(ys, dtype=np.float32).reshape(-1, horizon),
                     pd.DatetimeIndex(origins, name="origin"), counts)
