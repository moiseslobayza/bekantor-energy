"""Known future calendar covariates; no observed future demand or fitted statistics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .normalization import DemandNormalizer
from .windows import WindowSet

CALENDAR_FIELDS = ("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos")
FEATURE_MODES = ("consumption_only", "future_calendar")


def calendar_features(origins: pd.DatetimeIndex, horizon: int = 24) -> np.ndarray:
    """Shape (origin, horizon, 6), ordered by future interval then field.

    Uses the source clock without inventing a timezone. Month is zero-based for
    encoding (January=0), weekday Monday=0. All timestamps are known at issuance.
    """
    origins = pd.DatetimeIndex(origins)
    if origins.hasnans or type(horizon) is not int or horizon <= 0:
        raise ValueError("Valid origins and positive integer horizon required")
    if not (origins == origins.floor("h")).all():
        raise ValueError("Calendar origins must align with whole-hour intervals")
    if len(origins) == 0:
        return np.empty((0, horizon, 6), dtype=np.float32)
    times = pd.DatetimeIndex([origin + pd.Timedelta(hours=i) for origin in origins for i in range(horizon)])
    values = []
    for coordinate, period in ((times.hour, 24), (times.dayofweek, 7), (times.month - 1, 12)):
        phase = 2 * np.pi * np.asarray(coordinate, dtype=np.float64) / period
        values.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(values).astype(np.float32).reshape(len(origins), horizon, 6)


def with_calendar(windows: WindowSet) -> WindowSet:
    width = windows.y.shape[1] * len(CALENDAR_FIELDS)
    features = calendar_features(windows.origins, windows.y.shape[1]).reshape(len(windows.x), width)
    return WindowSet(np.concatenate([windows.x, features], axis=1), windows.y, windows.origins,
                     {**windows.report, "calendar_fields": list(CALENDAR_FIELDS),
                      "input_size": windows.x.shape[1] + features.shape[1]})


@dataclass(frozen=True)
class CalendarInputNormalizer:
    """Normalize demand only; sin/cos remain in their natural [-1,1] range."""

    demand: DemandNormalizer
    lookback: int
    horizon: int

    @property
    def scale_kwh(self):
        return self.demand.scale_kwh

    def inputs(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] != self.lookback + self.horizon * 6:
            raise ValueError("Expected demand lookback followed by six calendar values per future hour")
        calendar = values[:, self.lookback:]
        if not np.isfinite(values).all() or (np.abs(calendar) > 1.000001).any():
            raise ValueError("Invalid demand/calendar input values")
        return np.concatenate([self.demand.inputs(values[:, :self.lookback]), calendar.astype(np.float32)], axis=1)

    def targets(self, values):
        return self.demand.targets(values)

    def inverse_targets(self, values):
        return self.demand.inverse_targets(values)
