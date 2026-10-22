"""Direct seasonal persistence using only observations before each origin."""

import numpy as np


def seasonal_persistence(x: np.ndarray, *, lag: int, horizon: int = 24) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 2 or type(lag) is not int or type(horizon) is not int:
        raise ValueError("Expected a 2D history and integer lag/horizon")
    if not 0 < horizon <= lag <= x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Require finite history and horizon <= lag <= lookback")
    return x[:, x.shape[1] - lag + np.arange(horizon)].copy()
