"""Single-series normalization fitted on unique observed training hours only.

Inputs are standardized. Targets are divided by the same scale without centering
so a softplus output can enforce nonnegative energy. All reports invert to kWh.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.acquisition import write_json
from .windows import validate_hourly


@dataclass(frozen=True)
class DemandNormalizer:
    mean_kwh: float
    scale_kwh: float
    fitted_hours: int
    fitted_start: str
    fitted_end: str

    def __post_init__(self):
        if not math.isfinite(self.mean_kwh) or not math.isfinite(self.scale_kwh) or self.scale_kwh <= 0:
            raise ValueError("Invalid normalization parameters")
        if self.fitted_hours < 1:
            raise ValueError("Normalizer requires observed training hours")

    @classmethod
    def fit(cls, train: pd.DataFrame) -> "DemandNormalizer":
        validate_hourly(train)
        observed = train.loc[train.is_observed_complete.astype(bool), "load_kwh"]
        if observed.empty:
            raise ValueError("No observed training values for normalization")
        values = observed.to_numpy(dtype=np.float64)
        std = float(values.std())
        return cls(float(values.mean()), std if std > 1e-6 else 1.0, len(values),
                   observed.index.min().isoformat(), observed.index.max().isoformat())

    def inputs(self, values: np.ndarray) -> np.ndarray:
        return ((np.asarray(values) - self.mean_kwh) / self.scale_kwh).astype(np.float32)

    def targets(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) / self.scale_kwh).astype(np.float32)

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) * self.scale_kwh

    def save(self, path: Path) -> None:
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "DemandNormalizer":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
