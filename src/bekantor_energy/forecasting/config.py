"""Explicit bounded experiment configurations for human-supervised iteration."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Architecture:
    name: str
    hidden_units: tuple[int, ...]
    activation: str = "relu"
    dropout: float = 0.0
    l2: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "hidden_units", tuple(self.hidden_units))
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", self.name):
            raise ValueError("Architecture name must be a safe lowercase identifier")
        if not self.hidden_units or any(type(n) is not int or n <= 0 for n in self.hidden_units):
            raise ValueError("Positive integer hidden units required")
        if self.activation not in ("relu", "tanh", "elu"):
            raise ValueError("Supported activations: relu/tanh/elu")
        if not 0 <= self.dropout < 1 or not math.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("Invalid regularization")


@dataclass(frozen=True)
class ExperimentConfig:
    lookback: int = 168
    horizon: int = 24
    stride: int = 1
    seed: int = 2026
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 0.001
    patience: int = 6
    threads: int = 2
    architectures: tuple[Architecture, ...] = (
        Architecture("small", (64, 32)), Architecture("medium", (128, 64)),
        Architecture("large", (256, 128, 64)))

    def __post_init__(self):
        object.__setattr__(self, "architectures", tuple(
            item if isinstance(item, Architecture) else Architecture(**item) for item in self.architectures))
        for name in ("lookback", "horizon", "stride", "epochs", "batch_size", "patience", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.lookback < 168 or self.horizon > 24:
            raise ValueError("Phase 2 requires lookback >= 168 and horizon <= 24 for both baselines")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31)")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Invalid learning rate")
        if not self.architectures or len({a.name for a in self.architectures}) != len(self.architectures):
            raise ValueError("Nonempty unique architecture names required")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
