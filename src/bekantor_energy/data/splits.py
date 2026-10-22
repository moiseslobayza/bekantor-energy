"""Chronological partitions before any learned preprocessing or window building."""

import math

import pandas as pd


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.7,
                        validation_fraction: float = 0.15) -> dict[str, pd.DataFrame]:
    if not all(math.isfinite(v) and v > 0 for v in (train_fraction, validation_fraction)):
        raise ValueError("Fractions must be finite and positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Leave a nonempty test fraction")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans:
        raise ValueError("Valid DatetimeIndex required")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Sorted unique timestamps required")
    n_train = int(len(frame) * train_fraction)
    n_validation = int(len(frame) * validation_fraction)
    if min(n_train, n_validation, len(frame) - n_train - n_validation) < 1:
        raise ValueError("Not enough hours for three nonempty chronological partitions")
    return {"train": frame.iloc[:n_train].copy(),
            "validation": frame.iloc[n_train:n_train + n_validation].copy(),
            "test": frame.iloc[n_train + n_validation:].copy()}
