"""Metrics in physical units, equal weighting over origin/horizon pairs."""

import numpy as np


def demand_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual, predicted = np.asarray(actual, dtype=np.float64), np.asarray(predicted, dtype=np.float64)
    if actual.ndim != 2 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Nonempty equally shaped (origins, horizon) arrays required")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Cannot evaluate unknown or infinite values")
    error = predicted - actual
    mse_h = np.mean(error**2, axis=0)
    return {"origins": len(actual), "horizon": actual.shape[1],
            "evaluated_pairs": int(actual.size),
            "mae_kwh": float(np.mean(np.abs(error))),
            "rmse_kwh": float(np.sqrt(np.mean(error**2))),
            "mse_kwh2": float(np.mean(error**2)), "bias_kwh": float(np.mean(error)),
            "by_horizon": [{"hour": i + 1, "mae_kwh": float(mae), "rmse_kwh": float(np.sqrt(mse))}
                           for i, (mae, mse) in enumerate(zip(np.mean(np.abs(error), axis=0), mse_h))]}
