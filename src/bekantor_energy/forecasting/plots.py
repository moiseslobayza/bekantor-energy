"""Static reproducible forecasting figures, all physical metrics in kWh."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"small": "#246b9e", "medium": "#b78022", "large": "#8a597e",
          "selected_mlp": "#246b9e", "seasonal_24h": "#b78022", "seasonal_168h": "#727e45"}


def _save(fig, directory: Path, name: str):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / name, dpi=140)
    plt.close(fig)


def training_figures(reports: dict, scale_kwh: float, directory: Path):
    fig, axes = plt.subplots(1, len(reports), figsize=(5 * len(reports), 4), squeeze=False, sharey=True)
    for ax, (name, report) in zip(axes[0], reports.items()):
        history = report["history"]
        epochs = np.arange(1, len(history["loss"]) + 1)
        ax.plot(epochs, np.asarray(history["loss"]) * scale_kwh**2,
                label="Entrenamiento", color="#246b9e")
        ax.plot(epochs, np.asarray(history["val_loss"]) * scale_kwh**2,
                label="Validación", color="#b78022", linestyle="--")
        ax.axvline(report["best_epoch"], color="#777777", linewidth=0.8, linestyle=":")
        ax.set(title=name, xlabel="Época", ylabel="MSE + L2 (kWh² eq.)" if report["architecture"]["l2"] else "MSE (kWh²)")
        ax.legend()
    _save(fig, directory, "training_loss.png")
    names = list(reports)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(positions - 0.18, [reports[n]["train"]["rmse_kwh"] for n in names],
           width=0.36, color="#246b9e", label="Train, checkpoint seleccionado")
    ax.bar(positions + 0.18, [reports[n]["validation"]["rmse_kwh"] for n in names],
           width=0.36, color="#b78022", label="Validación")
    ax.set(xticks=positions, xticklabels=names, ylabel="RMSE (kWh)", title="Comparación controlada de arquitecturas")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, directory, "architecture_comparison.png")


def comparison_figures(scores: dict, directory: Path, split: str):
    names = list(scores)
    labels = [name.replace("selected_mlp", "MLP seleccionada") for name in names]
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(positions - 0.18, [scores[n]["mae_kwh"] for n in names], width=0.36,
           label="MAE", color="#246b9e")
    ax.bar(positions + 0.18, [scores[n]["rmse_kwh"] for n in names], width=0.36,
           label="RMSE", color="#b78022")
    ax.set(xticks=positions, xticklabels=labels, ylabel="Error (kWh)", title=f"MLP y baselines — {split}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, directory, f"{split}_baseline_comparison.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in names:
        rows = scores[name]["by_horizon"]
        ax.plot([r["hour"] for r in rows], [r["rmse_kwh"] for r in rows],
                label=name, color=COLORS.get(name, "#777777"), marker="o", markersize=3)
    ax.set(xlabel="Horizonte (horas desde la emisión)", ylabel="RMSE (kWh)",
           title=f"Error según horizonte — {split}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, directory, f"{split}_horizon_error.png")


def forecast_example(origins: pd.DatetimeIndex, actual: np.ndarray, predictions: dict,
                     directory: Path, split: str):
    # First eligible origin is fixed beforehand; no cherry-picking a good day.
    timestamps = pd.date_range(origins[0], periods=actual.shape[1], freq="h")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, actual[0], label="Consumo observado", color="#242424", linewidth=2)
    for name, values in predictions.items():
        ax.plot(timestamps, values[0], label=name, color=COLORS.get(name, "#777777"),
                linestyle="--", linewidth=1.3)
    ax.set(xlabel="Fecha/hora del reloj UCI (zona no especificada)", ylabel="Energía horaria (kWh)",
           title=f"Primer pronóstico elegible de 24 horas — {split}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    _save(fig, directory, f"{split}_actual_vs_predicted.png")
