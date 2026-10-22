"""Headless, inspectable descriptive analysis. Default pipeline scope is train."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .data.acquisition import write_json


def run_eda(hourly: pd.DataFrame, directory: Path, *, scope: str, source_kind: str) -> dict:
    if hourly.empty or "load_kwh" not in hourly:
        raise ValueError("EDA requires nonempty hourly consumption data")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    load = hourly["load_kwh"]
    summary = {"scope": scope, "source_kind": source_kind, "hours": len(hourly),
               "start": hourly.index.min().isoformat(), "end": hourly.index.max().isoformat(),
               "missing_load_hours": int(load.isna().sum()),
               "available_energy_kwh": float(load.sum()) if load.notna().any() else None,
               "energy_note": "Sum over available complete hours only; not a full-period total if gaps exist",
               "statistics": json.loads(load.describe().to_json()),
               "missing_fraction": json.loads(hourly.isna().mean().to_json())}
    write_json(directory / "summary.json", summary)
    hourly.select_dtypes(include="number").describe().to_csv(directory / "descriptive_statistics.csv")
    by_hour = load.groupby(load.index.hour).agg(["mean", "count", "std"])
    by_hour.to_csv(directory / "hour_of_day.csv", index_label="source_wall_clock_hour")
    monthly = load.resample("MS").agg(["sum", "count"])
    monthly["sum"] = monthly["sum"].where(monthly["count"] > 0)
    monthly.to_csv(directory / "monthly_available_energy.csv")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=130)
        plt.close(fig)

    label = f"{source_kind} | {scope}"
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(load.index, load, linewidth=0.7, color="#246b9e")
    ax.set(xlabel="Fecha (reloj original UCI, zona no especificada)", ylabel="Energía horaria (kWh)",
           title=f"Consumo horario — {label}")
    save(fig, "hourly_consumption.png")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if load.notna().any():
        ax.hist(load.dropna(), bins=min(40, max(1, int(load.notna().sum() ** 0.5))), color="#246b9e")
    ax.set(xlabel="Energía horaria (kWh)", ylabel="Horas", title=f"Distribución — {label}")
    save(fig, "distribution.png")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(by_hour.index, by_hour["mean"], marker="o", color="#246b9e")
    ax.set(xlabel="Hora del reloj original (0–23)", ylabel="Media (kWh por intervalo)",
           title=f"Perfil por hora — {label}", xticks=range(0, 24, 3))
    save(fig, "hour_of_day.png")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(hourly.index, hourly["observed_coverage"], linewidth=0.7, color="#246b9e")
    ax.set(xlabel="Fecha", ylabel="Fracción de minutos observados", ylim=(-0.05, 1.05),
           title=f"Cobertura observada (antes del relleno) — {label}")
    save(fig, "coverage.png")
    return summary
