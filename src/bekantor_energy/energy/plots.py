"""Standalone scenario plots, no dashboard or model inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"self_consumption": "#2563a6", "astar_mlp": "#b57a15"}
LABELS = {"self_consumption": "Autoconsumo reactivo", "astar_mlp": "A* + MLP (MPC)"}


def energy_figures(run_dir):
    run_dir = Path(run_dir)
    ledger = pd.read_csv(run_dir / "ledger.csv", parse_dates=["timestamp"])
    first = pd.read_csv(run_dir / "first_plan.csv", parse_dates=["timestamp"])
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["home"]
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18})

    def save(fig, name):
        fig.text(.5, .012, "Escenario simulado · UCI y PVGIS independientes · tarifas SIM ilustrativas · sin ahorros reales demostrados", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .04, 1, 1))
        fig.savefig(plots / name, dpi=160)
        plt.close(fig)

    actual = ledger[ledger.strategy == "astar_mlp"]
    x = np.arange(24)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, actual.demand_kwh, color="#252525", label="Demanda UCI realizada")
    axes[0].plot(x, first.predicted_demand_kwh, color=COLORS["astar_mlp"], linestyle="--", label="Pronóstico inicial MLP")
    axes[0].plot(x, actual.solar_kwh, color="#718436", label="PVGIS modelada (escenario)")
    axes[0].set(ylabel="kWh por hora", title="Demanda realizada, pronóstico inicial y generación del escenario")
    axes[0].legend(ncol=3)
    for name, color in COLORS.items():
        group = ledger[ledger.strategy == name]
        axes[1].step(np.arange(25), np.r_[config["initial_soc"], group.soc], where="post", label=LABELS[name], color=color)
        axes[2].step(x, group.grid_import_kwh, where="mid", label=LABELS[name], color=color)
    axes[1].axhline(config["min_soc"], color="#555555", linestyle=":", label="Reserva mínima")
    axes[1].set(ylabel="SOC (fracción)", ylim=(0, 1.05), title="Estado de carga al inicio de cada intervalo")
    axes[1].legend(ncol=3)
    axes[2].set(ylabel="Importación (kWh)", xlabel="Hora relativa del escenario", title="Importación realizada desde la red")
    axes[2].legend()
    axes[2].set_xticks([0, 4, 8, 12, 16, 20, 24])
    save(fig, "scenario_and_soc.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, name in zip(axes, COLORS):
        group = ledger[ledger.strategy == name]
        ax.bar(x, group.actual_charge_kw, color=COLORS[name], label="Carga AC")
        ax.bar(x, -group.actual_discharge_kw, color=COLORS[name], alpha=.45, label="Descarga AC (signo negativo)")
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(ylabel="Potencia (kW)", title=LABELS[name], ylim=(-config["max_discharge_kw"] - .2, config["max_charge_kw"] + .2))
        ax.legend()
    axes[1].set(xlabel="Hora relativa del escenario")
    save(fig, "realized_decisions.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, column, title, unit in zip(axes, ("cost", "grid_import_kwh", "curtailed_kwh"),
            ("Costo de importación realizado", "Energía importada", "Solar recortada"), ("SIM", "kWh", "kWh")):
        for i, name in enumerate(COLORS):
            ax.bar(i, ledger.loc[ledger.strategy == name, column].sum(), color=COLORS[name])
        ax.set_xticks([0, 1], ["Reactivo", "A* + MLP"])
        ax.set(title=title, ylabel=unit)
        ax.set_ylim(bottom=0)
    save(fig, "comparison.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].step(x, first.charge_kw, where="mid", label="Carga planificada", color=COLORS["astar_mlp"])
    axes[0].step(x, -first.discharge_kw, where="mid", label="Descarga planificada", color=COLORS["self_consumption"])
    axes[0].set(ylabel="Potencia AC (kW)", title="Primer plan completo de 24 h: después se replanifica cada hora")
    axes[0].legend()
    axes[1].plot(np.arange(25), np.r_[config["initial_soc"], first.stored_kwh / config["battery_capacity_kwh"]], color=COLORS["astar_mlp"])
    axes[1].axhline(config["min_soc"], color="#555555", linestyle=":")
    axes[1].set(ylabel="SOC previsto", xlabel="Hora relativa", ylim=(0, 1.05))
    axes[1].set_xlim(0, 24)
    axes[1].set_xticks([0, 4, 8, 12, 16, 20, 24])
    save(fig, "initial_24h_plan.png")
