"""Static scientific figures from paired mode replay, never from synthetic claims."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .mode_experiment import MODE_STRATEGIES

COLORS = {"reactive": "#666666", "economic": "#286b9a", "autonomy": "#71844b", "balanced": "#bc8125"}
LABELS = {"reactive": "Reactivo", "economic": "Económico", "autonomy": "Autonomía", "balanced": "Equilibrado"}
MARKERS = {"reactive": "x", "economic": "o", "autonomy": "^", "balanced": "D"}


def finish(fig, path, note="Validación simulada · 2 bloques de 7 días · solar idealizada · dispersión descriptiva, sin cortes"):
    fig.text(.015, .012, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mode_figures(run_dir):
    root = Path(run_dir)
    output = root / "plots"
    output.mkdir(exist_ok=True)
    rows = pd.read_csv(root / "paired_results.csv")
    if rows.empty:
        return
    cases = pd.read_csv(root / "cases.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, tariff in zip(axes, ("flat", "time_of_use")):
        frame = rows[rows.tariff_mode == tariff]
        for _, group in frame.groupby("case_id"):
            group = group.set_index("strategy").reindex(MODE_STRATEGIES)
            ax.plot(group.grid_import_kwh, group.adjusted_cost_sim, color="#b0b0b0", lw=.8, alpha=.8)
        for name in MODE_STRATEGIES:
            part = frame[frame.strategy == name]
            ax.scatter(part.grid_import_kwh, part.adjusted_cost_sim, marker=MARKERS[name], s=65, color=COLORS[name], label=LABELS[name])
        ax.set_title("Tarifa " + ("plana" if tariff == "flat" else "variable"))
        ax.set_xlabel("Importación de red (kWh / 7 días)")
        ax.set_ylabel("Costo ajustado (SIM / 7 días)")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle("Costo frente a importación · cada línea conecta el mismo bloque y perfil solar")
    finish(fig, output / "cost_grid_tradeoff.png")

    order = cases.case_id.tolist()
    case_labels = [f"{'Sep' if r.block_id == 'early' else 'Ene'} / PV {'ene' if r.solar_profile == 'january' else 'jul'} / {'plana' if r.tariff_mode == 'flat' else 'variable'}" for r in cases.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, metric, label in zip(axes, ("adjusted_cost_sim", "grid_import_kwh"), ("Costo ajustado − reactivo (SIM)", "Importación − reactivo (kWh)")):
        for i, name in enumerate(MODE_STRATEGIES[1:]):
            frame = rows[rows.strategy == name].set_index("case_id").reindex(order)
            ax.barh(np.arange(len(order)) + (i - 1) * .24, frame[f"delta_{metric}_vs_reactive"],
                    height=.22, color=COLORS[name], label=LABELS[name])
        ax.axvline(0, color="#333333", lw=1)
        ax.set_yticks(range(len(order)), case_labels)
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=9)
    fig.suptitle("Diferencias pareadas · valores negativos favorecen al modo respecto del reactivo")
    finish(fig, output / "paired_changes.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    for ax, metric, title in zip(axes, ("min_soc", "mean_soc", "final_soc"), ("SOC mínimo", "SOC medio horario", "SOC final")):
        for i, name in enumerate(MODE_STRATEGIES):
            values = rows[rows.strategy == name][metric].to_numpy() * 100
            ax.scatter(i + np.linspace(-.1, .1, len(values)), values, color=COLORS[name], s=25, alpha=.8)
            ax.plot([i - .18, i + .18], [values.mean()] * 2, color="#333333", lw=2)
        ax.set_xticks(range(4), ["React.", "Econ.", "Auton.", "Equil."])
        ax.set_title(title)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("SOC (%) · puntos: cada caso; raya: media")
    fig.suptitle("Autonomía de importación y energía almacenada son medidas distintas")
    finish(fig, output / "soc_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("planning_seconds", "expanded_nodes"), ("Búsqueda acumulada (s / 7 días)", "Nodos expandidos / 7 días")):
        for i, name in enumerate(MODE_STRATEGIES[1:]):
            values = rows[rows.strategy == name][metric].to_numpy()
            ax.bar(i, values.mean(), color=COLORS[name], width=.6, alpha=.7)
            ax.scatter(i + np.linspace(-.1, .1, len(values)), values, color=COLORS[name], s=20)
        ax.set_xticks(range(3), [LABELS[n] for n in MODE_STRATEGIES[1:]])
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Esfuerzo computacional · barras: media; puntos: ocho casos controlados")
    finish(fig, output / "planning_effort.png")

    sensitivity = pd.read_csv(root / "sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, tariff in zip(axes, ("flat", "time_of_use")):
        frame = sensitivity[sensitivity.tariff_mode == tariff]
        for name in MODE_STRATEGIES[1:]:
            values = frame[frame.strategy == name].groupby("valuation_sim_per_stored_kwh").delta_vs_reactive_sim.mean()
            ax.plot(values.index, values, marker=MARKERS[name], color=COLORS[name], label=LABELS[name])
        ax.axhline(0, color="#333333", lw=1)
        ax.axvline(.1425, color="#888888", lw=1, ls=":")
        ax.set_title("Tarifa " + ("plana" if tariff == "flat" else "variable"))
        ax.set_xlabel("Valor contable (SIM/kWh almacenado)")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Diferencia media de costo ajustado vs reactivo (SIM)")
    axes[0].legend(fontsize=9)
    fig.suptitle("Sensibilidad del saldo final · los pesos de Equilibrado permanecen fijos")
    finish(fig, output / "terminal_sensitivity.png")

    for case in cases.itertuples():
        directory = root / "cases" / case.case_id
        frame = pd.read_csv(directory / "ledger.csv")
        info = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
        initial = info["home"]["initial_soc"] * 100
        for name in MODE_STRATEGIES:
            group = frame[frame.strategy == name]
            if group.empty:
                continue
            hours = np.arange(len(group) + 1)
            axes[0].plot(hours, np.r_[initial, group.soc * 100], color=COLORS[name], lw=1, label=LABELS[name])
            axes[1].plot(hours, np.r_[0, group.cost.cumsum()], color=COLORS[name], lw=1.2)
            axes[2].plot(hours, np.r_[0, group.grid_import_kwh.cumsum()], color=COLORS[name], lw=1.2)
        axes[0].axhline(info["home"]["min_soc"] * 100, color="#333333", lw=1, ls=":")
        axes[0].set_ylabel("SOC (%)")
        axes[0].set_ylim(0, 105)
        axes[0].legend(ncol=4, fontsize=9, loc="upper right")
        axes[1].set_ylabel("Costo bruto acumulado (SIM)")
        axes[2].set_ylabel("Importación acumulada (kWh)")
        ticks = np.arange(0, case.hours + 1, 24)
        axes[2].set_xticks(ticks, pd.date_range(case.start, periods=len(ticks), freq="D").strftime("%d/%m"))
        axes[2].set_xlabel("Reloj del escenario UCI · SOC conservado entre días")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.set_xlim(0, case.hours)
        fig.suptitle("Modos en operación continua · " + case.case_id)
        finish(fig, output / f"trajectory_{case.case_id}.png")
