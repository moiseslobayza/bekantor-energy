"""Reproducible static scientific figures for continuous paired validation."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .continuous import STRATEGIES

COLORS = {"reactive": "#666666", "mlp_mse": "#286b9a", "mlp_mse_terminal": "#bc8125"}
LABELS = {"reactive": "Reactivo", "mlp_mse": "A* MSE original", "mlp_mse_terminal": "A* MSE terminal 50%"}


def finish(fig, path, note="Validación simulada · 2 bloques de 7 días · solar idealizada · costos SIM, no ahorros reales"):
    fig.text(.015, .012, note, fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .045, 1, .96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def continuous_figures(run_dir):
    root = Path(run_dir)
    output = root / "plots"
    output.mkdir(exist_ok=True)
    frame = pd.read_csv(root / "paired_results.csv")
    if frame.empty:
        return
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))
    value = design["configuration"]["valuation_sim_per_stored_kwh"]
    cases = pd.read_csv(root / "cases.csv")
    order = cases.case_id.tolist()
    labels = [f"{'Sep' if r.block_id == 'early' else 'Ene'} / {'PV ene' if r.solar_profile == 'january' else 'PV jul'} / {'plana' if r.tariff_mode == 'flat' else 'variable'}"
              for r in cases.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharex=True, sharey=True)
    for ax, metric, title in zip(axes, ("import_cost_sim", "adjusted_cost_sim"), ("Costo eléctrico bruto", f"Costo ajustado (v={value:g} SIM/kWh almacenado)")):
        for i, name in enumerate(STRATEGIES[1:]):
            rows = frame[frame.strategy == name].set_index("case_id").reindex(order)
            ax.barh(np.arange(len(order)) + (i - .5) * .32, rows[f"delta_{metric}_vs_reactive"],
                    height=.3, color=COLORS[name], label=LABELS[name])
        ax.axvline(0, color="#333333", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Diferencia frente a reactivo (SIM / 7 días)")
        ax.set_yticks(range(len(order)), labels)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle("Cambios pareados de costo: valores negativos favorecen A*")
    finish(fig, output / "paired_costs.png")

    sensitivity = pd.read_csv(root / "sensitivity.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (solar, tariff) in zip(axes.flat, ((s, t) for s in ("january", "july") for t in ("flat", "time_of_use"))):
        subset = sensitivity[(sensitivity.solar_profile == solar) & (sensitivity.tariff_mode == tariff)]
        for (name, block), rows in subset[subset.strategy != "reactive"].groupby(["strategy", "block_id"]):
            rows = rows.sort_values("valuation_sim_per_stored_kwh")
            ax.plot(rows.valuation_sim_per_stored_kwh, rows.delta_vs_reactive_sim,
                color=COLORS[name], linestyle="-" if block == "early" else "--", marker="o", ms=4,
                label=f"{LABELS[name]} / {'Sep' if block == 'early' else 'Ene'}")
        ax.axhline(0, color="#333333", lw=1)
        ax.axvline(value, color="#888888", lw=1, linestyle=":")
        ax.set_title(f"PV {'enero' if solar == 'january' else 'julio'} · tarifa {'plana' if tariff == 'flat' else 'variable'}")
        ax.set_ylabel("Costo ajustado − reactivo (SIM / 7 días)")
        ax.set_xlabel("Valor fijo (SIM/kWh almacenado)")
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Sensibilidad contable del saldo final · línea vertical: valoración principal")
    finish(fig, output / "terminal_sensitivity.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("grid_import_kwh", "solar_utilization_pct"), ("Importación (kWh / 7 días)", "Captación solar (%)")):
        for i, name in enumerate(STRATEGIES):
            for j, tariff in enumerate(("flat", "time_of_use")):
                values = frame[(frame.strategy == name) & (frame.tariff_mode == tariff)][metric].to_numpy()
                x = j + (i - 1) * .23
                ax.bar(x, values.mean(), width=.21, color=COLORS[name], alpha=.7, label=LABELS[name] if j == 0 else None)
                ax.scatter(x + np.linspace(-.045, .045, len(values)), values, s=22, color=COLORS[name], edgecolors="#222222", linewidths=.5)
        ax.set_title(title)
        ax.set_xticks([0, 1], ["Plana", "Variable"])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("Barras: media de 4 cruces por tarifa · puntos: cada bloque/perfil")
    finish(fig, output / "grid_solar.png")

    for case in cases.itertuples():
        directory = root / "cases" / case.case_id
        ledger = pd.read_csv(directory / "ledger.csv", parse_dates=["timestamp"])
        config = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        first = ledger[ledger.strategy == "reactive"]
        x = np.arange(len(first))
        axes[0].plot(x, first.demand_kwh, color="#333333", lw=1, label="Demanda realizada")
        axes[0].plot(x, first.solar_kwh, color="#8a9b53", lw=1, label="PVGIS modelado")
        axes[0].set_ylabel("Energía horaria (kWh)")
        axes[0].legend(loc="upper right", ncol=2, fontsize=9)
        for name in STRATEGIES:
            rows = ledger[ledger.strategy == name]
            if rows.empty:
                continue
            e = np.r_[config["home"]["initial_soc"] * 100, rows.soc.to_numpy() * 100]
            style = "--" if name.endswith("terminal") else "-"
            axes[1].plot(np.arange(len(e)), e, color=COLORS[name], linestyle=style, lw=1.2, label=LABELS[name])
            axes[2].plot(np.arange(len(rows) + 1), np.r_[0, rows.cost.cumsum()], color=COLORS[name], linestyle=style, lw=1.3)
            axes[3].plot(np.arange(len(rows)), rows.charge_kwh - rows.discharge_kwh, color=COLORS[name], linestyle=style, lw=1)
        axes[1].axhline(config["home"]["min_soc"] * 100, color="#333333", linestyle=":", lw=1)
        axes[1].set_ylabel("SOC (%)")
        axes[1].set_ylim(0, 105)
        axes[1].legend(loc="upper right", ncol=3, fontsize=8)
        axes[2].set_ylabel("Costo bruto\nacumulado (SIM)")
        axes[3].set_ylabel("Batería AC (kWh)\n+ carga / − descarga")
        ticks = np.arange(0, len(first) + 1, 24)
        tick_labels = pd.date_range(case.start, periods=len(ticks), freq="D").strftime("%d/%m")
        axes[3].set_xticks(ticks, tick_labels)
        axes[3].set_xlabel("Reloj del escenario UCI · sin reinicio de SOC a medianoche")
        for ax in axes:
            ax.set_xlim(0, len(first))
            ax.grid(alpha=.18)
        fig.suptitle(f"Operación continua · {case.case_id} · {len(first)} horas")
        finish(fig, output / f"trajectory_{case.case_id}.png")
