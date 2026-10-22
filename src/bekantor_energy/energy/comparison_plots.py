"""Static multi-scenario evidence; days are reused across factorial conditions."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"reactive": "#606060", "mlp_mse": "#2563a6", "mlp_weighted": "#b57a15", "oracle": "#718436"}
LABELS = {"reactive": "Reactivo", "mlp_mse": "A* MSE", "mlp_weighted": "A* ponderada", "oracle": "Oracle ideal"}


def comparison_figures(run_dir):
    run_dir = Path(run_dir)
    path = run_dir / "paired_results.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    unique_days = frame.episode_id.nunique()
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18})

    def save(fig, name, note=None):
        fig.text(.5, .012, f"Simulación de validación · escenarios repetidos sobre {unique_days} días · solar idealizada · oracle no operativo", ha="center", fontsize=9)
        if note:
            fig.text(.5, .045, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .09 if note else .05, 1, 1))
        fig.savefig(plots / name, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title, unit in zip(axes, ("import_cost_sim", "grid_import_kwh", "solar_utilization_pct"),
            ("Costo realizado por caso", "Importación realizada", "Captación solar AC"), ("SIM / 24h", "kWh / 24h", "%")):
        values = [frame.loc[frame.strategy == name, metric].to_numpy() for name in COLORS]
        box = ax.boxplot(values, tick_labels=list(LABELS.values()), patch_artist=True, showmeans=True,
                         medianprops={"color": "#252525", "linewidth": 1.3},
                         meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "#252525", "markersize": 5})
        for patch, color in zip(box["boxes"], COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(.55)
        ax.set(title=title, ylabel=unit)
        ax.tick_params(axis="x", rotation=15)
    save(fig, "strategy_distributions.png", "Caja: P25–P75 · línea: mediana · rombo: media · bigotes: 1,5 RIC · círculos: valores externos")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, (solar, tariff) in zip(axes.ravel(), [(s, t) for s in ("low", "high") for t in ("flat", "time_of_use")]):
        subset = frame[(frame.solar_level == solar) & (frame.tariff_mode == tariff)]
        for i, name in enumerate(COLORS):
            values = subset.loc[subset.strategy == name, "import_cost_sim"]
            ax.bar(i, values.mean(), color=COLORS[name], yerr=values.std(), capsize=4)
        ax.set_xticks(range(4), LABELS.values())
        ax.set(title=f"Solar {'baja' if solar == 'low' else 'alta'} · tarifa {'plana' if tariff == 'flat' else 'variable'}",
               ylabel="Costo medio (SIM) ± DE de casos")
        ax.set_ylim(bottom=0)
    bounds = frame.groupby(["solar_level", "tariff_mode", "strategy"]).import_cost_sim.agg(["mean", "std"])
    common_max = float((bounds["mean"] + bounds["std"]).max()) * 1.08
    for ax in axes.ravel():
        ax.set_ylim(0, common_max)
    save(fig, "cost_by_scenario.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ("mlp_mse", "mlp_weighted", "oracle"):
        subset = frame[frame.strategy == name]
        ax.scatter(subset.delta_grid_import_kwh_vs_reactive, subset.delta_import_cost_sim_vs_reactive,
                   color=COLORS[name], marker={"mlp_mse": "o", "mlp_weighted": "^", "oracle": "s"}[name],
                   alpha=.65, s=30, label=LABELS[name])
    ax.axhline(0, color="#555555", linewidth=1)
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set(xlabel="Δ importación frente al reactivo (kWh)", ylabel="Δ costo frente al reactivo (SIM)",
           title="Compromiso entre costo e importación: abajo/derecha = menor costo con más red")
    ax.legend()
    save(fig, "cost_grid_tradeoff.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    weighted = frame[frame.strategy == "mlp_weighted"]
    for ax, metric, title, unit in zip(axes, ("delta_import_cost_sim_vs_mlp_mse", "delta_grid_import_kwh_vs_mlp_mse"),
                                     ("Cambio de costo: ponderada − MSE", "Cambio de red: ponderada − MSE"), ("SIM", "kWh")):
        for i, tariff in enumerate(("flat", "time_of_use")):
            values = weighted.loc[weighted.tariff_mode == tariff, metric].to_numpy()
            offsets = np.linspace(-.12, .12, len(values))
            ax.scatter(i + offsets, np.sort(values), color=COLORS["mlp_weighted"], s=18, alpha=.6)
            ax.scatter(i, values.mean(), color="#252525", marker="D", s=55)
        ax.set_xticks([0, 1], ["Plana", "Variable"])
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set(title=title, ylabel=unit)
    save(fig, "weighted_vs_mse.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, tariff in zip(axes, ("flat", "time_of_use")):
        for name, color in COLORS.items():
            subset = frame[(frame.tariff_mode == tariff) & (frame.strategy == name)].groupby("episode_id").import_cost_sim.mean()
            ax.plot(np.arange(len(subset)), subset.to_numpy(), marker="o", markersize=4, color=color,
                    label=LABELS[name], linestyle="--" if name == "oracle" else "-")
        ax.set(title=f"Costo medio por día: tarifa {'plana' if tariff == 'flat' else 'variable'}",
               xlabel="Episodio cronológico", ylabel="SIM, media de cuatro condiciones solar/SOC")
        ax.set_xticks(range(len(subset)), [str(i + 1) for i in range(len(subset))])
        ax.legend()
    save(fig, "cost_by_day.png")
