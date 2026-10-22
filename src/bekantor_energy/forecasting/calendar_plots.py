"""Reproducible static scientific figures for the paired CPU experiment."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"reference": "#2563a6", "calendar": "#b57a15"}
LABELS = {"reference": "Original", "calendar": "Calendario"}


def calendar_figures(run_dir, model_dir):
    run_dir, model_dir = Path(run_dir), Path(model_dir)
    aggregate = pd.read_csv(run_dir / "metrics_aggregate.csv", dtype={"group": str})
    per_seed = pd.read_csv(run_dir / "metrics_by_seed.csv", dtype={"group": str})
    fits = pd.read_csv(run_dir / "fits.csv")
    n_seeds = fits.seed.nunique()
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "white"})

    def finish(fig, name, note=None):
        if note is None:
            note = f"Validación · barras/bandas: media ± DE muestral entre {n_seeds} semillas; no intervalo predictivo"
        fig.text(0.5, 0.015, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
        fig.savefig(plots / name, dpi=160)
        plt.close(fig)

    def bars(ax, scope, group, metric, title, unit):
        data = aggregate[(aggregate.scope == scope) & (aggregate.group == group) & (aggregate.metric == metric)]
        for x, variant in enumerate(COLORS):
            row = data[data.variant == variant].iloc[0]
            ax.bar(x, row["mean"], yerr=row["std"], color=COLORS[variant], capsize=5, width=0.6)
            points = per_seed[(per_seed.scope == scope) & (per_seed.group == group) &
                              (per_seed.metric == metric) & (per_seed.variant == variant)]
            ax.scatter(np.full(len(points), x), points.value, color="#252525", s=18, zorder=5)
        ax.set_xticks([0, 1], ["Original", "Calendario"])
        ax.set(title=title, ylabel=unit)
        ax.axhline(0, color="#555555", linewidth=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, metric in zip(axes, ("mae_kwh", "rmse_kwh")):
        bars(ax, "global", "all", metric, metric.split("_")[0].upper() + " global", "kWh por hora")
    finish(fig, "global_comparison.png")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, title, unit in zip(axes.ravel(),
            ("mae_kwh", "rmse_kwh", "bias_kwh", "underestimate_pct"),
            ("MAE en picos", "RMSE en picos", "Sesgo en picos (predicho − real)", "Subestimación en picos"),
            ("kWh por hora", "kWh por hora", "kWh por hora", "% de pares en picos")):
        bars(ax, "high_consumption", "q90", metric, title, unit)
    threshold = json.loads((run_dir / "thresholds.json").read_text())["thresholds_kwh"]["q90"]
    fig.suptitle(f"Consumo elevado: real > P90 de train = {threshold:.4f} kWh")
    finish(fig, "peak_comparison.png")

    for scope, xlabel, name in (("hour", "Hora del objetivo (reloj UCI)", "error_by_hour.png"),
                                ("horizon", "Horizonte (horas)", "error_by_horizon.png")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, metric in zip(axes, ("mae_kwh", "rmse_kwh", "bias_kwh")):
            for variant, color in COLORS.items():
                data = aggregate[(aggregate.scope == scope) & (aggregate.variant == variant) &
                                 (aggregate.metric == metric)].copy()
                data["position"] = data.group.astype(int)
                data = data.sort_values("position")
                x, y, std = data.position.to_numpy(), data["mean"].to_numpy(), data["std"].to_numpy()
                ax.plot(x, y, label=LABELS[variant], color=color, marker="o", markersize=3,
                        linestyle="-" if variant == "reference" else "--")
                ax.fill_between(x, y - std, y + std, color=color, alpha=0.15)
            ax.axhline(0, color="#555555", linewidth=0.8)
            ax.set(xlabel=xlabel, ylabel="kWh por hora", title=metric.split("_")[0].upper())
            ax.set_xticks([0, 4, 8, 12, 16, 20, 23] if scope == "hour" else [1, 4, 8, 12, 16, 20, 24])
            ax.legend()
        finish(fig, name)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, scope, group, title in ((axes[0], "global", "all", "RMSE global por semilla"),
                                    (axes[1], "high_consumption", "q90", "RMSE en picos por semilla")):
        data = per_seed[(per_seed.scope == scope) & (per_seed.group == group) & (per_seed.metric == "rmse_kwh")]
        for seed in sorted(data.seed.unique()):
            points = data[data.seed == seed].set_index("variant").loc[list(COLORS)]
            ax.plot([0, 1], points.value, color="#777777", linewidth=1)
            ax.annotate(str(seed), (0, points.value.iloc[0]), xytext=(-8, 4), textcoords="offset points", ha="right", fontsize=9)
            ax.scatter([0, 1], points.value, c=list(COLORS.values()), s=35)
        ax.set_xticks([0, 1], ["Original", "Calendario"])
        ax.set(xlim=(-0.4, 1.3), ylabel="kWh por hora", title=title)
    finish(fig, "paired_seeds.png", "Validación · cada línea conecta la misma semilla; ejes verticales independientes para ver las diferencias")

    scale = json.loads((run_dir / "normalization.json").read_text())["scale_kwh"]
    seeds = sorted(fits.seed.unique())
    fig, axes = plt.subplots(1, len(seeds), figsize=(5 * len(seeds), 4.5), squeeze=False, sharey=True)
    for ax, seed in zip(axes[0], seeds):
        for variant, color in COLORS.items():
            history = pd.read_csv(model_dir / f"seed_{seed}" / variant / "epochs.csv")
            ax.plot(history.epoch + 1, history.loss * scale ** 2, color=color, alpha=0.6, linestyle=":")
            ax.plot(history.epoch + 1, history.val_loss * scale ** 2, color=color, label=LABELS[variant] + " val")
        ax.set(title=f"Semilla {seed}", xlabel="Época", ylabel="MSE (kWh²)")
        ax.legend()
    finish(fig, "learning_curves.png", "Mismo Adam/MSE y regla de parada · punteado: train; continuo: validación · sin evaluación final de train")
