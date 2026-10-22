"""CPU experiment figures: equal metrics, raw weighted losses kept separate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"mse": "#2563a6", "weighted_mse": "#b57a15"}
LABELS = {"mse": "MSE", "weighted_mse": "MSE ponderada"}


def weighted_figures(run_dir, model_dir, reference_models):
    run_dir, model_dir, reference_models = map(Path, (run_dir, model_dir, reference_models))
    aggregate = pd.read_csv(run_dir / "metrics_aggregate.csv", dtype={"group": str})
    metrics = pd.read_csv(run_dir / "metrics_by_seed.csv", dtype={"group": str})
    fits = pd.read_csv(run_dir / "fits.csv")
    seeds = sorted(fits.seed.unique())
    thresholds = json.loads((run_dir / "thresholds.json").read_text(encoding="utf-8"))["thresholds_kwh"]
    plots = run_dir / "plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.18, "figure.facecolor": "white"})

    def finish(fig, filename, note=None):
        if note is None:
            note = "Sólo validación · media ± DE muestral entre 3 semillas; no intervalo de confianza ni predictivo"
        fig.text(0.5, .015, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .055, 1, 1))
        fig.savefig(plots / filename, dpi=160)
        plt.close(fig)

    def bars(ax, scope, group, metric, title):
        for x, variant in enumerate(COLORS):
            data = aggregate[(aggregate.scope == scope) & (aggregate.group == group) &
                             (aggregate.metric == metric) & (aggregate.variant == variant)].iloc[0]
            ax.bar(x, data["mean"], yerr=data["std"], color=COLORS[variant], width=.6, capsize=5)
            points = metrics[(metrics.scope == scope) & (metrics.group == group) &
                             (metrics.metric == metric) & (metrics.variant == variant)]
            ax.scatter(np.full(len(points), x), points.value, color="#252525", s=18, zorder=4)
        ax.set_xticks([0, 1], LABELS.values())
        ax.set(title=title, ylabel="% de pares" if metric.endswith("pct") else "kWh por intervalo horario")
        ax.axhline(0, color="#555555", linewidth=.8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
    for ax, metric, title in zip(axes, ("mae_kwh", "rmse_kwh", "bias_kwh"), ("MAE global", "RMSE global", "Sesgo global")):
        bars(ax, "global", "all", metric, title)
    finish(fig, "global_comparison.png")

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for i, group in enumerate(("q90", "q95")):
        for j, metric, title in zip(range(4), ("mae_kwh", "rmse_kwh", "bias_kwh", "underestimate_pct"),
                                    ("MAE", "RMSE", "Sesgo", "Subestimación (%)")):
            bars(axes[i, j], "high_consumption", group, metric, f"> P{group[1:]}: {title}")
    fig.suptitle(f"Umbrales de train: P90={thresholds['q90']:.4f} kWh · P95={thresholds['q95']:.4f} kWh")
    finish(fig, "peak_comparison.png")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.8))
    for ax, metric, title in zip(axes, ("mae_kwh", "rmse_kwh", "bias_kwh", "overestimate_pct"),
                                 ("MAE bajo", "RMSE bajo", "Sesgo bajo", "Sobreestimación (%)")):
        bars(ax, "low_consumption", "q25", metric, title)
    fig.suptitle(f"Consumo bajo: real ≤ P25 de train = {thresholds['q25']:.4f} kWh")
    finish(fig, "low_consumption.png")

    for scope, xlabel, filename in (("horizon", "Horizonte (horas)", "error_by_horizon.png"),
                                    ("hour", "Hora del objetivo (reloj UCI)", "error_by_hour.png")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for ax, metric, title in zip(axes, ("mae_kwh", "rmse_kwh", "bias_kwh"), ("MAE", "RMSE", "Sesgo")):
            for variant, color in COLORS.items():
                data = aggregate[(aggregate.scope == scope) & (aggregate.metric == metric) &
                                 (aggregate.variant == variant)].copy()
                data["x"] = data.group.astype(int)
                data = data.sort_values("x")
                x, y, sd = data.x.to_numpy(), data["mean"].to_numpy(), data["std"].to_numpy()
                ax.plot(x, y, color=color, label=LABELS[variant],
                        linestyle="-" if variant == "mse" else "--", marker="o", markersize=3)
                ax.fill_between(x, y - sd, y + sd, color=color, alpha=.15)
            ax.set(title=title, xlabel=xlabel, ylabel="kWh por intervalo horario")
            ax.set_xticks([1, 4, 8, 12, 16, 20, 24] if scope == "horizon" else [0, 4, 8, 12, 16, 20, 23])
            ax.axhline(0, color="#555555", linewidth=.8)
            ax.legend()
        finish(fig, filename)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, scope, group, title in ((axes[0], "global", "all", "RMSE global"),
            (axes[1], "high_consumption", "q90", "RMSE > P90"), (axes[2], "low_consumption", "q25", "RMSE bajo")):
        data = metrics[(metrics.scope == scope) & (metrics.group == group) & (metrics.metric == "rmse_kwh")]
        for seed in seeds:
            values = data[data.seed == seed].set_index("variant").loc[list(COLORS)].value.to_numpy()
            ax.plot([0, 1], values, color="#777777", linewidth=1)
            ax.scatter([0, 1], values, c=list(COLORS.values()), s=35)
            # Label the more separated weighted endpoints to avoid seed overlap.
            ax.annotate(str(seed), (1, values[1]), xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9)
        ax.set_xticks([0, 1], LABELS.values())
        ax.set(title=title, ylabel="kWh por intervalo horario", xlim=(-.35, 1.65))
    finish(fig, "paired_seeds.png", "Validación · cada línea conecta una semilla · ejes verticales independientes y centrados en las diferencias")

    scale = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))["scale_kwh"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, seed in zip(axes, seeds):
        for variant, color in COLORS.items():
            directory = (reference_models / f"seed_{seed}" / "calendar" if variant == "mse"
                         else model_dir / f"seed_{seed}" / "weighted_mse")
            history = pd.read_csv(directory / "epochs.csv")
            ax.plot(history.epoch + 1, history.rmse * scale, color=color, linestyle=":", alpha=.6)
            ax.plot(history.epoch + 1, history.val_rmse * scale, color=color, label=LABELS[variant])
        ax.set(title=f"Semilla {seed}", xlabel="Época", ylabel="RMSE sin ponderar (kWh)")
        ax.legend()
    finish(fig, "learning_curves.png", "Continuo: validación · punteado: train · misma métrica sin ponderar; las pérdidas de entrenamiento son distintas")
