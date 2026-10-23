"""Reproducible scientific figures from complete paired resilience cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.resilience import POLICIES

LABELS = {"reactive": "Reactivo", "fixed_reserve": "Reserva fija 50%", "adaptive_mlp": "Reserva adaptativa",
    "astar_persistence": "A* persistencia", "astar_mlp": "A* MLP", "astar_oracle": "A* oracle demanda"}
COLORS = ("#67737d", "#8161a8", "#319362", "#c4862d", "#326db3", "#bd4e4e")


def finish(fig, folder, name, bottom=.10):
    fig.text(.01, .012, "Validación simulada · crítico 40% sintético · solar idealizada · cortes imprevistos · no respaldo real certificado", fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, bottom, 1, .96))
    fig.savefig(folder / name, dpi=160)
    plt.close(fig)


def plot(root):
    root = Path(root)
    folder = root / "plots"
    folder.mkdir(exist_ok=True)
    paired = pd.read_csv(root / "paired_metrics.csv")
    cuts = paired[paired.duration_hours > 0]
    if cuts.empty:
        return
    durations = sorted(cuts.duration_hours.unique())
    fig, axes = plt.subplots(1, len(durations), figsize=(15, 5.5), sharey=True, squeeze=False)
    for ax, duration in zip(axes[0], durations):
        subset = cuts[cuts.duration_hours == duration]
        for i, (policy, color) in enumerate(zip(POLICIES, COLORS)):
            values = subset[subset.policy == policy].critical_ens_kwh.to_numpy()
            ax.bar(i, values.mean(), color=color, alpha=.75)
            ax.scatter(i + np.linspace(-.16, .16, len(values)), values, color=color, s=9, alpha=.55, edgecolors="black", linewidths=.2)
        ax.set(title=f"Corte de {int(duration)} h", xticks=range(6), xticklabels=range(1, 7), xlabel="Estrategia (leyenda)")
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].set_ylim(0, 1.08 * max(float(cuts.critical_ens_kwh.max()), 1e-6))
    axes[0, 0].set_ylabel("ENS crítica por evento (kWh)")
    fig.suptitle("Déficit crítico: barras = medias; puntos = cada escenario pareado")
    handles = [plt.Line2D([], [], marker="s", linestyle="", color=c, label=f"{i+1}. {LABELS[p]}") for i, (p, c) in enumerate(zip(POLICIES, COLORS))]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=3, fontsize=9)
    finish(fig, folder, "critical_ens.png", .18)

    table = np.array([[100 * cuts[(cuts.policy == p) & (cuts.duration_hours == d)].no_critical_deficit.astype(bool).mean()
                       for d in durations] for p in POLICIES])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    handle = ax.imshow(table, vmin=0, vmax=100, cmap="Blues", aspect="auto")
    for y in range(6):
        for x, duration in enumerate(durations):
            group = cuts[(cuts.policy == POLICIES[y]) & (cuts.duration_hours == duration)]
            completed = int(group.no_critical_deficit.astype(bool).sum())
            ax.text(x, y, f"{table[y,x]:.1f}%\n({completed}/{len(group)})", ha="center", va="center",
                    color="white" if table[y,x] > 55 else "black", fontsize=10)
    ax.set(xticks=range(len(durations)), xticklabels=[f"{int(d)} horas" for d in durations],
           yticks=range(6), yticklabels=[LABELS[p] for p in POLICIES], title="Escenarios con servicio crítico completo durante todo el corte")
    fig.colorbar(handle, ax=ax, label="Casos completos (%)")
    finish(fig, folder, "critical_service_coverage.png", .08)

    tariffs = sorted(cuts.tariff_mode.unique())
    fig, axes = plt.subplots(1, len(tariffs), figsize=(14, 6), squeeze=False)
    for ax, tariff in zip(axes[0], tariffs):
        subset = cuts[cuts.tariff_mode == tariff]
        for i, (policy, color) in enumerate(zip(POLICIES, COLORS)):
            group = subset[subset.policy == policy]
            x, y = group.adjusted_cost_sim.mean(), group.critical_ens_kwh.mean()
            ax.scatter(x, y, color=color, s=70, marker="D" if policy == "astar_oracle" else "o")
            ax.annotate(str(i + 1), (x, y), xytext=(6, 7 + 4 * (i % 2)), textcoords="offset points", fontsize=10)
        ax.set(title="Tarifa plana" if tariff == "flat" else "Tarifa variable", xlabel="Costo ajustado semanal (SIM, no factura)", ylabel="ENS crítica media por evento (kWh)")
        ax.margins(.18)
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    fig.suptitle("Compromiso entre servicio crítico y costo: seis políticas observadas")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=3, fontsize=9)
    finish(fig, folder, "service_cost_tradeoff.png", .18)

    example_candidates = cuts[(cuts.tariff_mode == "time_of_use") & (cuts.event_id == "hour22_duration06")]
    if len(example_candidates):
        base_id = sorted(example_candidates.case_id.unique())[0]
        example = example_candidates[example_candidates.case_id == base_id]
        first = pd.read_csv(Path(example.iloc[0].output_dir) / "ledger.csv")
        start = int(example.iloc[0].onset_step)
        end = start + 6
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        select = first.step.between(start - 12, end + 12)
        axes[0].plot(first.step[select], first.original_demand_kwh[select], color="#777777", label="Demanda total")
        axes[0].plot(first.step[select], first.critical_required_kwh[select], color="#bd4e4e", label="Demanda crítica sintética")
        axes[0].plot(first.step[select], first.solar_kwh[select], color="#c4862d", label="Solar independiente")
        axes[0].legend(loc="upper left", ncol=3, fontsize=9)
        axes[0].set_ylabel("Energía horaria (kWh)")
        for policy, color in zip(POLICIES, COLORS):
            ledger = pd.read_csv(Path(example[example.policy == policy].iloc[0].output_dir) / "ledger.csv")
            selected = ledger[ledger.step.between(start - 12, end + 12)]
            axes[1].plot(selected.step + 1, selected.soc * 100, color=color, label=LABELS[policy], linestyle="--" if policy == "astar_oracle" else "-")
            axes[2].plot(selected.step, selected.critical_ens_kwh, color=color, linestyle="--" if policy == "astar_oracle" else "-", marker="." )
        axes[1].axhline(20, color="black", linewidth=.8, linestyle=":")
        axes[1].axhline(50, color="#999999", linewidth=.8, linestyle=":")
        axes[1].set(ylabel="SOC (%)", ylim=(15, 105))
        axes[2].set(ylabel="ENS crítica (kWh)", xlabel="Hora transcurrida del escenario semanal")
        for ax in axes:
            ax.axvspan(start, end, color="#b24c4c", alpha=.10)
            ax.grid(alpha=.15)
        fig.suptitle(f"Ejemplo cronológico fijo: {base_id}, corte 22:00 de 6 h")
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .04), ncol=3, fontsize=9)
        finish(fig, folder, "soc_and_critical_service.png", .16)

    differences = pd.read_csv(root / "paired_differences.csv")
    labels = (("A_reserve", "Fija vs.\nsin reserva"), ("B_adaptive", "Adaptativa vs.\nfija"),
              ("C_search", "A* persistencia\nvs. adaptativa"), ("D_neural", "A* MLP vs.\nA* persistencia"), ("E_primary", "A* MLP vs.\nfija"))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for x, (comparison, label) in enumerate(labels):
        group = differences[(differences.comparison == comparison) & (differences.duration_hours > 0)]
        values = group.avoided_critical_kwh.to_numpy()
        ax.scatter(x + np.linspace(-.18, .18, len(values)), values, color="#637f9c", alpha=.4, s=15)
        ax.scatter(x, values.mean(), color="black", marker="D", s=50)
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xticks=range(5), xticklabels=[label for _, label in labels], ylabel="ENS crítica evitada por evento (kWh)",
           title="Beneficio incremental: positivo mejora; negativo perjudica")
    ax.grid(axis="y", alpha=.2)
    finish(fig, folder, "incremental_benefit.png", .10)
    write_json(root / "plot_provenance.json", {"plot_source_sha256": sha256(Path(__file__)),
        "figures": {str(p): sha256(p) for p in folder.glob("*.png")},
        "sample_units": "paired factorial scenarios, not independent household/annual observations"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    plot(parser.parse_args().run_dir)
