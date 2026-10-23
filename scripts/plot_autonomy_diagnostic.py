"""Static scientific diagnostic figures, exclusively from the new result tables."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save(fig, folder, name, note):
    fig.text(.01, .015, note, fontsize=9, color="#444444")
    fig.tight_layout(rect=(0, .06, 1, .97))
    fig.savefig(folder / name, dpi=160)
    plt.close(fig)


def plot(root):
    folder = root / "plots"
    folder.mkdir(exist_ok=True)
    metrics = pd.read_csv(root / "metrics.csv")
    good = metrics[metrics.complete & (metrics.strategy == "autonomy")]
    reference = good[good.storage_step_kwh == .25]
    cases = sorted(reference.case_id.unique())
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(cases))
    for method, shift, color, marker in (("mlp", -.12, "#3268a8", "o"), ("oracle", .12, "#b85d1c", "D")):
        group = reference[reference.forecast == method].set_index("case_id").reindex(cases)
        ax.scatter(group.delta_grid_import_kwh, y + shift, color=color, marker=marker,
                   label="MLP congelada" if method == "mlp" else "Oracle diagnóstico 24 h", s=45)
    ax.axvline(0, color="black", linewidth=1, label="Reactivo = cota física")
    ax.set(yticks=y, yticklabels=cases, xlabel="Importación adicional frente al reactivo (kWh / semana)",
           title="Grilla de 0,25 kWh: la demanda perfecta reduce parte de la brecha")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.2)
    ax.legend(loc="lower center")
    save(fig, folder, "oracle_vs_mlp.png", "8 escenarios simulados de validación; 2 semanas × 2 perfiles solares × 2 tarifas. Solar futura idealizada.")

    variable = good[good.tariff_mode == "time_of_use"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method, color, marker in (("mlp", "#3268a8", "o"), ("oracle", "#b85d1c", "D")):
        group = variable[variable.forecast == method]
        for _, case in group.groupby("case_id"):
            case = case.sort_values("storage_step_kwh")
            ax.plot(case.storage_step_kwh, case.delta_grid_import_kwh, color=color, alpha=.18, linewidth=1)
        means = group.groupby("storage_step_kwh").delta_grid_import_kwh.mean().sort_index()
        ax.plot(means.index, means.values, color=color, marker=marker, linewidth=2.5,
                label="Media MLP" if method == "mlp" else "Media oracle diagnóstico")
    ax.axhline(0, color="black", linewidth=1)
    ax.set(xticks=[.125, .25, .5], xticklabels=["0,125", "0,25", "0,5"],
           xlabel="Resolución de energía almacenada (kWh)", ylabel="Importación adicional (kWh / semana)",
           title="Sensibilidad de grilla con el objetivo Autonomía intacto")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, folder, "grid_sensitivity.png", "Mismos 4 casos con tarifa variable; líneas tenues = cada caso, líneas gruesas = medias. No son intervalos de confianza.")

    groups = list(variable.groupby(["forecast", "storage_step_kwh"], sort=True))
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(groups))
    positive = np.zeros(len(groups))
    negative = np.zeros(len(groups))
    for column, label, color in (("delta_curtailed_kwh", "Recorte solar adicional", "#3c8c64"),
                                 ("delta_conversion_loss_kwh", "Pérdidas adicionales", "#c88132"),
                                 ("delta_battery_final_kwh", "Diferencia de energía final", "#7b65a4")):
        values = np.array([g[column].mean() for _, g in groups])
        bottom = np.where(values >= 0, positive, negative)
        ax.bar(x, values, bottom=bottom, color=color, label=label)
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    totals = [g.delta_grid_import_kwh.mean() for _, g in groups]
    ax.scatter(x, totals, color="black", marker="D", s=28, label="Brecha total de importación", zorder=4)
    extent = max(positive.max(), max(totals))
    ax.set_ylim(min(0., negative.min()) - .03 * extent, 1.18 * extent)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xticks=x, xticklabels=[f'{method.upper()}\n{step:g} kWh' for (method, step), _ in groups],
           ylabel="Diferencia frente al reactivo (kWh / semana)",
           title="Balance de energía: recorte + pérdidas + energía final = importación adicional")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=.2)
    save(fig, folder, "physical_gap_decomposition.png", "Medias de los mismos 4 casos con tarifa variable. Identidad contable: no separa causas independientes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    plot(parser.parse_args().run_dir)
