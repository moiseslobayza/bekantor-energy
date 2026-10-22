"""Phase 2.1: read saved validation predictions only; never fit or infer a model.

Run from the repository root: .venv/Scripts/python.exe scripts/diagnose_validation.py
Outputs are descriptive diagnostics, not independent significance tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments/phase2/reference-20261022-cpu"
OUT = ROOT / "experiments/phase2_1/diagnostic"
COLORS = {"MLP": "#246b9e", "Diaria 24h": "#b78022", "Semanal 168h": "#727e45"}
MODELS = {"MLP": "selected_mlp_kwh", "Diaria 24h": "seasonal_24h_kwh", "Semanal 168h": "seasonal_168h_kwh"}
DAYS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def summarize(group):
    e = group.error.to_numpy()
    return {"n_pairs": len(group), "n_target_hours": group.target_timestamp.nunique(),
            "actual_mean_kwh": group.actual_kwh.mean(), "predicted_mean_kwh": group.prediction.mean(),
            "mae_kwh": np.abs(e).mean(), "rmse_kwh": np.sqrt(np.mean(e**2)),
            "bias_kwh": e.mean(), "under_fraction": np.mean(e < 0),
            "over_fraction": np.mean(e > 0), "sse_kwh2": np.sum(e**2),
            "under_gt1_fraction": np.mean(e < -1),
            "mean_under_kwh": -e[e < 0].mean() if (e < 0).any() else None,
            "mean_over_kwh": e[e > 0].mean() if (e > 0).any() else None}


def aggregate(long, dimensions):
    rows = []
    for key, group in long.groupby(["model", *dimensions], observed=True, sort=True):
        rows.append({**dict(zip(["model", *dimensions], key)), **summarize(group)})
    return pd.DataFrame(rows)


def figsave(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Explicit source allowlist: no glob over experiment/test artifacts and no TensorFlow import.
    sources = {"predictions": RUN / "validation_predictions.csv", "selection": RUN / "selection.json",
               "train": ROOT / "data/processed/uci/train.csv",
               "validation": ROOT / "data/processed/uci/validation.csv",
               "config": RUN / "config.json", "phase1_manifest": ROOT / "data/processed/uci/manifest.json"}
    for name in ("small", "medium", "large"):
        sources[f"history_{name}"] = ROOT / f"models/phase2/reference-20261022-cpu/{name}/training.json"
    before = {key: sha(path) for key, path in sources.items()}
    df = pd.read_csv(sources["predictions"], parse_dates=["origin", "target_timestamp"])
    # Phase 2 serialized float32 truth/baselines as short decimals. Restore their
    # original precision before comparing to the recorded float64 metric results.
    for name in ("actual_kwh", "seasonal_24h_kwh", "seasonal_168h_kwh"):
        df[name] = df[name].astype(np.float32).astype(np.float64)
    train = pd.read_csv(sources["train"], parse_dates=["timestamp"])
    validation = pd.read_csv(sources["validation"], parse_dates=["timestamp"])
    selection = json.loads(sources["selection"].read_text(encoding="utf-8"))
    assert not df.duplicated(["origin", "horizon_hour"]).any()
    assert df.groupby("origin").horizon_hour.agg(list).map(lambda v: v == list(range(1, 25))).all()
    assert (df.target_timestamp - df.origin == pd.to_timedelta(df.horizon_hour - 1, unit="h")).all()
    assert df.target_timestamp.between(validation.timestamp.min(), validation.timestamp.max()).all()
    assert np.isfinite(df[["actual_kwh", *MODELS.values()]].to_numpy()).all()
    actual_unique = df.groupby("target_timestamp").actual_kwh.agg(["min", "max"])
    assert np.allclose(actual_unique["min"], actual_unique["max"])
    truth = validation.set_index("timestamp").loc[df.target_timestamp]
    assert truth.is_observed_complete.all()
    np.testing.assert_allclose(df.actual_kwh, truth.load_kwh, atol=5e-7, rtol=1e-7)
    train_values = train.loc[train.is_observed_complete, "load_kwh"]
    q50, q90, q95 = train_values.quantile([0.5, 0.9, 0.95])
    df["hour"] = df.target_timestamp.dt.hour
    df["weekday"] = df.target_timestamp.dt.dayofweek
    df["month"] = df.target_timestamp.dt.strftime("%Y-%m")
    df["load_band"] = pd.cut(df.actual_kwh, [-np.inf, q50, q90, q95, np.inf],
                              labels=["Bajo ≤P50", "Medio P50–P90", "Alto P90–P95", "Muy alto >P95"])
    long = pd.concat([df.assign(model=name, prediction=df[column]) for name, column in MODELS.items()], ignore_index=True)
    long["error"] = long.prediction - long.actual_kwh
    long["abs_error"] = long.error.abs()
    long["squared_error"] = long.error**2
    tables = {"overall": aggregate(long, []), "horizon": aggregate(long, ["horizon_hour"]),
              "hour": aggregate(long, ["hour"]), "weekday": aggregate(long, ["weekday"]),
              "load_band": aggregate(long, ["load_band"]), "month": aggregate(long, ["month"])}
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False)
    for name, column in MODELS.items():
        scores = selection["validation_comparison"][column.removesuffix("_kwh")]
        got = tables["overall"].set_index("model").loc[name]
        assert abs(got.rmse_kwh - scores["rmse_kwh"]) < 1e-9
        assert abs(got.mae_kwh - scores["mae_kwh"]) < 1e-9
    mlp = long.loc[long.model == "MLP"].copy()
    high = mlp.loc[mlp.actual_kwh > q90]
    high_rows = aggregate(long.loc[long.actual_kwh > q90], [])
    high_rows.to_csv(OUT / "high_load.csv", index=False)
    # Reweight targets equally without averaging predictions before errors are calculated.
    target_errors = mlp.groupby("target_timestamp").agg(
        mean_abs=("abs_error", "mean"), mean_sq=("squared_error", "mean"), mean_error=("error", "mean"))
    sensitivity = {"mae_kwh": target_errors.mean_abs.mean(), "rmse_kwh": np.sqrt(target_errors.mean_sq.mean()),
                   "bias_kwh": target_errors.mean_error.mean()}
    # Calibration by forecast level can be evaluated at issuance; differs from conditioning on realized peaks.
    mlp["prediction_decile"] = pd.qcut(mlp.prediction, 10, labels=False, duplicates="drop")
    calibration = aggregate(mlp, ["prediction_decile"])
    calibration.to_csv(OUT / "forecast_level_calibration.csv", index=False)
    # Unique observed hours describe distributions, not additional evaluation of models on train.
    distributions = []
    for split, frame in (("train", train), ("validation", validation)):
        values = frame.loc[frame.is_observed_complete, "load_kwh"]
        distributions.append({"split": split, "hours": len(values), "mean_kwh": values.mean(),
                              "std_kwh": values.std(ddof=0), "median_kwh": values.median(),
                              "p90_kwh": values.quantile(.9), "p95_kwh": values.quantile(.95),
                              "high_fraction": (values > q90).mean()})
    distributions = pd.DataFrame(distributions)
    distributions.to_csv(OUT / "distributions.csv", index=False)
    train_observed = train.loc[train.is_observed_complete].assign(
        calendar_month=lambda x: x.timestamp.dt.month, hour=lambda x: x.timestamp.dt.hour)
    validation_observed = validation.loc[validation.is_observed_complete].assign(
        calendar_month=lambda x: x.timestamp.dt.month, hour=lambda x: x.timestamp.dt.hour)
    cell_means = train_observed.groupby(["calendar_month", "hour"]).load_kwh.mean()
    weights = validation_observed.groupby(["calendar_month", "hour"]).size() / len(validation_observed)
    assert cell_means.reindex(weights.index).notna().all()
    calendar_matched_mean = float((cell_means.reindex(weights.index) * weights).sum())
    histories = {name: json.loads(sources[f"history_{name}"].read_text(encoding="utf-8")) for name in ("small", "medium", "large")}
    near = mlp.loc[mlp.horizon_hour <= 6]
    far = mlp.loc[mlp.horizon_hour >= 19]
    mlp["weekend"] = mlp.weekday >= 5
    cells = mlp.groupby(["weekend", "hour", "load_band"], observed=True).agg(
        mse=("squared_error", "mean"), n=("error", "size"))
    work, weekend = cells.loc[False], cells.loc[True]
    common = work.index.intersection(weekend.index)
    covered_work = float(work.loc[common, "n"].sum() / work.n.sum())
    covered_end = float(weekend.loc[common, "n"].sum() / weekend.n.sum())
    work, weekend = work.loc[common], weekend.loc[common]
    w_work, w_end = work.n / work.n.sum(), weekend.n / weekend.n.sum()
    mix = float(((w_end - w_work) * work.mse).sum())
    within = float((w_end * (weekend.mse - work.mse)).sum())
    delta = float((w_end * weekend.mse).sum() - (w_work * work.mse).sum())
    assert abs(mix + within - delta) < 1e-10
    summary = {"scope": "validation only; saved predictions; no fitting or inference",
               "origins": df.origin.nunique(), "pairs": len(df), "unique_target_hours": df.target_timestamp.nunique(),
               "start": df.target_timestamp.min().isoformat(), "end": df.target_timestamp.max().isoformat(),
               "thresholds_train": {"p50_kwh": q50, "p90_kwh": q90, "p95_kwh": q95},
               "high_load": {**summarize(high), "pair_share": len(high) / len(mlp),
                              "sse_share": (high.error**2).sum() / (mlp.error**2).sum()},
               "near_rmse_kwh": np.sqrt(np.mean(near.error**2)), "far_rmse_kwh": np.sqrt(np.mean(far.error**2)),
               "equal_target_weight_sensitivity": sensitivity,
               "calendar_matched_train_mean_kwh": calendar_matched_mean,
               "weekend_decomposition": {"mse_difference_kwh2": delta,
                                           "mix_hour_actual_band_kwh2": mix,
                                           "within_cells_kwh2": within,
                                           "weekday_pair_coverage": covered_work,
                                           "weekend_pair_coverage": covered_end,
                                           "note": "Common hour/actual-band cells only; renormalized weights; weekday cell MSE reference; not causal"},
               "calendar_mix_note": "train month/hour means weighted with validation observed month/hour mix; descriptive, not causal",
               "global": tables["overall"].to_dict("records")}
    save_json(OUT / "summary.json", summary)
    # Five compact export figures accompany the report and inspectable CSVs.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for model, color in COLORS.items():
        rows = tables["horizon"].query("model == @model")
        axes[0].plot(rows.horizon_hour, rows.rmse_kwh, label=model, color=color)
        axes[1].plot(rows.horizon_hour, rows.bias_kwh, label=model, color=color)
    axes[0].set(ylabel="RMSE (kWh)", xlabel="Horizonte (h)", title="Validación: error por horizonte", xticks=[1, 6, 12, 18, 24])
    axes[1].set(ylabel="Sesgo: predicho − real (kWh)", xlabel="Horizonte (h)", title="Sesgo por horizonte", xticks=[1, 6, 12, 18, 24])
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[0].legend()
    figsave(fig, "horizon.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for model, color in COLORS.items():
        rows = tables["hour"].query("model == @model")
        axes[0].plot(rows.hour, rows.rmse_kwh, label=model, color=color)
        axes[1].plot(rows.hour, rows.bias_kwh, label=model, color=color)
    axes[0].set(xlabel="Hora objetivo, reloj UCI", ylabel="RMSE (kWh)", title="Error según hora del día", xticks=range(0, 24, 3))
    axes[1].set(xlabel="Hora objetivo, reloj UCI", ylabel="Sesgo (kWh)", title="Sesgo según hora del día", xticks=range(0, 24, 3))
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[0].legend()
    figsave(fig, "hour_of_day.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for model, color in COLORS.items():
        rows = tables["weekday"].query("model == @model")
        axes[0].plot(rows.weekday, rows.rmse_kwh, label=model, color=color, marker="o", markersize=3)
        axes[1].plot(rows.weekday, rows.bias_kwh, label=model, color=color, marker="o", markersize=3)
    for ax in axes:
        ax.set(xticks=range(7), xticklabels=[d[:3] for d in DAYS], xlabel="Día objetivo")
    axes[0].set(ylabel="RMSE (kWh)", title="Error según día de la semana")
    axes[1].set(ylabel="Sesgo (kWh)", title="Sesgo según día de la semana")
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[0].legend()
    figsave(fig, "weekday.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    bands = ["Bajo ≤P50", "Medio P50–P90", "Alto P90–P95", "Muy alto >P95"]
    for i, (model, color) in enumerate(COLORS.items()):
        rows = tables["load_band"].query("model == @model").set_index("load_band").reindex(bands)
        axes[0].bar(np.arange(4) + (i - 1) * .25, rows.rmse_kwh, width=.25, label=model, color=color)
        axes[1].bar(np.arange(4) + (i - 1) * .25, rows.bias_kwh, width=.25, label=model, color=color)
    for ax in axes:
        ax.set(xticks=range(4), xticklabels=["≤P50", "P50–P90", "P90–P95", ">P95"], xlabel="Consumo REAL; umbrales calculados en train")
    axes[0].set(ylabel="RMSE (kWh)", title="Error por nivel de consumo real")
    axes[1].set(ylabel="Sesgo (kWh)", title="Los picos observados se subestiman")
    axes[1].axhline(0, color="#777777", linewidth=.8)
    axes[0].legend()
    figsave(fig, "high_load.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(calibration.predicted_mean_kwh, calibration.actual_mean_kwh, color=COLORS["MLP"], marker="o")
    limits = [0, max(calibration.predicted_mean_kwh.max(), calibration.actual_mean_kwh.max()) * 1.05]
    axes[0].plot(limits, limits, linestyle="--", color="#777777")
    axes[0].set(xlabel="Predicción media por decil (kWh)", ylabel="Consumo real medio (kWh)", title="Calibración por nivel pronosticado")
    rows = tables["month"].query("model == 'MLP'")
    axes[1].plot(rows.month, rows.rmse_kwh, color=COLORS["MLP"], marker="o", label="RMSE MLP")
    axes[1].set(ylabel="RMSE (kWh)", title="Variación entre meses de validación")
    axes[1].tick_params(axis="x", rotation=30)
    figsave(fig, "calibration_month.png")
    provenance = {"script": str(Path(__file__).relative_to(ROOT)), "script_sha256": sha(Path(__file__)),
                  "sources": {key: {"path": str(path.relative_to(ROOT)), "sha256": digest} for key, path in sources.items() for digest in [before[key]]},
                  "checks": ["4478 complete 24-hour origins", "unique origin/horizon pairs", "correct timestamp offsets", "labels match observed validation hours", "metrics reproduce saved validation to 1e-9", "train/validation source files unchanged"],
                  "limitations": ["Overlapping windows; no IID significance claims", "Target hour/day use unspecified UCI wall-clock", "High actual consumption is an ex-post diagnostic, not known at forecast time", "No test data or test metrics opened", "No TensorFlow import, fit, predict or optimizer execution"]}
    assert before == {key: sha(path) for key, path in sources.items()}
    save_json(OUT / "provenance.json", provenance)
    # Bounded reviewed data for the local report app; no raw household series embedded.
    queries = {}
    for name, table in tables.items():
        records = table.astype(object).where(pd.notna(table), None).to_dict("records")
        queries[name] = {"rows": records, "source": {
            "label": "BEKANTOR: predicciones guardadas de validación", "files": [str(sources["predictions"].relative_to(ROOT))],
            "filters": [f"Objetivos {summary['start']} a {summary['end']}", "Sólo etiquetas observadas; mismos pares origen/horizonte para tres estrategias"],
            "metricDefinitions": [{"label": "Errores de validación", "definition": "Error = predicho-real. MAE=media(abs(error)); RMSE=sqrt(media(error²)); sesgo=media(error), todos en kWh. n_pairs cuenta pares solapados, no horas independientes."}],
            "evidenceFlow": [{"title": "Fuente local inmutable", "detail": f"validation_predictions.csv SHA-256 {before['predictions']}"},
                             {"title": "Cálculo reproducible", "detail": f"scripts/diagnose_validation.py; agrupación {name}; sin reentrenamiento ni test."}]},
                         "methods": [{"language": "Python", "code": f"python scripts/diagnose_validation.py; salida experiments/phase2_1/diagnostic/{name}.csv"}]}
    for name, table in (("high_load", high_rows), ("calibration", calibration), ("distributions", distributions)):
        queries[name] = {"rows": table.astype(object).where(pd.notna(table), None).to_dict("records"),
                         "source": {"label": "Diagnóstico descriptivo de validación / contexto train",
                                    "files": [str(sources[k].relative_to(ROOT)) for k in ("predictions", "train", "validation")],
                                    "filters": ["Sólo validación para errores; train únicamente para umbrales y contexto de distribución"],
                                    "evidenceFlow": [{"title": "Reproducción", "detail": f"python scripts/diagnose_validation.py; {name}.csv"}]}}
    queries["load_band"]["source"]["files"].append(str(sources["train"].relative_to(ROOT)))
    queries["load_band"]["source"]["metricDefinitions"].append({"label": "Consumo elevado", "definition": f"Consumo REAL > P90 de las horas observadas de train ({q90:.9f} kWh); categoría descriptiva ex post, no variable conocida al emitir el pronóstico."})
    history_rows = [{"model": name, "parameters": history["parameters"], "best_epoch": history["best_epoch"],
                     "epochs_run": history["epochs_run"], "best_validation_rmse_kwh": history["validation"]["rmse_kwh"],
                     "last_validation_rmse_scaled": history["history"]["val_rmse"][-1],
                     "best_validation_rmse_scaled": min(history["history"]["val_rmse"])} for name, history in histories.items()]
    queries["training_history"] = {"rows": history_rows, "source": {
        "label": "Historias de entrenamiento ya guardadas; sin nuevos ajustes",
        "files": [str(sources[f"history_{name}"].relative_to(ROOT)) for name in histories],
        "filters": ["Comparación predictiva únicamente por métricas de validación"],
        "evidenceFlow": [{"title": "Lectura local", "detail": "JSON training.json de small, medium y large; sin cargar pesos ni ejecutar TensorFlow."}]}}
    queries["diagnostic_context"] = {"rows": [{
        "origins": summary["origins"], "pairs": summary["pairs"], "unique_target_hours": summary["unique_target_hours"],
        "high_pair_share": summary["high_load"]["pair_share"], "high_sse_share": summary["high_load"]["sse_share"],
        "calendar_matched_train_mean_kwh": calendar_matched_mean,
        "equal_target_rmse_kwh": sensitivity["rmse_kwh"],
        "weekend_mse_gap_common_kwh2": delta, "weekend_mix_contribution_kwh2": mix,
        "weekend_within_contribution_kwh2": within, "weekday_pair_coverage": covered_work,
        "weekend_pair_coverage": covered_end}], "source": {
            "label": "Controles y descomposición descriptiva", "files": [str(sources[k].relative_to(ROOT)) for k in ("predictions", "train", "validation")],
            "metricDefinitions": [{"label": "Mezcla fin de semana", "definition": "Sobre celdas comunes hora×banda de consumo real, pesos normalizados dentro de laborable/fin de semana: ΔMSE=sum((w_fin-w_lab)*MSE_lab)+sum(w_fin*(MSE_fin-MSE_lab)). Descomposición contable, no causal."}],
            "evidenceFlow": [{"title": "Código y comprobaciones", "detail": "scripts/diagnose_validation.py; summary.json y provenance.json; fuentes inalteradas; sin test."}]}}
    snapshot = {"title": "BEKANTOR Energy — Diagnóstico de validación", "generatedAt": datetime.now(timezone.utc).isoformat(),
                "queries": queries}
    save_json(OUT / "reviewed.json", snapshot)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
