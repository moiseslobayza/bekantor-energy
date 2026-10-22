"""Run `python -m bekantor_energy --help` after editable installation."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from urllib.error import URLError

from .config import HomeConfig
from .data.acquisition import acquire_uci
from .data.pvgis import PVGISQuery, acquire_pvgis, parse_pvgis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BEKANTOR Energy — datos y predicción local en CPU")
    commands = parser.add_subparsers(dest="command", required=True)
    uci = commands.add_parser("download-uci", help="Descargar UCI y registrar SHA-256/procedencia")
    uci.add_argument("--directory", type=Path, default=Path("data/raw/uci"))
    uci.add_argument("--force", action="store_true")
    pv = commands.add_parser("download-pvgis", help="Obtener generación modelada independiente, PVGIS v6")
    pv.add_argument("--home-config", type=Path, default=Path("configs/home.example.json"))
    pv.add_argument("--latitude", type=float, default=-34.6037)
    pv.add_argument("--longitude", type=float, default=-58.3816)
    pv.add_argument("--start", default="2023-01-01T00:00:00")
    pv.add_argument("--end", default="2023-01-02T23:00:00")
    pv.add_argument("--tilt", type=float, default=30.0)
    pv.add_argument("--orientation", type=float, default=0.0)
    pv.add_argument("--irradiance-source", choices=["ERA5", "SARAH-3"], default="ERA5")
    pv.add_argument("--output", type=Path, default=Path("data/raw/pvgis/generation.json"))
    pv.add_argument("--processed-output", type=Path, default=Path("data/processed/pvgis/hourly.csv"))
    pv.add_argument("--force", action="store_true")
    process = commands.add_parser("process", help="Validar, procesar, dividir cronológicamente y ejecutar EDA")
    process.add_argument("--input", type=Path, required=True)
    process.add_argument("--source-kind", choices=["uci_measured", "synthetic_fixture"], required=True)
    process.add_argument("--output", type=Path, default=Path("data/processed/uci"))
    process.add_argument("--reports", type=Path, default=Path("experiments/uci"))
    process.add_argument("--max-ffill-minutes", type=int, default=5)
    process.add_argument("--train-fraction", type=float, default=0.7)
    process.add_argument("--validation-fraction", type=float, default=0.15)
    process.add_argument("--eda-scope", choices=["train", "all"], default="train")
    eda = commands.add_parser("eda", help="Repetir EDA de una partición CSV procesada")
    eda.add_argument("--input", type=Path, required=True)
    eda.add_argument("--output", type=Path, default=Path("experiments/eda"))
    eda.add_argument("--source-kind", choices=["uci_measured", "synthetic_fixture"], required=True)
    eda.add_argument("--scope", choices=["train", "validation", "test", "all"], default="train")
    config = commands.add_parser("check-config", help="Validar parámetros de la vivienda")
    config.add_argument("--input", type=Path, default=Path("configs/home.example.json"))
    training = commands.add_parser("train-mlp", help="Comparar MLP sólo con train/validation; no abre test")
    training.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    training.add_argument("--config", type=Path, default=Path("configs/mlp.reference.json"))
    training.add_argument("--run-dir", type=Path, required=True)
    training.add_argument("--model-dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate-mlp", help="Evaluar una sola vez el modelo seleccionado y baselines en test")
    evaluate.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    evaluate.add_argument("--run-dir", type=Path, required=True)
    plots = commands.add_parser("plot-mlp", help="Regenerar figuras desde artefactos guardados, sin reevaluar test")
    plots.add_argument("--run-dir", type=Path, required=True)
    calendar = commands.add_parser("compare-calendar", help="Comparación pareada 64-32 con calendario; sólo train/validation")
    calendar.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    calendar.add_argument("--config", type=Path, default=Path("configs/mlp.calendar-comparison.json"))
    calendar.add_argument("--run-dir", type=Path, required=True)
    calendar.add_argument("--model-dir", type=Path, required=True)
    calendar.add_argument("--reference-run", type=Path, default=Path("experiments/phase2/reference-20261022-cpu"))
    calendar.add_argument("--reference-models", type=Path, default=Path("models/phase2/reference-20261022-cpu"))
    weighted = commands.add_parser("compare-weighted-mse", help="MSE ponderada 1/2 contra control calendario congelado; sin test")
    weighted.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    weighted.add_argument("--config", type=Path, default=Path("configs/mlp.weighted-mse.json"))
    weighted.add_argument("--run-dir", type=Path, required=True)
    weighted.add_argument("--model-dir", type=Path, required=True)
    weighted.add_argument("--reference-run", type=Path, default=Path("experiments/phase2_2/calendar-20261022"))
    weighted.add_argument("--reference-models", type=Path, default=Path("models/phase2_2/calendar-20261022"))
    energy = commands.add_parser("energy-demo", help="A* + simulación física sobre validación; no reentrena ni abre test")
    energy.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    energy.add_argument("--model-bundle", type=Path, default=Path("models/phase2_2/calendar-20261022/seed_2026/calendar/bundle"))
    energy.add_argument("--solar-raw", type=Path, default=Path("data/raw/pvgis/cordoba-20230921.json"))
    energy.add_argument("--home-config", type=Path, default=Path("configs/home.phase3.json"))
    energy.add_argument("--planner-config", type=Path, default=Path("configs/planner.phase3.json"))
    energy.add_argument("--previous-verification", type=Path, default=Path("experiments/phase2_3/weighted-mse-20261022/verification.json"))
    energy.add_argument("--run-dir", type=Path, required=True)
    energy.add_argument("--scenario-dir", type=Path, required=True)
    compare = commands.add_parser("compare-energy", help="Cuatro estrategias pareadas sobre validación; redes congeladas y oracle explícito")
    compare.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    compare.add_argument("--mse-bundle", type=Path, default=Path("models/phase2_2/calendar-20261022/seed_2026/calendar/bundle"))
    compare.add_argument("--weighted-bundle", type=Path, default=Path("models/phase2_3/weighted-mse-20261022/seed_2026/weighted_mse/bundle"))
    compare.add_argument("--solar-raw", type=Path, default=Path("data/raw/pvgis/cordoba-20230921.json"))
    compare.add_argument("--home-config", type=Path, default=Path("configs/home.phase3.json"))
    compare.add_argument("--planner-config", type=Path, default=Path("configs/planner.phase3.json"))
    compare.add_argument("--config", type=Path, default=Path("configs/energy.comparison.json"))
    compare.add_argument("--previous-verification", type=Path, default=Path("experiments/phase3/validation-20261022/verification.json"))
    compare.add_argument("--run-dir", type=Path, required=True)
    compare.add_argument("--scenario-dir", type=Path, required=True)
    continuous = commands.add_parser("continuous-energy", help="Operación continua de validación, SOC persistente y tratamiento terminal explícito")
    continuous.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    continuous.add_argument("--model-bundle", type=Path, default=Path("models/phase2_2/calendar-20261022/seed_2026/calendar/bundle"))
    continuous.add_argument("--home-config", type=Path, default=Path("configs/home.phase3.json"))
    continuous.add_argument("--planner-config", type=Path, default=Path("configs/planner.phase3.json"))
    continuous.add_argument("--config", type=Path, default=Path("configs/energy.continuous.json"))
    continuous.add_argument("--previous-verification", type=Path, default=Path("experiments/phase4_1/paired-20261022/verification.json"))
    continuous.add_argument("--run-dir", type=Path, required=True)
    continuous.add_argument("--scenario-dir", type=Path, required=True)
    modes = commands.add_parser("energy-modes", help="Comparar Económico, Autonomía y Equilibrado en validación continua")
    modes.add_argument("--config", type=Path, default=Path("configs/energy.modes.json"))
    modes.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    modes.add_argument("--run-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download-uci":
            print(acquire_uci(args.directory, force=args.force))
        elif args.command == "download-pvgis":
            home = HomeConfig.load(args.home_config)
            query = PVGISQuery(latitude=args.latitude, longitude=args.longitude, start=args.start,
                               end=args.end, peak_kwp=home.pv_peak_kwp,
                               system_efficiency=home.pv_system_efficiency, tilt_deg=args.tilt,
                               orientation_deg=args.orientation, irradiance_source=args.irradiance_source)
            path = acquire_pvgis(query, args.output, force=args.force)
            frame = parse_pvgis(json.loads(path.read_text(encoding="utf-8")), query)
            args.processed_output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.processed_output, index_label="timestamp")
            print(f"PVGIS v6: {len(frame)} horas UTC -> {args.processed_output}")
        elif args.command == "process":
            from .pipeline import process_uci
            result = process_uci(args.input, args.output, args.reports,
                                 max_ffill_minutes=args.max_ffill_minutes,
                                 train_fraction=args.train_fraction, validation_fraction=args.validation_fraction,
                                 source_kind=args.source_kind, eda_scope=args.eda_scope)
            print(json.dumps({"quality": result["quality"], "partitions": result["partitions"]}, indent=2))
        elif args.command == "eda":
            import pandas as pd
            from .eda import run_eda
            frame = pd.read_csv(args.input, index_col="timestamp", parse_dates=["timestamp"])
            print(json.dumps(run_eda(frame, args.output, scope=args.scope, source_kind=args.source_kind), indent=2))
        elif args.command == "check-config":
            print(json.dumps(asdict(HomeConfig.load(args.input)), indent=2))
        elif args.command == "train-mlp":
            from .forecasting.config import ExperimentConfig
            from .forecasting.experiment import train_experiment
            train_experiment(args.data_dir, ExperimentConfig.load(args.config), args.run_dir, args.model_dir)
        elif args.command == "evaluate-mlp":
            from .forecasting.experiment import evaluate_test_once
            result = evaluate_test_once(args.data_dir, args.run_dir)
            print(json.dumps(result, indent=2))
        elif args.command == "plot-mlp":
            from .forecasting.experiment import replot_experiment
            replot_experiment(args.run_dir)
        elif args.command == "compare-calendar":
            from .forecasting.calendar_experiment import CalendarComparisonConfig, run_calendar_comparison
            run_calendar_comparison(args.data_dir, CalendarComparisonConfig.load(args.config),
                                    args.run_dir, args.model_dir, reference_run=args.reference_run,
                                    reference_models=args.reference_models)
        elif args.command == "compare-weighted-mse":
            from .forecasting.weighted_experiment import WeightedComparisonConfig, run_weighted_comparison
            run_weighted_comparison(args.data_dir, WeightedComparisonConfig.load(args.config),
                                    args.run_dir, args.model_dir, args.reference_run, args.reference_models)
        elif args.command == "energy-demo":
            from .energy.experiment import run_validation_demo
            run_validation_demo(args.data_dir, args.model_bundle, args.solar_raw, args.home_config,
                                args.planner_config, args.run_dir, args.scenario_dir,
                                previous_verification=args.previous_verification)
        elif args.command == "compare-energy":
            from .energy.comparison import ComparisonConfig, run_comparison
            run_comparison(args.data_dir, args.mse_bundle, args.weighted_bundle, args.solar_raw,
                           args.home_config, args.planner_config, ComparisonConfig.load(args.config),
                           args.run_dir, args.scenario_dir, args.previous_verification)
        elif args.command == "continuous-energy":
            from .energy.continuous import ContinuousConfig
            from .energy.continuous_experiment import run_continuous_experiment
            run_continuous_experiment(args.data_dir, args.model_bundle, args.home_config,
                                     args.planner_config, ContinuousConfig.load(args.config),
                                     args.run_dir, args.scenario_dir, args.previous_verification)
        elif args.command == "energy-modes":
            from .energy.mode_experiment import ModeComparisonConfig, run_mode_experiment
            run_mode_experiment(ModeComparisonConfig.load(args.config), args.run_dir, data_dir=args.data_dir)
    except (ValueError, TypeError, KeyError, OSError, URLError) as exc:
        parser.exit(2, f"Error: {exc}\n")
