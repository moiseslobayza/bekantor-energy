"""Predeclared two-block validation experiment, frozen MSE forecasts only."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from ..config import HomeConfig
from ..data.acquisition import sha256, write_json
from ..forecasting.experiment import _checked_partition
from ..forecasting.mlp import MLPDemandPredictor
from ..forecasting.windows import make_windows
from .continuous import STRATEGIES, run_continuous_case, select_continuous_blocks, summarize_continuous
from .experiment import preservation_snapshot
from .planner import PlannerConfig
from .scenarios import prepare_solar_scenario


def run_continuous_experiment(data_dir, model_bundle, home_config, planner_config, config,
                              run_dir, scenario_dir, previous_verification, *, spot_check=True):
    data_dir, model_bundle, run_dir, scenario_dir = map(Path, (data_dir, model_bundle, run_dir, scenario_dir))
    if run_dir.exists() or scenario_dir.exists():
        raise ValueError("New result and scenario directories required; prior results are immutable")
    protected = [data_dir, model_bundle.parent, Path(previous_verification).parent]
    protected.extend(Path(p).parent for p in config.solar_sources.values())
    for output in (run_dir, scenario_dir):
        if any(output.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(output.resolve()) for p in protected):
            raise ValueError("Outputs must be disjoint from input artifacts")
    if run_dir.resolve().is_relative_to(scenario_dir.resolve()) or scenario_dir.resolve().is_relative_to(run_dir.resolve()):
        raise ValueError("Separate result/scenario roots required")
    home = HomeConfig.load(home_config)
    settings = PlannerConfig(**json.loads(Path(planner_config).read_text(encoding="utf-8")))
    if home.horizon_steps != 24 or home.step_hours != 1 or any(home.export_prices) or settings.terminal_min_soc not in (None, home.min_soc):
        raise ValueError("Keep the hourly 24h reserve-only original reference")
    if config.initial_soc < home.min_soc or config.terminal_min_soc < home.min_soc:
        raise ValueError("Scenario/terminal SOC below physical reserve")
    before = preservation_snapshot(previous_verification)
    for p in (Path(home_config), Path(planner_config), Path(__file__).parent / "planner.py", Path(__file__).parent / "simulator.py"):
        before[str(p.resolve())] = sha256(p)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    train = _checked_partition(data_dir, "train", manifest)  # Past input context, never a scored partition.
    validation = _checked_partition(data_dir, "validation", manifest)
    windows = make_windows(validation, history=train)
    metadata = json.loads((model_bundle / "predictor.json").read_text(encoding="utf-8"))
    if metadata["feature_mode"] != "future_calendar" or metadata["config"]["seed"] != config.seed:
        raise ValueError("Expected frozen calendar MSE seed-2026 bundle")
    prediction_path = model_bundle.parent / "validation_predictions.npz"
    if str(prediction_path.resolve()) not in before:
        raise ValueError("Frozen forecast provenance absent from protected experiments")
    with np.load(prediction_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["origins"], windows.origins.to_numpy())
        np.testing.assert_array_equal(saved["actual_kwh"], windows.y)
        predictions = saved["predicted_kwh"].copy()
    blocks, candidates = select_continuous_blocks(windows.origins, validation, config.days_per_block)
    spot = {}
    if spot_check:
        first = blocks.start.iloc[0]
        history = pd.concat([train, validation.loc[validation.index < first]])
        predictor = MLPDemandPredictor.load(model_bundle)
        inferred = np.asarray(predictor.predict(history, pd.date_range(first, periods=24, freq="h").to_pydatetime()).demand_kwh)
        expected = predictions[windows.origins.get_loc(first)]
        np.testing.assert_allclose(inferred, expected, atol=2e-6, rtol=1e-6)
        spot = {"origin": first.isoformat(), "last_history_hour": history.index[-1].isoformat(),
                "max_abs_difference_kwh": float(np.abs(inferred - expected).max())}
    hours = config.days_per_block * 24
    profiles, solar_meta = {}, {}
    source_times = []
    for label, raw in config.solar_sources.items():
        source, meta = prepare_solar_scenario(raw, home)
        if abs(meta["query"]["latitude"] + 31.4201) > 1e-6 or abs(meta["query"]["longitude"] + 64.1888) > 1e-6:
            raise ValueError("Use the declared Cordoba location")
        if len(source) < hours + 23:
            raise ValueError("Chronological solar source does not cover execution plus every forecast tail")
        selected = source.iloc[:hours + 23].copy()
        source_times.append(set(selected.pvgis_timestamp_utc))
        profiles[label], solar_meta[label] = selected, meta
        for p in (Path(raw), Path(raw).with_suffix(".metadata.json")):
            before[str(p.resolve())] = sha256(p)
    if source_times[0] & source_times[1]:
        raise ValueError("Solar scenario source periods must be independent/nonoverlapping")
    run_dir.mkdir(parents=True)
    scenario_dir.mkdir(parents=True)
    blocks.to_csv(run_dir / "blocks.csv", index=False)
    candidates.to_csv(run_dir / "candidate_blocks.csv", index=False)
    design = {"configuration": asdict(config), "home_reference": asdict(home), "planner_reference": asdict(settings),
        "terminal_variant": {**asdict(settings), "terminal_min_soc": config.terminal_min_soc},
        "selection": "first complete midnight block in each validation half; completeness only, not cost/forecast accuracy",
        "blocks": [{k: str(v) for k, v in row.items()} for row in blocks.to_dict("records")],
        "solar": solar_meta, "solar_assumption": "ideal knowledge of modeled chronological PVGIS path; no repeated synthetic days",
        "causality": "only frozen causal forecasts reach A*; future realized load is used solely in replay/scoring",
        "model_bundle": str(model_bundle.resolve()), "model_id": metadata["model_id"],
        "frozen_predictions": str(prediction_path.resolve()), "spot_inference": spot,
        "terminal_goal_scope": "predicted minimum SOC at each rolling 24h endpoint; not a hard realized reserve of 50%, nor an equal final block SOC guarantee",
        "accounting": "C_adjusted=C_grid+v*(E_initial-E_final); stored kWh, fixed v across policies/tariffs; not a bill or search cost",
        "valuation_reason": "v=0.15 SIM/kWh_AC * eta_discharge(0.95)=0.1425 SIM/kWh_stored, reference avoided import; sensitivity: zero, cheap tariff, reference, replacement, peak tariff",
        "heuristic_proof": "terminal constraint only removes goals/feasible paths; nonnegative edge costs unchanged; old relaxed bound ignores terminal replenishment, remains optimistic; checked against UCS",
        "statistical_unit": "two independent demand blocks crossed with two independent seasonal solar paths and two illustrative tariffs, not eight independent time series"}
    write_json(run_dir / "design.json", design)  # Persist BEFORE simulating outcomes.
    tasks, descriptions = [], []
    for block, solar_label, tariff in product(blocks.to_dict("records"), profiles, ("flat", "time_of_use")):
        times = pd.date_range(block["start"], periods=hours, freq="h")
        source = profiles[solar_label].copy()
        source["uci_scenario_timestamp"] = pd.date_range(block["start"], periods=len(source), freq="h")
        case_id = f"{block['block_id']}_{solar_label}_{tariff}"
        source.to_csv(scenario_dir / f"{case_id}.csv", index=False)
        scenario_home = replace(home, initial_soc=config.initial_soc,
            import_tariff_per_kwh=(config.flat_tariff_sim_per_kwh,) if tariff == "flat" else home.import_tariff_per_kwh)
        case = {"case_id": case_id, "block_id": block["block_id"], "solar_profile": solar_label,
                "tariff_mode": tariff, "start": block["start"].isoformat(), "hours": hours,
                "home": asdict(scenario_home), "planner": asdict(settings), "terminal_min_soc": config.terminal_min_soc,
                "valuation_sim_per_stored_kwh": config.valuation_sim_per_stored_kwh,
                "actual_demand": validation.loc[times, "load_kwh"].tolist(), "solar": source.solar_kwh.tolist(),
                "predictions": predictions[windows.origins.get_indexer(times)].tolist(), "model_id": metadata["model_id"]}
        tasks.append(case)
        descriptions.append({k: case[k] for k in ("case_id", "block_id", "solar_profile", "tariff_mode", "start", "hours")})
    pd.DataFrame(descriptions).to_csv(run_dir / "cases.csv", index=False)
    write_json(run_dir / "reproducibility.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "data": {name: manifest["partitions"][name] for name in ("train", "validation")},
        "source_sha256": {str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "new_neural_training": False, "evaluation_partition": "validation", "workers": config.workers,
        "scenario_directory": str(scenario_dir.resolve())})
    started, results = perf_counter(), []
    by_id = {c["case_id"]: c for c in descriptions}

    def record(result):
        results.extend({**by_id[result["case_id"]], **m} for m in result["metrics"])
        write_json(run_dir / "progress.json", {"completed_cases": len(results) // 3, "total_cases": len(tasks),
            "elapsed_seconds": perf_counter() - started, "failed_strategy_runs": sum(r["status"] != "completed" for r in results)})
        print(f"Completed {len(results) // 3}/{len(tasks)} continuous cases; failures={sum(r['status'] != 'completed' for r in results)}", flush=True)

    if config.workers == 1:
        for case in tasks:
            record(run_continuous_case(case, run_dir / "cases" / case["case_id"]))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = {pool.submit(run_continuous_case, case, run_dir / "cases" / case["case_id"]): case for case in tasks}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    failed_dir = run_dir / "cases" / case["case_id"]
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    write_json(failed_dir / "worker_failure.json", {"exception_type": type(error).__name__, "message": str(error)})
                    result = {"case_id": case["case_id"], "metrics": [{"strategy": n, "status": "worker_failed", "hours_completed": 0} for n in STRATEGIES]}
                record(result)
    results.sort(key=lambda row: (row["case_id"], STRATEGIES.index(row["strategy"])))
    summarize_continuous(results, run_dir, config.valuation_sensitivity)
    after = {path: sha256(Path(path)) for path in before}
    write_json(run_dir / "preservation.json", {"verified": before == after,
        "original_artifacts_before": before, "original_artifacts_after": after})
    if before != after:
        raise ValueError("Original artifacts or reference planner/physical engine changed")
    write_json(run_dir / "completed.json", {"status": "completed", "cases": len(tasks), "strategy_runs": len(results),
        "independent_demand_blocks": len(blocks), "hours_per_block": hours, "elapsed_seconds": perf_counter() - started,
        "failures": sum(r["status"] != "completed" for r in results), "new_neural_training": False,
        "evaluation_partition": "validation"})
    from .continuous_plots import continuous_figures
    continuous_figures(run_dir)
    return pd.DataFrame(results)
