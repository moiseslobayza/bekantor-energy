"""Run the predeclared Phase 4.4 ablations; never fit or access held-out data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.autonomy_diagnostics import (
    import_lower_bound, ledger_diagnostics, oracle_windows, replay_diagnostic,
    stock_adjusted_import,
)
from bekantor_energy.forecasting.windows import read_hourly


def load_cases(prior, validation):
    cases = []
    for folder in sorted((prior / "cases").iterdir()):
        case = json.loads((folder / "case.json").read_text(encoding="utf-8"))
        perfect = oracle_windows(validation, case["start"], case["hours"])
        np.testing.assert_allclose(perfect[:, 0], case["actual_demand"], atol=1e-12, rtol=0)
        frozen = pd.read_csv(folder / "forecasts.csv.gz", parse_dates=["origin", "target_timestamp"])
        origins = pd.date_range(case["start"], periods=case["hours"], freq="h")
        matrix = frozen.pivot(index="origin", columns="horizon", values="predicted_demand_kwh").reindex(origins)
        if list(matrix.columns) != list(range(1, 25)) or matrix.isna().any().any():
            raise ValueError("Incomplete frozen MLP forecasts")
        for row in frozen.itertuples():
            if row.target_timestamp != row.origin + pd.Timedelta(hours=row.horizon - 1):
                raise ValueError("Forecast calendar is inconsistent")
        cases.append((case, {"mlp": matrix.to_numpy(), "oracle": perfect}))
    return cases


def snapshot(prior):
    verified = json.loads((prior / "verification.json").read_text(encoding="utf-8"))
    preserved = json.loads((prior / "preservation.json").read_text(encoding="utf-8"))
    expected = {**preserved["original_artifacts_before"], **verified["artifact_sha256"]}
    expected[str(prior / "verification.json")] = sha256(prior / "verification.json")
    # Freeze the entire committed tree as well as the historical artifacts.
    for name in subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0"):
        if name:
            expected[str(Path(name).resolve())] = sha256(Path(name))
    for name, digest in expected.items():
        if Path(name).name.lower() in ("test.csv", "test.npz"):
            raise ValueError("Held-out data must not be in the snapshot")
        if sha256(Path(name)) != digest:
            raise ValueError(f"Historical artifact changed: {name}")
    return expected


def task_run(task):
    case, forecasts, spec, folder = task
    try:
        result = replay_diagnostic(case, forecasts, spec["storage_step_kwh"], spec["forecast"], folder)
        return {**spec, "output_dir": str(folder), **result}
    except Exception as error:
        folder.mkdir(parents=True, exist_ok=True)
        failure = {"exception_type": type(error).__name__, "message": str(error)}
        write_json(folder / "worker_failure.json", failure)
        return {**spec, "output_dir": str(folder), "case_id": case["case_id"],
                "failures": {"worker": failure}, "metrics": [{"strategy": "autonomy", "status": "worker_failed", "hours_completed": 0}]}


def metric_record(case, metric, folder, forecast, step):
    home = HomeConfig(**case["home"])
    result = {k: case[k] for k in ("case_id", "block_id", "solar_profile", "tariff_mode")}
    result.update(metric, forecast=forecast, storage_step_kwh=step, output_dir=str(folder))
    ledger_path = folder / "ledger.csv"
    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        ledger = ledger[ledger.strategy == metric["strategy"]]
        if len(ledger):
            result.update(ledger_diagnostics(ledger, home))
    return result


def analyze(root, cases, completions, config):
    prior = Path(config["previous_run"])
    case_map = {c["case_id"]: c for c, _ in cases}
    records, certificates = [], []
    for case, _ in cases:
        folder = prior / "cases" / case["case_id"]
        metrics = json.loads((folder / "metrics.json").read_text())
        for metric in metrics:
            if metric["strategy"] in ("reactive", "autonomy"):
                records.append(metric_record(case, metric, folder,
                        "reactive" if metric["strategy"] == "reactive" else "mlp", config["reference_step_kwh"]))
        bound = import_lower_bound(case["actual_demand"], case["solar"][:case["hours"]], HomeConfig(**case["home"]))
        reactive = next(m for m in metrics if m["strategy"] == "reactive")
        bound.update(case_id=case["case_id"], reactive_grid_kwh=reactive["grid_import_kwh"],
                     reactive_gap_to_bound_kwh=reactive["grid_import_kwh"] - bound["import_lower_bound_kwh"])
        certificates.append(bound)
    for result in completions:
        case = case_map[result["case_id"]]
        for metric in result["metrics"]:
            if metric["strategy"] == "autonomy":
                records.append(metric_record(case, metric, Path(result["output_dir"]), result["forecast"], result["storage_step_kwh"]))
            elif metric["strategy"] == "reactive":
                ref = next(r for r in records if r["case_id"] == case["case_id"] and r["strategy"] == "reactive")
                for key in ("grid_import_kwh", "battery_initial_kwh", "battery_final_kwh", "import_cost_sim", "unserved_kwh"):
                    if abs(metric[key] - ref[key]) > 1e-8:
                        raise ArithmeticError("Paired reactive conditions differ")
    frame = pd.DataFrame(records)
    frame["complete"] = (frame.status == "completed") & (frame.hours_completed == 168)
    base = frame[frame.strategy == "reactive"].set_index("case_id")
    bound_map = pd.DataFrame(certificates).set_index("case_id").import_lower_bound_kwh
    for name in ("grid_import_kwh", "curtailed_kwh", "conversion_loss_kwh", "battery_final_kwh", "adjusted_cost_sim", "import_cost_sim"):
        frame[f"delta_{name}"] = frame[name] - frame.case_id.map(base[name])
    frame["gap_to_bound_kwh"] = frame.grid_import_kwh - frame.case_id.map(bound_map)
    frame["grid_improvement_pct"] = -100 * frame.delta_grid_import_kwh / frame.case_id.map(base.grid_import_kwh)
    frame["same_terminal_as_reactive"] = frame.delta_battery_final_kwh.abs() < 1e-8
    frame["balance_gap_residual_kwh"] = (frame.delta_grid_import_kwh - frame.delta_curtailed_kwh
            - frame.delta_conversion_loss_kwh - frame.delta_battery_final_kwh)
    frame["material_improvement"] = (frame.complete & (frame.grid_improvement_pct > config["material_improvement_pct"])
            & (-frame.delta_grid_import_kwh > config["material_improvement_kwh_per_week"]))
    frame.to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(certificates).to_csv(root / "physical_bound.csv", index=False)
    stock = []
    for row in frame[frame.complete].to_dict("records"):
        reference = base.loc[row["case_id"]]
        for value in config["terminal_energy_values"]:
            adjusted = stock_adjusted_import(row["grid_import_kwh"], row["battery_initial_kwh"], row["battery_final_kwh"], value)
            base_adjusted = stock_adjusted_import(reference.grid_import_kwh, reference.battery_initial_kwh, reference.battery_final_kwh, value)
            stock.append({k: row[k] for k in ("case_id", "forecast", "storage_step_kwh", "strategy")} | {
                "stored_kwh_value_ac": value, "stock_adjusted_import_kwh_equivalent": adjusted,
                "delta_equivalent_vs_reactive": adjusted - base_adjusted})
    pd.DataFrame(stock).to_csv(root / "terminal_sensitivity.csv", index=False)
    summaries = []
    for scope, mask in (("all_eight", frame.storage_step_kwh == .25),
                        ("four_variable", frame.tariff_mode == config["sensitivity_tariff"])):
        selected = frame[mask]
        for (forecast, step), group in selected.groupby(["forecast", "storage_step_kwh"]):
            good = group[group.complete]
            item = {"scope": scope, "forecast": forecast, "storage_step_kwh": step,
                    "cases": len(group), "completed": len(good), "failures": len(group) - len(good)}
            for name in ("grid_import_kwh", "delta_grid_import_kwh", "gap_to_bound_kwh", "delta_curtailed_kwh",
                         "delta_conversion_loss_kwh", "delta_battery_final_kwh", "grid_charge_kwh", "pv_charge_kwh",
                         "unused_instant_discharge_kwh", "unused_instant_pv_charge_kwh", "import_cost_sim", "adjusted_cost_sim",
                         "planning_seconds", "expanded_nodes"):
                item[name + "_mean"] = float(good[name].mean())
                item[name + "_std"] = float(good[name].std(ddof=1))
            denominator = good.case_id.map(base.grid_import_kwh).sum()
            item["grid_change_pct_ratio_of_sums"] = 100 * good.delta_grid_import_kwh.sum() / denominator if denominator else None
            item["min_delta_grid_kwh"] = float(good.delta_grid_import_kwh.min())
            item["max_delta_grid_kwh"] = float(good.delta_grid_import_kwh.max())
            item["material_improvement_cases"] = int(good.material_improvement.sum())
            summaries.append(item)
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    coverage = {"new_tasks": len(completions), "new_completed_tasks": sum(not c["failures"] for c in completions),
                "historical_cases": len(cases), "failed_task_ids": [c["task_id"] for c in completions if c["failures"]],
                "oracle_reference_not_production": True, "historical_results_rewritten": False}
    write_json(root / "coverage.json", coverage)
    return frame


def run(config, root, data_dir):
    if root.exists():
        raise ValueError("A new run directory is required")
    prior = Path(config["previous_run"])
    if root.resolve().is_relative_to(prior.resolve()) or root.resolve().is_relative_to(data_dir.resolve()):
        raise ValueError("Output must be separate from historical inputs")
    expected = snapshot(prior)
    # The only partition file opened by this script is validation.csv.
    validation_path = data_dir / "validation.csv"
    repro = json.loads((prior / "reproducibility.json").read_text())
    if sha256(validation_path) != repro["validation_sha256"]:
        raise ValueError("Validation differs from the historical paired comparison")
    validation = read_hourly(validation_path)
    cases = load_cases(prior, validation)
    root.mkdir(parents=True)
    tasks, specifications = [], []
    for case, forecasts in cases:
        variants = [("oracle", config["reference_step_kwh"])]
        if case["tariff_mode"] == config["sensitivity_tariff"]:
            variants += [(method, step) for step in config["sensitivity_steps_kwh"] for method in ("mlp", "oracle")]
        for method, step in variants:
            spec = {"task_id": f'{case["case_id"]}__{method}__{step:g}'.replace(".", "p"),
                    "forecast": method, "storage_step_kwh": step}
            folder = root / "cases" / spec["task_id"]
            tasks.append((case, forecasts[method], spec, folder))
            specifications.append({**spec, "case_id": case["case_id"]})
    write_json(root / "design.json", {"config": config, "tasks": specifications,
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
        "neural_training": False, "partition": "validation", "oracle_only_receives_realized_future": True,
        "objective": "unchanged lexicographic (grid kWh, monetary SIM)",
        "validation_sha256": sha256(validation_path),
        "design_document_sha256": sha256(Path("docs/phase4_4-design.md")),
        "source_sha256": {str(p): sha256(p) for p in [Path(__file__), Path("src/bekantor_energy/energy/autonomy_diagnostics.py")]}})
    write_json(root / "preservation.json", {"before": expected})
    started, completions = perf_counter(), []
    with ProcessPoolExecutor(max_workers=config["workers"]) as executor:
        futures = {executor.submit(task_run, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            completions.append(result)
            write_json(root / "progress.json", {"completed_tasks": len(completions), "tasks": len(tasks),
                "failed_tasks": sum(bool(c["failures"]) for c in completions), "seconds": perf_counter() - started})
            print(f'{len(completions)}/{len(tasks)}: {result["task_id"]}; failures={list(result["failures"])}', flush=True)
    completions.sort(key=lambda c: c["task_id"])
    write_json(root / "completions.json", completions)
    analyze(root, cases, completions, config)
    for name, digest in expected.items():
        if sha256(Path(name)) != digest:
            raise ArithmeticError(f"Protected input changed: {name}")
    write_json(root / "preservation.json", {"before": expected, "after_matches": True, "protected_files": len(expected)})
    write_json(root / "completed.json", {"tasks": len(tasks), "failed_tasks": sum(bool(c["failures"]) for c in completions),
            "seconds": perf_counter() - started, "no_training": True, "partition": "validation"})
    print(json.dumps(json.loads((root / "completed.json").read_text()), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/energy.autonomy-diagnostic.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.config.read_text()), args.run_dir, args.data_dir)
