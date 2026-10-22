"""Verify four-way comparison: frozen artifacts, physical replay and paired stats.

No training, inference, test access or modification of the A* objective.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.comparison import STRATEGIES, METRIC_COLUMNS, comparison_metrics
from bekantor_energy.energy.simulator import PhysicalSimulator
from bekantor_energy.forecasting.windows import read_hourly


def verify(run_dir, data_dir):
    run_dir, data_dir = Path(run_dir), Path(data_dir)
    design = json.loads((run_dir / "design.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "reproducibility.json").read_text(encoding="utf-8"))
    preserved = json.loads((run_dir / "preservation.json").read_text(encoding="utf-8"))
    if not preserved["verified"]:
        raise ValueError("Previous artifact preservation failed")
    for value, expected in preserved["original_artifacts_before"].items():
        path = Path(value)
        if path.name.startswith("test") or sha256(path) != expected:
            raise ValueError(f"Changed previous artifact or forbidden source: {path}")
    path = data_dir / "validation.csv"
    if sha256(path) != meta["data"]["validation"]["sha256"]:
        raise ValueError("Validation source checksum mismatch")
    validation = read_hourly(path)
    frozen = {}
    for strategy, model in design["models"].items():
        with np.load(model["predictions_path"], allow_pickle=False) as saved:
            frozen[strategy] = {pd.Timestamp(t): values for t, values in zip(saved["origins"], saved["predicted_kwh"])}
    episodes = pd.read_csv(run_dir / "episodes.csv", parse_dates=["start"])
    if episodes.start.duplicated().any() or (episodes.start.diff().dropna() < pd.Timedelta(hours=24)).any():
        raise ValueError("Executed demand episodes overlap")
    records = pd.read_csv(run_dir / "results_by_case.csv")
    case_manifest = pd.read_csv(run_dir / "cases.csv")
    if len(records) != 4 * len(case_manifest) or records.groupby("case_id").strategy.nunique().ne(4).any():
        raise ValueError("Incomplete accounting of four strategies")
    checked_steps, checked_forecasts, planned_transitions, planned_count = 0, 0, 0, 0
    hashes = {}
    for manifest_row in case_manifest.itertuples():
        case_dir = run_dir / "cases" / manifest_row.case_id
        if (case_dir / "worker_failure.json").exists():
            hashes[str(case_dir / "worker_failure.json")] = sha256(case_dir / "worker_failure.json")
            continue
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        home = HomeConfig(**case["home"])
        ledger = pd.read_csv(case_dir / "ledger.csv", parse_dates=["timestamp"])
        results = {r["strategy"]: r for r in json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))}
        audit = json.loads((case_dir / "plan_audit.json").read_text(encoding="utf-8"))
        times = pd.date_range(case["start"], periods=24, freq="h")
        np.testing.assert_allclose(validation.loc[times, "load_kwh"].to_numpy(), case["actual_demand"], atol=1e-12)
        for strategy in STRATEGIES:
            group = ledger[ledger.strategy == strategy].reset_index(drop=True)
            if results[strategy]["status"] == "completed" and len(group) != 24:
                raise ValueError("Completed strategy has a partial ledger")
            state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
            for t, row in enumerate(group.itertuples()):
                if row.timestamp != times[t]:
                    raise ValueError("Temporal inconsistency in realized replay")
                np.testing.assert_allclose(row.demand_kwh, case["actual_demand"][t], atol=1e-12)
                np.testing.assert_allclose(row.solar_kwh, case["solar"][t], atol=1e-12)
                np.testing.assert_allclose(row.tariff_sim_per_kwh, home.import_prices[times[t].hour], atol=1e-12)
                result = PhysicalSimulator().step(state, EnergyAction(row.requested_charge_kw, row.requested_discharge_kw),
                         row.demand_kwh, row.solar_kwh, home, repair=True, import_price=row.tariff_sim_per_kwh)
                for field in ("grid_import_kwh", "cost", "curtailed_kwh", "charge_kwh", "discharge_kwh", "pv_direct_kwh",
                              "pv_charge_kwh", "grid_charge_kwh", "unserved_kwh", "conversion_loss_kwh"):
                    np.testing.assert_allclose(getattr(result, field), getattr(row, field), atol=1e-9, rtol=1e-9)
                np.testing.assert_allclose(result.state.stored_kwh, row.stored_kwh, atol=1e-9)
                if result.constraint_violations:
                    raise ValueError("Physical violation in saved trace")
                state = result.state
                checked_steps += 1
            if results[strategy]["status"] == "completed":
                actual_metrics = comparison_metrics(group.to_dict("records"), home)
                for field, value in actual_metrics.items():
                    if value is None:
                        assert results[strategy][field] is None
                    else:
                        np.testing.assert_allclose(value, results[strategy][field], atol=1e-9, rtol=1e-9)
                summary = records[(records.case_id == case["case_id"]) & (records.strategy == strategy)].iloc[0]
                for field in METRIC_COLUMNS:
                    np.testing.assert_allclose(actual_metrics[field], summary[field], atol=1e-9, rtol=1e-9)
            if strategy != "reactive":
                expected = 24 if results[strategy]["status"] == "completed" else len(group)
                if audit[strategy]["checked_plans"] != expected or audit[strategy]["checked_transitions"] != expected * 24:
                    raise ValueError("Missing full-plan feasibility checks")
                if audit[strategy]["max_balance_residual"] > 1e-9 or audit[strategy]["max_storage_residual"] > 1e-9:
                    raise ValueError("Predicted-plan balance audit failed")
                planned_count += audit[strategy]["checked_plans"]
                planned_transitions += audit[strategy]["checked_transitions"]
        forecasts = pd.read_csv(case_dir / "forecasts.csv.gz", parse_dates=["origin", "target_timestamp"])
        for (strategy, origin), group in forecasts.groupby(["strategy", "origin"]):
            expected_times = pd.date_range(origin, periods=24, freq="h")
            if not pd.DatetimeIndex(group.target_timestamp).equals(expected_times):
                raise ValueError("Forecast horizon does not align with issue time")
            if strategy == "oracle":
                expected = validation.loc[expected_times, "load_kwh"].to_numpy()
                if not group.is_oracle.all():
                    raise ValueError("Unmarked oracle data")
            else:
                expected = frozen[strategy][origin]
                if group.is_oracle.any():
                    raise ValueError("Oracle leakage into neural forecast")
            np.testing.assert_allclose(group.predicted_kwh, expected, atol=1e-12, rtol=1e-12)
            checked_forecasts += 1
        for path in case_dir.iterdir():
            if path.is_file():
                hashes[str(path)] = sha256(path)
    complete = records[records.four_way_complete]
    aggregate = pd.read_csv(run_dir / "aggregate.csv").set_index("strategy")
    for strategy, group in complete.groupby("strategy"):
        for metric in METRIC_COLUMNS:
            for stat in ("mean", "std", "median", "min", "max", "sum"):
                np.testing.assert_allclose(getattr(group[metric], stat)(), aggregate.loc[strategy, f"{metric}__{stat}"], atol=1e-9, rtol=1e-9)
    paired = pd.read_csv(run_dir / "paired_results.csv")
    for case_id, group in paired.groupby("case_id"):
        base = group.set_index("strategy").loc["reactive"]
        for row in group.itertuples():
            np.testing.assert_allclose(row.delta_import_cost_sim_vs_reactive, row.import_cost_sim - base.import_cost_sim, atol=1e-12)
            expected = 100 * (base.import_cost_sim - row.import_cost_sim) / base.import_cost_sim if base.import_cost_sim > 1e-12 else np.nan
            np.testing.assert_allclose(row.improvement_pct_import_cost_sim_vs_reactive, expected, atol=1e-9, equal_nan=True)
    for path in run_dir.iterdir():
        if path.is_file() and path.name != "verification.json":
            hashes[str(path)] = sha256(path)
    for path in (run_dir / "plots").glob("*.png"):
        hashes[str(path)] = sha256(path)
    result = {"status": "passed", "unique_nonoverlapping_days": len(episodes), "cases": len(case_manifest),
              "strategy_runs": len(records), "failed_strategy_runs": int(records.status.ne("completed").sum()),
              "realized_transitions_replayed": checked_steps, "full_plans_checked": planned_count,
              "planned_transitions_physically_checked": planned_transitions, "forecasts_checked": checked_forecasts,
              "same_conditions_verified": True, "frozen_forecasts_and_oracle_separation_verified": True,
              "aggregate_metrics_verified": True, "original_artifacts_unchanged": True,
              "protected_files": len(preserved["original_artifacts_before"]), "artifact_sha256": hashes}
    result["final_rendering_source_sha256"] = sha256(Path("src/bekantor_energy/energy/comparison_plots.py"))
    result["analysis_source_sha256"] = sha256(Path("scripts/analyze_energy_comparison.py"))
    write_json(run_dir / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    args = parser.parse_args()
    verify(args.run_dir, args.data_dir)
