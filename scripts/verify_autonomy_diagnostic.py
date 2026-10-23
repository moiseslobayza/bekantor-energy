"""Read-only independent physical/accounting audit; writes only the new receipt."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.autonomy_diagnostics import import_lower_bound, oracle_windows
from bekantor_energy.energy.comparison import comparison_metrics
from bekantor_energy.energy.continuous import adjusted_cost
from bekantor_energy.energy.simulator import PhysicalSimulator, SelfConsumptionStrategy
from bekantor_energy.forecasting.windows import read_hourly


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)


def verify(root, data_dir):
    design = json.loads((root / "design.json").read_text())
    config = design["config"]
    prior = Path(config["previous_run"])
    preservation = json.loads((root / "preservation.json").read_text())
    for name, checksum in preservation["before"].items():
        if sha256(Path(name)) != checksum:
            raise AssertionError(f"Protected historical input changed: {name}")
    for name, checksum in design["source_sha256"].items():
        assert sha256(Path(name)) == checksum, name
    assert sha256(Path("docs/phase4_4-design.md")) == design["design_document_sha256"]
    validation_path = data_dir / "validation.csv"
    assert sha256(validation_path) == design["validation_sha256"]
    validation = read_hourly(validation_path)  # The only partition opened.
    metrics = pd.read_csv(root / "metrics.csv")
    sim = PhysicalSimulator()
    realized, predicted, midnights, forecast_rows, failed = 0, 0, 0, 0, []
    for task in design["tasks"]:
        folder = root / "cases" / task["task_id"]
        failures = json.loads((folder / "failures.json").read_text()) if (folder / "failures.json").exists() else {"worker": True}
        if failures:
            failed.append(task["task_id"])
        if not (folder / "ledger.csv").exists():
            assert failures
            continue
        case = json.loads((folder / "case.json").read_text())
        original = json.loads((prior / "cases" / case["case_id"] / "case.json").read_text())
        assert case["home"] == original["home"]
        assert case["solar"] == original["solar"] and case["actual_demand"] == original["actual_demand"]
        for key, value in original["planner"].items():
            assert case["planner"][key] == (task["storage_step_kwh"] if key == "storage_step_kwh" else value)
        home = HomeConfig(**case["home"])
        times = pd.date_range(case["start"], periods=case["hours"], freq="h")
        perfect = oracle_windows(validation, case["start"], case["hours"])
        close(perfect[:, 0], case["actual_demand"])
        forecast = pd.read_csv(folder / "forecasts.csv.gz", parse_dates=["origin", "target_timestamp"])
        if task["forecast"] == "oracle":
            expected = perfect
            assert case["model_id"] == "ORACLE_DIAGNOSTIC_NOT_PRODUCTION"
        else:
            frozen = pd.read_csv(prior / "cases" / case["case_id"] / "forecasts.csv.gz", parse_dates=["origin"])
            expected = frozen.pivot(index="origin", columns="horizon", values="predicted_demand_kwh").reindex(times).to_numpy()
            assert case["model_id"] == original["model_id"]
        ledger = pd.read_csv(folder / "ledger.csv", parse_dates=["timestamp"])
        plans = pd.read_csv(folder / "plans.csv.gz", parse_dates=["origin", "target_timestamp"])
        searches = pd.read_csv(folder / "searches.csv", parse_dates=["origin"])
        for row in forecast.itertuples():
            t = times.get_loc(row.origin)
            close(row.predicted_demand_kwh, expected[t, row.horizon - 1])
            assert row.target_timestamp == row.origin + pd.Timedelta(hours=row.horizon - 1)
            forecast_rows += 1
        first_actions = {}
        for origin, group in plans.groupby("origin", sort=True):
            group = group.sort_values("horizon")
            assert list(group.horizon) == list(range(1, 25))
            state = EnergyState(0, float(group.initial_kwh.iloc[0]))
            total = imported = 0.
            t = times.get_loc(origin)
            for row in group.itertuples():
                h = row.horizon - 1
                close(row.predicted_demand_kwh, expected[t, h])
                close(row.solar_kwh, case["solar"][t + h])
                close(row.tariff_sim_per_kwh, home.import_prices[row.target_timestamp.hour])
                close(state.stored_kwh, row.initial_kwh)
                action = EnergyAction(row.charge_kw, row.discharge_kw)
                result = sim.step(state, action, row.predicted_demand_kwh, row.solar_kwh,
                                  home, import_price=row.tariff_sim_per_kwh)
                close(result.state.stored_kwh, row.stored_kwh)
                close(result.grid_import_kwh, row.grid_import_kwh)
                close(result.cost, row.cost)
                state = result.state
                total += result.cost
                imported += result.grid_import_kwh
                predicted += 1
            assert state.stored_kwh >= home.min_soc * home.battery_capacity_kwh - 1e-9
            search = searches[searches.origin == origin].iloc[0]
            assert search.objective_mode == "autonomy"
            close(search.predicted_cost, total)
            close(search.objective_primary, imported)
            close(search.objective_secondary, total)
            first_actions[origin] = EnergyAction(group.charge_kw.iloc[0], group.discharge_kw.iloc[0])
        for strategy, group in ledger.groupby("strategy"):
            group = group.sort_values("step")
            state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
            for row in group.itertuples():
                t = row.step
                assert row.timestamp == times[t] and t == state.step
                close(row.demand_kwh, case["actual_demand"][t])
                close(row.solar_kwh, case["solar"][t])
                close(row.initial_kwh, state.stored_kwh)
                requested = EnergyAction(row.requested_charge_kw, row.requested_discharge_kw)
                expected_action = (SelfConsumptionStrategy().action(row.demand_kwh, row.solar_kwh, home)
                                   if strategy == "reactive" else first_actions[row.timestamp])
                close([requested.charge_kw, requested.discharge_kw], [expected_action.charge_kw, expected_action.discharge_kw])
                result = sim.step(state, requested, row.demand_kwh, row.solar_kwh, home,
                                  repair=True, import_price=home.import_prices[row.timestamp.hour])
                for name, value in asdict(result).items():
                    if name != "state":
                        close(value, getattr(row, name))
                close(result.state.stored_kwh, row.stored_kwh)
                if t and row.timestamp.hour == 0:
                    midnights += 1
                state = result.state
                realized += 1
            if strategy == "autonomy" and not failures:
                assert len(group) == 168
                record = metrics[(metrics.case_id == case["case_id"]) & (metrics.forecast == task["forecast"])
                                 & (metrics.storage_step_kwh == task["storage_step_kwh"])].iloc[0]
                recalculated = comparison_metrics(group.to_dict("records"), home)
                for name in ("grid_import_kwh", "import_cost_sim", "battery_initial_kwh", "battery_final_kwh",
                             "curtailed_kwh", "conversion_loss_kwh", "constraint_violations", "unserved_kwh"):
                    close(record[name], recalculated[name])
                close(record.adjusted_cost_sim, adjusted_cost(record.import_cost_sim, record.battery_initial_kwh,
                      record.battery_final_kwh, case["valuation_sim_per_stored_kwh"]))
    certificates = pd.read_csv(root / "physical_bound.csv")
    for row in certificates.itertuples():
        case = json.loads((prior / "cases" / row.case_id / "case.json").read_text())
        proof = import_lower_bound(case["actual_demand"], case["solar"][:168], HomeConfig(**case["home"]))
        close(row.import_lower_bound_kwh, proof["import_lower_bound_kwh"])
        close(row.reactive_grid_kwh, proof["import_lower_bound_kwh"])
        close(row.flow_cut_residual_kwh, 0.)
    complete = metrics[metrics.complete]
    close(complete.balance_gap_residual_kwh, 0.)
    assert (complete.gap_to_bound_kwh >= -1e-8).all()
    assert (complete.constraint_violations == 0).all() and (complete.unserved_kwh == 0).all()
    assert not complete.material_improvement.any()
    for row in complete.to_dict("records"):
        historical = json.loads((prior / "cases" / row["case_id"] / "metrics.json").read_text())
        reference = next(m for m in historical if m["strategy"] == "reactive")
        for key in ("grid_import_kwh", "curtailed_kwh", "conversion_loss_kwh", "battery_final_kwh", "adjusted_cost_sim", "import_cost_sim"):
            close(row["delta_" + key], row[key] - reference[key])
        if row["forecast"] in ("mlp", "reactive") and row["storage_step_kwh"] == .25:
            old = next(m for m in historical if m["strategy"] == row["strategy"])
            for key in ("grid_import_kwh", "import_cost_sim", "battery_initial_kwh", "battery_final_kwh"):
                close(row[key], old[key])
    for row in pd.read_csv(root / "summary.csv").to_dict("records"):
        subset = metrics[(metrics.forecast == row["forecast"]) & (metrics.storage_step_kwh == row["storage_step_kwh"])]
        if row["scope"] == "four_variable":
            subset = subset[subset.tariff_mode == "time_of_use"]
        good = subset[subset.complete]
        assert row["cases"] == len(subset) and row["completed"] == len(good)
        assert row["failures"] == len(subset) - len(good)
        for name in ("grid_import_kwh", "delta_grid_import_kwh", "gap_to_bound_kwh", "delta_curtailed_kwh",
                     "delta_conversion_loss_kwh", "delta_battery_final_kwh", "grid_charge_kwh", "pv_charge_kwh",
                     "unused_instant_discharge_kwh", "unused_instant_pv_charge_kwh", "import_cost_sim", "adjusted_cost_sim",
                     "planning_seconds", "expanded_nodes"):
            close(row[name + "_mean"], good[name].mean())
            close(row[name + "_std"], good[name].std(ddof=1))
    sensitivity = pd.read_csv(root / "terminal_sensitivity.csv")
    for row in sensitivity.itertuples():
        selected = complete[(complete.case_id == row.case_id) & (complete.forecast == row.forecast)
                            & (complete.storage_step_kwh == row.storage_step_kwh)].iloc[0]
        expected_value = selected.grid_import_kwh + row.stored_kwh_value_ac * (selected.battery_initial_kwh - selected.battery_final_kwh)
        close(row.stock_adjusted_import_kwh_equivalent, expected_value)
    differences = pd.read_csv(root / "controlled_differences.csv")
    for row in differences.itertuples():
        selected = complete[(complete.case_id == row.case_id) & (complete.strategy == "autonomy")].set_index(["forecast", "storage_step_kwh"])
        if row.intervention == "replace_mlp_by_oracle":
            before, after = selected.loc[("mlp", row.storage_step_kwh)], selected.loc[("oracle", row.storage_step_kwh)]
        else:
            before, after = selected.loc[(row.forecast, .25)], selected.loc[(row.forecast, .125)]
        close(row.grid_reduction_kwh, before.grid_import_kwh - after.grid_import_kwh)
        close(row.gap_before_kwh, before.delta_grid_import_kwh)
        close(row.gap_after_kwh, after.delta_grid_import_kwh)
    for row in pd.read_csv(root / "controlled_summary.csv").to_dict("records"):
        group = differences[(differences.intervention == row["intervention"]) &
                            (differences.forecast == row["forecast"]) &
                            (differences.storage_step_kwh == row["storage_step_kwh"])]
        assert len(group) == row["cases"]
        close(row["mean_grid_reduction_kwh"], group.grid_reduction_kwh.mean())
        close(row["std_grid_reduction_kwh"], group.grid_reduction_kwh.std(ddof=1))
        close(row["pct_excess_removed_ratio_of_sums"],
              100 * (group.gap_before_kwh.sum() - group.gap_after_kwh.sum()) / group.gap_before_kwh.sum())
    for row in pd.read_csv(root / "grid_charge_sources.csv").itertuples():
        record = complete[(complete.case_id == row.case_id) & (complete.forecast == row.forecast)
                          & (complete.storage_step_kwh == row.storage_step_kwh)].iloc[0]
        close(row.realized_grid_charge_kwh, record.grid_charge_kwh)
        close(row.realized_grid_charge_kwh - row.planned_grid_charge_kwh,
              row.unexpected_extra_grid_charge_kwh - row.less_grid_charge_than_planned_kwh)
    coverage = json.loads((root / "coverage.json").read_text())
    assert sorted(failed) == sorted(coverage["failed_task_ids"])
    summary = {"status": "passed", "new_tasks": len(design["tasks"]), "failed_tasks": failed,
               "historical_cases_at_physical_bound": len(certificates), "realized_steps_replayed": realized,
               "predicted_steps_replayed": predicted, "forecast_values_checked": forecast_rows,
               "midnight_continuity_checks": midnights, "protected_prior_files": len(preservation["before"]),
               "only_partition": "validation", "new_training": False,
               "verifier_sha256": sha256(Path(__file__)),
               "analysis_source_sha256": sha256(Path("scripts/analyze_autonomy_diagnostic.py")),
               "plot_source_sha256": sha256(Path("scripts/plot_autonomy_diagnostic.py")),
               "tests_source_sha256": sha256(Path("tests/test_autonomy_diagnostics.py")),
               "artifact_sha256": {str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "verification.json"}}
    write_json(root / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    args = parser.parse_args()
    verify(args.run_dir, args.data_dir)
