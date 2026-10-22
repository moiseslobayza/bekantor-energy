"""Independent physical, objective, pairing and frozen-provenance verification."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.mode_experiment import MODE_STRATEGIES
from bekantor_energy.energy.simulator import PhysicalSimulator
from bekantor_energy.forecasting.windows import read_hourly


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)


def verify(run_dir, data_dir):
    root = Path(run_dir)
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))
    repro = json.loads((root / "reproducibility.json").read_text(encoding="utf-8"))
    preservation = json.loads((root / "preservation.json").read_text(encoding="utf-8"))
    assert preservation["verified"]
    for filename, expected in preservation["original_artifacts_before"].items():
        path = Path(filename)
        assert not path.name.startswith("test"), "Held-out file in preservation allowlist"
        assert sha256(path) == expected, f"Prior result/model/data or source archive changed: {path}"
    for name, expected in repro["source_sha256"].items():
        if Path(name).name != "mode_plots.py":
            assert sha256(Path(name)) == expected, f"Simulation source changed after running: {name}"
    path = Path(data_dir) / "validation.csv"
    assert sha256(path) == repro["validation_sha256"]
    validation = read_hourly(path)
    prior = Path(design["configuration"]["previous_run"])
    frozen_path = design["previous_design"]["frozen_predictions"]
    with np.load(frozen_path, allow_pickle=False) as saved:
        frozen = {pd.Timestamp(t): values.copy() for t, values in zip(saved["origins"], saved["predicted_kwh"])}
    records = pd.read_csv(root / "results_by_case.csv")
    cases = pd.read_csv(root / "cases.csv")
    assert len(records) == 4 * len(cases) and records.groupby("case_id").strategy.nunique().eq(4).all()
    assert cases.case_id.tolist() == pd.read_csv(prior / "cases.csv").case_id.tolist()
    transitions = planned_transitions = plans_count = forecasts_count = midnight_checks = 0
    economic_cases = []
    for manifest in cases.itertuples():
        directory = root / "cases" / manifest.case_id
        if (directory / "worker_failure.json").exists():
            assert records[records.case_id == manifest.case_id].status.eq("worker_failed").all()
            continue
        case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        original = json.loads((prior / "cases" / manifest.case_id / "case.json").read_text(encoding="utf-8"))
        for field in ("home", "planner", "actual_demand", "solar", "model_id", "start", "hours"):
            assert case[field] == original[field], f"Conditions changed: {field}"
        assert case["objectives"] == design["objectives"]
        home = HomeConfig(**case["home"])
        hours = case["hours"]
        times = pd.date_range(case["start"], periods=hours, freq="h")
        tail = pd.date_range(case["start"], periods=hours + 23, freq="h")
        assert times[0] >= validation.index[0] and tail[-1] <= validation.index[-1]
        assert validation.loc[tail, "is_observed_complete"].all()
        close(validation.loc[times, "load_kwh"], case["actual_demand"])
        solar = np.asarray(case["solar"], float)
        scenario = pd.read_csv(Path(design["scenario_directory"]) / f"{manifest.case_id}.csv")
        utc = pd.DatetimeIndex(pd.to_datetime(scenario.pvgis_timestamp_utc, utc=True))
        assert utc.is_unique and utc.equals(pd.date_range(utc[0], periods=len(tail), freq="h"))
        assert pd.DatetimeIndex(pd.to_datetime(scenario.uci_scenario_timestamp)).equals(tail)
        close(scenario.solar_kwh, solar)
        ledger = pd.read_csv(directory / "ledger.csv", parse_dates=["timestamp"])
        searches = pd.read_csv(directory / "searches.csv", parse_dates=["origin"])
        plans = pd.read_csv(directory / "plans.csv.gz", parse_dates=["origin", "target_timestamp"])
        forecasts = pd.read_csv(directory / "forecasts.csv.gz", parse_dates=["origin", "target_timestamp", "last_available_history_hour"])
        scores = {r["strategy"]: r for r in json.loads((directory / "metrics.json").read_text(encoding="utf-8"))}
        failures = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
        for name in MODE_STRATEGIES:
            rows = ledger[ledger.strategy == name].reset_index(drop=True)
            assert len(rows) == scores[name]["hours_completed"]
            if scores[name]["status"] == "completed":
                assert len(rows) == hours and name not in failures
            else:
                assert name in failures
            if rows.empty:
                continue
            assert pd.DatetimeIndex(rows.timestamp).equals(times[:len(rows)])
            close(rows.demand_kwh, case["actual_demand"][:len(rows)])
            close(rows.solar_kwh, solar[:len(rows)])
            close(rows.tariff_sim_per_kwh, [home.import_prices[t.hour] for t in times[:len(rows)]])
            fields = ("cost", "grid_import_kwh", "curtailed_kwh", "charge_kwh", "discharge_kwh", "pv_direct_kwh", "pv_charge_kwh",
                      "grid_charge_kwh", "conversion_loss_kwh", "unserved_kwh", "constraint_violations")
            replayed = []
            state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
            for t, row in enumerate(rows.itertuples()):
                result = PhysicalSimulator().step(state, EnergyAction(row.requested_charge_kw, row.requested_discharge_kw),
                    row.demand_kwh, row.solar_kwh, home, repair=True, import_price=row.tariff_sim_per_kwh)
                replayed.append([state.stored_kwh, result.state.stored_kwh, *[getattr(result, f) for f in fields]])
                state = result.state
                transitions += 1
                midnight_checks += int(t > 0 and row.timestamp.hour == 0)
            close(np.array(replayed), rows[["initial_kwh", "stored_kwh", *fields]].to_numpy())
            close(rows.soc, rows.stored_kwh / home.battery_capacity_kwh)
            close(rows.initial_kwh.iloc[1:], rows.stored_kwh.iloc[:-1])
            assert rows.constraint_violations.eq(0).all()
            close(rows.initial_kwh.iloc[0] + rows.solar_kwh.sum() + rows.grid_import_kwh.sum(),
                rows.stored_kwh.iloc[-1] + (rows.demand_kwh - rows.unserved_kwh).sum() + rows.curtailed_kwh.sum()
                + rows.grid_export_kwh.sum() + rows.conversion_loss_kwh.sum())
            gross = float((rows.grid_import_kwh * rows.tariff_sim_per_kwh).sum())
            value = case["valuation_sim_per_stored_kwh"]
            adjusted = gross + value * (rows.initial_kwh.iloc[0] - rows.stored_kwh.iloc[-1])
            independent = {"import_cost_sim": gross, "adjusted_cost_sim": adjusted,
                "grid_import_kwh": rows.grid_import_kwh.sum(), "battery_initial_kwh": rows.initial_kwh.iloc[0],
                "battery_final_kwh": rows.stored_kwh.iloc[-1], "curtailed_kwh": rows.curtailed_kwh.sum(),
                "solar_utilization_pct": 100 * (rows.pv_direct_kwh + rows.pv_charge_kwh).sum() / rows.solar_kwh.sum(),
                "min_soc": min(home.initial_soc, rows.soc.min()),
                "mean_soc": ((rows.initial_kwh + rows.stored_kwh) / (2 * home.battery_capacity_kwh)).mean(), "final_soc": rows.soc.iloc[-1],
                "unserved_kwh": rows.unserved_kwh.sum(), "constraint_violations": rows.constraint_violations.sum()}
            summary = records[(records.case_id == manifest.case_id) & (records.strategy == name)].iloc[0]
            for field, expected in independent.items():
                close(expected, scores[name][field])
                close(expected, summary[field])
            if name == "economic" and scores[name]["status"] == "completed":
                old = pd.read_csv(prior / "cases" / manifest.case_id / "ledger.csv", parse_dates=["timestamp"])
                old = old[old.strategy == "mlp_mse"].reset_index(drop=True)
                close(old[["initial_kwh", "stored_kwh", "requested_charge_kw", "requested_discharge_kw", *fields]],
                      rows[["initial_kwh", "stored_kwh", "requested_charge_kw", "requested_discharge_kw", *fields]])
                assert old.timestamp.equals(rows.timestamp)
                economic_cases.append(manifest.case_id)
        for origin, group in forecasts.groupby("origin"):
            assert pd.DatetimeIndex(group.target_timestamp).equals(pd.date_range(origin, periods=24, freq="h"))
            close(group.predicted_demand_kwh, frozen[origin])
            assert group.last_available_history_hour.eq(origin - pd.Timedelta(hours=1)).all()
            forecasts_count += 1
        for (name, origin), group in plans.groupby(["strategy", "origin"]):
            assert len(group) == 24 and group.horizon.tolist() == list(range(1, 25))
            assert pd.DatetimeIndex(group.target_timestamp).equals(pd.date_range(origin, periods=24, freq="h"))
            t = times.get_loc(origin)
            close(group.predicted_demand_kwh, frozen[origin])
            close(group.solar_kwh, solar[t:t + 24])
            prices = tuple(home.import_prices[dt.hour] for dt in pd.date_range(origin, periods=24, freq="h"))
            close(group.tariff_sim_per_kwh, prices)
            shifted = replace(home, import_tariff_per_kwh=prices)
            search = searches[(searches.strategy == name) & (searches.origin == origin)]
            assert len(search) == 1
            search = search.iloc[0]
            actual = ledger[(ledger.strategy == name) & (ledger.timestamp == origin)].iloc[0]
            close(actual.initial_kwh, search.initial_kwh)
            close(actual.requested_charge_kw, group.charge_kw.iloc[0])
            close(actual.requested_discharge_kw, group.discharge_kw.iloc[0])
            state, expected_rows = EnergyState(0, search.initial_kwh), []
            for row in group.itertuples():
                result = PhysicalSimulator().step(state, EnergyAction(row.charge_kw, row.discharge_kw),
                    row.predicted_demand_kwh, row.solar_kwh, shifted)
                expected_rows.append([state.stored_kwh, result.state.stored_kwh, result.cost, result.grid_import_kwh])
                state = result.state
            close(expected_rows, group[["initial_kwh", "stored_kwh", "cost", "grid_import_kwh"]])
            assert state.stored_kwh >= home.min_soc * home.battery_capacity_kwh - 1e-9
            gross, grid = group.cost.sum(), group.grid_import_kwh.sum()
            close(search.predicted_cost, gross)
            close(search.predicted_grid_import_kwh, grid)
            assert search.objective_mode == name
            objective = design["objectives"][name]
            if name == "autonomy":
                close(search.objective_primary, grid)
                close(search.objective_secondary, gross)
                assert json.loads(search.objective_units) == ["kWh", "SIM"]
            elif name == "economic":
                close(search.objective_primary, gross)
                assert pd.isna(search.objective_secondary)
                assert json.loads(search.objective_units) == ["SIM"]
            else:
                expected = objective["cost_weight"] * gross / objective["cost_scale_sim"] + (1 - objective["cost_weight"]) * grid / objective["grid_scale_kwh"]
                close(search.objective_primary, expected)
                assert pd.isna(search.objective_secondary)
                assert json.loads(search.objective_units) == ["dimensionless"]
            plans_count += 1
            planned_transitions += len(group)
        assert len(searches) * 24 == len(plans)
    coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    good = records.groupby("case_id").status.apply(lambda values: values.eq("completed").all())
    assert coverage["complete_four_way_cases"] == int(good.sum())
    complete = records[records.case_id.isin(good[good].index)]
    paired = pd.read_csv(root / "paired_results.csv")
    for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
        for name in ("reactive", "economic"):
            base = complete[complete.strategy == name].set_index("case_id")[metric]
            close(paired[f"delta_{metric}_vs_{name}"], paired[metric] - paired.case_id.map(base))
    for filename, keys in (("aggregate.csv", ["strategy"]), ("by_scenario.csv", ["solar_profile", "tariff_mode", "strategy"]),
                           ("by_block.csv", ["block_id", "tariff_mode", "strategy"])):
        stored = pd.read_csv(root / filename).set_index(keys)
        for key, group in complete.groupby(keys):
            if len(keys) == 1:
                key = key[0]
            for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh", "curtailed_kwh", "mean_soc", "battery_final_kwh"):
                for suffix, value in (("mean", group[metric].mean()), ("std", group[metric].std()), ("min", group[metric].min()), ("max", group[metric].max()), ("sum", group[metric].sum())):
                    close(stored.loc[key, f"{metric}__{suffix}"], value)
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    close(sensitivity.adjusted_cost_sim, sensitivity.import_cost_sim + sensitivity.valuation_sim_per_stored_kwh * (sensitivity.battery_initial_kwh - sensitivity.battery_final_kwh))
    daily = pd.read_csv(root / "daily_metrics.csv")
    for (case_id, name), group in daily.groupby(["case_id", "strategy"]):
        total = complete[(complete.case_id == case_id) & (complete.strategy == name)].iloc[0]
        assert len(group) == 7
        close(group.initial_stored_kwh.iloc[1:], group.final_stored_kwh.iloc[:-1])
        close(group.gross_cost_sim.sum(), total.import_cost_sim)
        close(group.adjusted_cost_sim.sum(), total.adjusted_cost_sim)
        close(group.grid_import_kwh.sum(), total.grid_import_kwh)
    summaries = pd.read_csv(root / "mode_summary.csv")
    decomposition = pd.read_csv(root / "grid_gap_decomposition.csv")
    for field in ("grid_import_kwh", "curtailed_kwh", "conversion_loss_kwh", "battery_initial_kwh", "battery_final_kwh"):
        reference = complete[complete.strategy == "reactive"].set_index("case_id")[field]
        expected = paired[field] - paired.case_id.map(reference)
        close(decomposition[f"delta_{field}"], expected)
    close(decomposition.balance_residual_kwh, 0.)
    for row in summaries.itertuples():
        scope = complete if row.tariff_mode == "all" else complete[complete.tariff_mode == row.tariff_mode]
        candidate = scope[scope.strategy == row.strategy]
        for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
            close(getattr(row, f"{metric}_mean"), candidate[metric].mean())
            close(getattr(row, f"{metric}_std"), candidate[metric].std())
            for ref in ("reactive", "economic"):
                baseline = scope[scope.strategy == ref][metric].sum()
                close(getattr(row, f"reduction_pct_{metric}_vs_{ref}"), 100 * (baseline - candidate[metric].sum()) / baseline)
    regression = json.loads((root / "economic_regression.json").read_text(encoding="utf-8"))
    assert regression["all_match"] and len(economic_cases) == len(cases)
    summary = {"status": "passed", "cases": len(cases), "strategy_runs": len(records),
        "failed_strategy_runs": int((records.status != "completed").sum()), "realized_steps_replayed": transitions,
        "full_plans_replayed": plans_count, "predicted_steps_replayed": planned_transitions,
        "frozen_forecasts_checked": forecasts_count, "midnight_continuity_checks": midnight_checks,
        "protected_prior_artifacts_and_archives": len(preservation["original_artifacts_before"]),
        "economic_reference_identical_cases": len(economic_cases), "physical_and_objective_audits_passed": True,
        "prior_results_models_data_unchanged": True, "active_source_evolution_is_explicit": True,
        "artifact_sha256": {str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "verification.json"},
        "verifier_sha256": sha256(Path(__file__)), "analysis_source_sha256": sha256(Path(__file__).with_name("analyze_energy_modes.py")),
        "final_rendering_source_sha256": sha256(Path("src/bekantor_energy/energy/mode_plots.py"))}
    write_json(root / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "artifact_sha256"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    args = parser.parse_args()
    verify(args.run_dir, args.data_dir)
