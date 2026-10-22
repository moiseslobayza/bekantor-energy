"""Independent ledger/accounting/causality checks; no fitting or test access."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.data.pvgis import PVGISQuery, parse_pvgis
from bekantor_energy.energy.simulator import PhysicalSimulator
from bekantor_energy.forecasting.windows import read_hourly


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-10)


def verify(run_dir, data_dir):
    root = Path(run_dir)
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "reproducibility.json").read_text(encoding="utf-8"))
    preservation = json.loads((root / "preservation.json").read_text(encoding="utf-8"))
    assert preservation["verified"]
    for filename, expected in preservation["original_artifacts_before"].items():
        path = Path(filename)
        assert not path.name.startswith("test"), "Held-out path in preservation ledger"
        assert sha256(path) == expected, f"Protected artifact changed: {path}"
    validation_path = Path(data_dir) / "validation.csv"
    assert sha256(validation_path) == meta["data"]["validation"]["sha256"]
    validation = read_hourly(validation_path)
    with np.load(design["frozen_predictions"], allow_pickle=False) as saved:
        frozen = {pd.Timestamp(t): values.copy() for t, values in zip(saved["origins"], saved["predicted_kwh"])}
    blocks = pd.read_csv(root / "blocks.csv", parse_dates=["start", "end_exclusive", "forecast_tail_end"])
    assert len(blocks) >= 2
    assert (blocks.end_exclusive - blocks.start >= pd.Timedelta(days=7)).all()
    assert blocks.start.iloc[1] > blocks.forecast_tail_end.iloc[0]
    raw = pd.read_csv(root / "results_by_case.csv")
    cases = pd.read_csv(root / "cases.csv")
    assert len(raw) == 3 * len(cases)
    assert raw.groupby("case_id").strategy.nunique().eq(3).all()
    physical_steps = predicted_steps = forecast_count = plan_count = continuity_checks = 0
    for manifest in cases.itertuples():
        directory = root / "cases" / manifest.case_id
        if (directory / "worker_failure.json").exists():
            assert raw[raw.case_id == manifest.case_id].status.eq("worker_failed").all()
            continue
        case = json.loads((directory / "case.json").read_text(encoding="utf-8"))
        home = HomeConfig(**case["home"])
        hours = case["hours"]
        times = pd.date_range(case["start"], periods=hours, freq="h")
        tail = pd.date_range(case["start"], periods=hours + 23, freq="h")
        assert tail[0] >= validation.index[0] and tail[-1] <= validation.index[-1]
        assert validation.loc[tail, "is_observed_complete"].all()
        close(validation.loc[times, "load_kwh"], case["actual_demand"])
        for key, expected in design["home_reference"].items():
            if key not in ("initial_soc", "import_tariff_per_kwh"):
                assert case["home"][key] == expected
        assert home.initial_soc == design["configuration"]["initial_soc"]
        expected_tariffs = ((design["configuration"]["flat_tariff_sim_per_kwh"],) if case["tariff_mode"] == "flat"
                            else tuple(design["home_reference"]["import_tariff_per_kwh"]))
        assert home.import_tariff_per_kwh == expected_tariffs
        assert case["planner"] == design["planner_reference"]
        assert case["terminal_min_soc"] == design["configuration"]["terminal_min_soc"]
        solar_frame = pd.read_csv(Path(meta["scenario_directory"]) / f"{manifest.case_id}.csv")
        utc = pd.DatetimeIndex(pd.to_datetime(solar_frame.pvgis_timestamp_utc, utc=True))
        assert utc.is_unique and utc.equals(pd.date_range(utc[0], periods=hours + 23, freq="h"))
        assert utc[0].tz_convert("America/Argentina/Cordoba").hour == 0
        mapped = pd.DatetimeIndex(pd.to_datetime(solar_frame.uci_scenario_timestamp))
        assert mapped.equals(tail)
        local = pd.DatetimeIndex(pd.to_datetime(solar_frame.pvgis_timestamp_cordoba, utc=True))
        assert local.equals(utc)
        solar_source = design["solar"][case["solar_profile"]]
        query = PVGISQuery(**solar_source["query"])
        payload = json.loads(Path(design["configuration"]["solar_sources"][case["solar_profile"]]).read_text(encoding="utf-8"))
        solar = parse_pvgis(payload, query).pv_kwh_estimate.to_numpy()[:hours + 23]
        close(solar, solar_frame.solar_kwh)
        close(solar, case["solar"])
        results = {r["strategy"]: r for r in json.loads((directory / "metrics.json").read_text(encoding="utf-8"))}
        failures = json.loads((directory / "failures.json").read_text(encoding="utf-8"))
        ledger = pd.read_csv(directory / "ledger.csv", parse_dates=["timestamp"])
        for name, scores in results.items():
            rows = ledger[ledger.strategy == name].reset_index(drop=True)
            assert len(rows) == scores["hours_completed"]
            if scores["status"] == "completed":
                assert len(rows) == hours and name not in failures
            else:
                assert name in failures
            state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
            for t, row in enumerate(rows.itertuples()):
                assert row.timestamp == times[t] and row.step == t
                close(row.initial_kwh, state.stored_kwh)
                close(row.demand_kwh, case["actual_demand"][t])
                close(row.solar_kwh, solar[t])
                close(row.tariff_sim_per_kwh, home.import_prices[times[t].hour])
                result = PhysicalSimulator().step(state, EnergyAction(row.requested_charge_kw, row.requested_discharge_kw),
                    row.demand_kwh, row.solar_kwh, home, repair=True, import_price=row.tariff_sim_per_kwh)
                for field in ("cost", "grid_import_kwh", "grid_export_kwh", "curtailed_kwh", "charge_kwh", "discharge_kwh",
                    "unserved_kwh", "pv_direct_kwh", "pv_charge_kwh", "grid_charge_kwh", "conversion_loss_kwh"):
                    close(getattr(result, field), getattr(row, field))
                close(result.state.stored_kwh, row.stored_kwh)
                close(row.soc, row.stored_kwh / home.battery_capacity_kwh)
                assert not result.constraint_violations
                if t and times[t].hour == 0:
                    close(row.initial_kwh, rows.stored_kwh.iloc[t - 1])
                    continuity_checks += 1
                state = result.state
                physical_steps += 1
            if rows.empty:
                continue
            # Independent telescoping physical balance, not a metric helper.
            close(home.initial_soc * home.battery_capacity_kwh + rows.solar_kwh.sum() + rows.grid_import_kwh.sum(),
                rows.stored_kwh.iloc[-1] + (rows.demand_kwh - rows.unserved_kwh).sum() + rows.curtailed_kwh.sum()
                + rows.grid_export_kwh.sum() + rows.conversion_loss_kwh.sum())
            used = (rows.pv_direct_kwh + rows.pv_charge_kwh).sum()
            independently = {"import_cost_sim": (rows.grid_import_kwh * rows.tariff_sim_per_kwh).sum(),
                "grid_import_kwh": rows.grid_import_kwh.sum(), "battery_initial_kwh": rows.initial_kwh.iloc[0],
                "battery_final_kwh": rows.stored_kwh.iloc[-1], "solar_utilization_pct": 100 * used / rows.solar_kwh.sum(),
                "unserved_kwh": rows.unserved_kwh.sum(), "constraint_violations": rows.constraint_violations.sum(),
                "charge_kwh": rows.charge_kwh.sum(), "discharge_kwh": rows.discharge_kwh.sum()}
            independently["adjusted_cost_sim"] = independently["import_cost_sim"] + case["valuation_sim_per_stored_kwh"] * (rows.initial_kwh.iloc[0] - rows.stored_kwh.iloc[-1])
            summary = raw[(raw.case_id == case["case_id"]) & (raw.strategy == name)].iloc[0]
            for key, value in independently.items():
                close(value, scores[key])
                close(value, summary[key])
        searches = pd.read_csv(directory / "searches.csv", parse_dates=["origin"])
        plans = pd.read_csv(directory / "plans.csv.gz", parse_dates=["origin", "target_timestamp"])
        forecasts = pd.read_csv(directory / "forecasts.csv.gz", parse_dates=["origin", "target_timestamp", "last_available_history_hour"])
        for origin, group in forecasts.groupby("origin"):
            assert pd.DatetimeIndex(group.target_timestamp).equals(pd.date_range(origin, periods=24, freq="h"))
            assert (group.last_available_history_hour == origin - pd.Timedelta(hours=1)).all()
            close(group.predicted_demand_kwh, frozen[origin])
            forecast_count += 1
        for (name, origin), group in plans.groupby(["strategy", "origin"]):
            assert len(group) == 24 and group.horizon.tolist() == list(range(1, 25))
            assert pd.DatetimeIndex(group.target_timestamp).equals(pd.date_range(origin, periods=24, freq="h"))
            search = searches[(searches.strategy == name) & (searches.origin == origin)]
            assert len(search) == 1
            search = search.iloc[0]
            t = times.get_loc(origin)
            close(group.predicted_demand_kwh, frozen[origin])
            close(group.solar_kwh, solar[t:t + 24])
            prices = tuple(home.import_prices[dt.hour] for dt in pd.date_range(origin, periods=24, freq="h"))
            shifted = replace(home, import_tariff_per_kwh=prices)
            close(group.tariff_sim_per_kwh, prices)
            actual_row = ledger[(ledger.strategy == name) & (ledger.timestamp == origin)].iloc[0]
            close(group.initial_kwh.iloc[0], actual_row.initial_kwh)
            close(search.initial_kwh, actual_row.initial_kwh)
            close(group.charge_kw.iloc[0], actual_row.requested_charge_kw)
            close(group.discharge_kw.iloc[0], actual_row.requested_discharge_kw)
            state = EnergyState(0, search.initial_kwh)
            for row in group.itertuples():
                close(row.initial_kwh, state.stored_kwh)
                result = PhysicalSimulator().step(state, EnergyAction(row.charge_kw, row.discharge_kw),
                    row.predicted_demand_kwh, row.solar_kwh, shifted)
                close(result.state.stored_kwh, row.stored_kwh)
                close(result.cost, row.cost)
                close(result.grid_import_kwh, row.grid_import_kwh)
                state = result.state
                predicted_steps += 1
            target = (case["terminal_min_soc"] if name.endswith("terminal") else home.min_soc) * home.battery_capacity_kwh
            assert state.stored_kwh >= target - 1e-9
            close(search.required_terminal_kwh, target)
            close(search.predicted_terminal_kwh, state.stored_kwh)
            close(search.predicted_cost, group.cost.sum())
            plan_count += 1
        assert len(searches) * 24 == len(plans)
    coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    good = raw.groupby("case_id").status.apply(lambda s: s.eq("completed").all())
    assert coverage["complete_three_way_cases"] == int(good.sum())
    complete = raw[raw.case_id.isin(good[good].index)]
    paired = pd.read_csv(root / "paired_results.csv")
    for name, group in complete.groupby("strategy"):
        stored = pd.read_csv(root / "aggregate.csv").set_index("strategy").loc[name]
        for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh", "battery_final_kwh"):
            for suffix, value in (("count", len(group)), ("mean", group[metric].mean()), ("std", group[metric].std()),
                                   ("min", group[metric].min()), ("max", group[metric].max()), ("sum", group[metric].sum())):
                close(stored[f"{metric}__{suffix}"], value)
    for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
        for label, name in (("reactive", "reactive"), ("original_mse", "mlp_mse")):
            base = complete[complete.strategy == name].set_index("case_id")[metric]
            close(paired[f"delta_{metric}_vs_{label}"], paired[metric] - paired.case_id.map(base))
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    assert set(sensitivity.valuation_sim_per_stored_kwh.round(10)) == set(round(v, 10) for v in design["configuration"]["valuation_sensitivity"])
    close(sensitivity.adjusted_cost_sim, sensitivity.import_cost_sim + sensitivity.valuation_sim_per_stored_kwh * (sensitivity.battery_initial_kwh - sensitivity.battery_final_kwh))
    for value, group in sensitivity.groupby("valuation_sim_per_stored_kwh"):
        base = group[group.strategy == "reactive"].set_index("case_id").adjusted_cost_sim
        close(group.delta_vs_reactive_sim, group.adjusted_cost_sim - group.case_id.map(base))
    daily = pd.read_csv(root / "daily_metrics.csv")
    for (case_id, name), group in daily.groupby(["case_id", "strategy"]):
        expected = raw[(raw.case_id == case_id) & (raw.strategy == name)].iloc[0]
        assert len(group) == design["configuration"]["days_per_block"]
        close(group.initial_stored_kwh.iloc[1:], group.final_stored_kwh.iloc[:-1])
        close(group.gross_cost_sim.sum(), expected.import_cost_sim)
        close(group.adjusted_cost_sim.sum(), expected.adjusted_cost_sim)
        close(group.grid_import_kwh.sum(), expected.grid_import_kwh)
    economic = pd.read_csv(root / "economic_summary.csv")
    for row in economic.itertuples():
        scope = complete if row.tariff_mode == "all" else complete[complete.tariff_mode == row.tariff_mode]
        candidate = scope[scope.strategy == row.strategy]
        baseline = scope[scope.strategy == "reactive"]
        for metric in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
            close(getattr(row, f"{metric}_mean"), candidate[metric].mean())
            close(getattr(row, f"{metric}_std"), candidate[metric].std())
            close(getattr(row, f"improvement_pct_{metric}_ratio_of_sums"),
                  100 * (baseline[metric].sum() - candidate[metric].sum()) / baseline[metric].sum())
    for source, expected in meta["source_sha256"].items():
        if Path(source).name != "continuous_plots.py":
            assert sha256(Path(source)) == expected, f"Simulation source changed after running: {source}"
    hashes = {str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "verification.json"}
    summary = {"status": "passed", "cases": len(cases), "strategy_runs": len(raw), "failures": int((raw.status != "completed").sum()),
        "continuous_blocks": len(blocks), "physical_steps_replayed": physical_steps, "midnight_continuity_checks": continuity_checks,
        "full_plans_replayed": plan_count, "predicted_steps_replayed": predicted_steps, "frozen_forecasts_verified": forecast_count,
        "protected_files": len(preservation["original_artifacts_before"]), "original_artifacts_unchanged": True,
        "terminal_accounting_and_sensitivity_verified": True, "same_conditions_and_solar_timestamps_verified": True,
        "artifact_sha256": hashes, "verifier_sha256": sha256(Path(__file__)),
        "analysis_source_sha256": sha256(Path(__file__).with_name("analyze_continuous_energy.py")),
        "final_rendering_source_sha256": sha256(Path("src/bekantor_energy/energy/continuous_plots.py"))}
    write_json(root / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "artifact_sha256"}, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    args = parser.parse_args()
    verify(args.run_dir, args.data_dir)
