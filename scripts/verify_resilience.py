"""Independent replay, source isolation, causal-prefix and paired-metric audit."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.resilience import POLICIES
from bekantor_energy.energy.resilience_experiment import load_bases
from bekantor_energy.energy.simulator import PhysicalSimulator, SelfConsumptionStrategy


def close(a, b, tolerance=1e-8):
    if not np.isfinite(float(a)) or not np.isfinite(float(b)) or abs(float(a) - float(b)) > tolerance:
        raise AssertionError(f"Mismatch: {a} vs {b}")


def emergency_phi(demand, solar, start, stored, home, fraction, duration):
    state, missing = EnergyState(0, stored), 0.
    simulator = PhysicalSimulator()
    for h in range(start, start + duration):
        load = fraction * demand[h]
        result = simulator.step(state, SelfConsumptionStrategy().action(load, solar[h], home), load, solar[h], home,
                                grid_available=False, repair=True)
        state, missing = result.state, missing + result.unserved_kwh
    return missing


def verify(root, data_dir, verified_cache=None):
    root = Path(root)
    design = json.loads((root / "design.json").read_text())
    protected = json.loads((root / "preservation.json").read_text())["before"]
    for name, checksum in protected.items():
        assert sha256(Path(name)) == checksum, name
    for name, checksum in design["source_sha256"].items():
        assert sha256(Path(name)) == checksum, name
    assert sha256(Path("docs/phase4_5-design-proposal.md")) == design["proposal_sha256"]
    assert sha256(data_dir / "validation.csv") == design["validation_sha256"]
    assert sha256(data_dir / "train.csv") == design["train_past_context_sha256"]
    config = design["config"]["resilience"]
    fraction, risk_hours = config["critical_fraction"], config["risk_hours"]
    bases = {b["case"]["case_id"]: b for b in load_bases(design["config"]["paired_reference"], data_dir)}
    old_plan_hashes = {}
    if verified_cache:
        receipt_path = Path(verified_cache) / "verification.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["status"] == "passed" and receipt["verifier_sha256"] == sha256(Path(__file__))
        old_plan_hashes = receipt["plan_sha256"]
    metrics = pd.read_csv(root / "metrics.csv")
    assert len(metrics) == design["expected_replays"]
    assert not metrics.duplicated(["case_id", "event_id", "policy"]).any()
    plans, plan_hashes = {}, {}
    new_plans, reused_plans, predicted_steps = 0, 0, 0
    for base_dir in sorted((root / "bases").iterdir()):
        base = bases[base_dir.name]
        inputs_file = json.loads((base_dir / "inputs.json").read_text())
        assert inputs_file["case"] == base["case"]
        for name in ("mlp", "persistence", "oracle"):
            np.testing.assert_allclose(inputs_file["forecasts"][name], base["forecasts"][name], atol=0, rtol=0)
        origin_times = pd.date_range(base["case"]["start"], periods=base["case"]["hours"], freq="h")
        for path in sorted((base_dir / "plans").glob("*.json.gz")):
            key = path.name.removesuffix(".json.gz")
            checksum = sha256(path)
            with gzip.open(path, "rt", encoding="utf-8") as stream: record = json.load(stream)
            inputs, plan = record["inputs"], record["plan"]
            assert hashlib.sha256(json.dumps(inputs, sort_keys=True, allow_nan=False).encode()).hexdigest() == key
            assert set(inputs) == {"policy", "origin", "forecast", "model_id", "solar", "home", "initial_kwh", "planner", "resilience", "sources"}
            assert inputs["sources"] == design["source_sha256"]
            assert inputs["resilience"] == config
            source = "oracle" if inputs["policy"] == "astar_oracle" else "persistence" if inputs["policy"] == "astar_persistence" else "mlp"
            t = origin_times.get_loc(pd.Timestamp(inputs["origin"]))
            np.testing.assert_allclose(inputs["forecast"], base["forecasts"][source][t], rtol=0, atol=0)
            np.testing.assert_allclose(inputs["solar"], base["case"]["solar"][t:t + 24], rtol=0, atol=0)
            home = HomeConfig(**inputs["home"])
            original_home = HomeConfig(**base["case"]["home"])
            for h in range(24): close(home.import_prices[h], original_home.import_prices[(origin_times[t].hour + h) % 24])
            assert plan["objective_mode"] == "resilience_experimental" and plan["heuristic_name"] == "zero"
            assert inputs["planner"]["heuristic"] == "zero"
            assert plan["objective_units"] == ["critical_kWh_readiness", "SIM"]
            assert len(plan["actions"]) == 24 and len(plan["states"]) == 25
            if old_plan_hashes.get(key) == checksum:
                reused_plans += 1
            else:
                state = EnergyState(0, inputs["initial_kwh"])
                close(state.stored_kwh, plan["states"][0]["stored_kwh"])
                cost, risk = 0., 0.
                for h, command in enumerate(plan["actions"]):
                    result = PhysicalSimulator().step(state, EnergyAction(**command), inputs["forecast"][h], inputs["solar"][h], home)
                    close(result.state.stored_kwh, plan["states"][h + 1]["stored_kwh"])
                    state, cost = result.state, cost + result.cost
                    if h + 1 <= 24 - risk_hours:
                        risk += emergency_phi(inputs["forecast"], inputs["solar"], h + 1,
                                    state.stored_kwh, home, fraction, risk_hours) / (24 - risk_hours)
                    predicted_steps += 1
                close(plan["objective_primary"], risk)
                close(plan["objective_secondary"], cost)
                close(plan["predicted_cost"], cost)
                assert home.min_soc * home.battery_capacity_kwh - 1e-8 <= state.stored_kwh <= home.battery_capacity_kwh + 1e-8
                new_plans += 1
            plan_hashes[key] = checksum
            plans[key] = record
    actual_steps, midnights, prefix_steps, restoration_steps = 0, 0, 0, 0
    ledgers = {}
    for metric in metrics.to_dict("records"):
        if metric["status"] != "completed":
            assert Path(metric["output_dir"], "failure.json").exists() if "output_dir" in metric else True
            continue
        base = bases[metric["case_id"]]
        case = base["case"]
        home = HomeConfig(**case["home"])
        folder = Path(metric["output_dir"])
        ledger = pd.read_csv(folder / "ledger.csv", parse_dates=["timestamp"])
        assert len(ledger) == case["hours"] == 168
        assert json.loads((folder / "failure.json").read_text()) is None
        state, ens, cost, imported, shed = EnergyState(0, home.initial_soc * home.battery_capacity_kwh), 0., 0., 0., 0.
        first_fail, full_hours, cut_steps = None, 0, []
        for row in ledger.itertuples():
            t = state.step
            up = not (metric["duration_hours"] and metric["onset_step"] <= t < metric["onset_step"] + metric["duration_hours"])
            assert row.grid_available == up
            assert row.timestamp == pd.Timestamp(case["start"]) + pd.Timedelta(hours=t)
            load, solar = case["actual_demand"][t], case["solar"][t]
            critical = fraction * load
            close(row.original_demand_kwh, load)
            close(row.critical_required_kwh, critical)
            close(row.noncritical_required_kwh, load - critical)
            close(row.initial_kwh, state.stored_kwh)
            requested = EnergyAction(row.requested_charge_kw, row.requested_discharge_kw)
            if not up:
                emergency = SelfConsumptionStrategy().action(critical, solar, home)
                close(requested.charge_kw, emergency.charge_kw)
                close(requested.discharge_kw, emergency.discharge_kw)
                assert pd.isna(row.plan_id)
            elif metric["policy"].startswith("astar_"):
                record = plans[row.plan_id]
                close(record["inputs"]["initial_kwh"], state.stored_kwh)
                assert record["inputs"]["origin"] == row.timestamp.isoformat()
                assert record["inputs"]["policy"] == metric["policy"]
                close(requested.charge_kw, record["plan"]["actions"][0]["charge_kw"])
                close(requested.discharge_kw, record["plan"]["actions"][0]["discharge_kw"])
            else:
                target = row.target_kwh
                if metric["policy"] == "reactive": close(target, home.min_soc * home.battery_capacity_kwh)
                if metric["policy"] == "fixed_reserve": close(target, config["operational_soc"] * home.battery_capacity_kwh)
                if metric["policy"] == "adaptive_mlp":
                    forecast = base["forecasts"]["mlp"][t]
                    future_solar = case["solar"][t:t + 24]
                    phi = emergency_phi(forecast, future_solar, 1, target, home, fraction, risk_hours)
                    if row.target_achievable:
                        close(phi, 0.)
                        if target > home.min_soc * home.battery_capacity_kwh + 1e-5:
                            assert emergency_phi(forecast, future_solar, 1, target - 1e-5, home, fraction, risk_hours) > 0
                    else:
                        close(target, home.battery_capacity_kwh)
                        assert phi > 0
                expected_charge = max(max(0., solar - load), max(0., target - state.stored_kwh) / home.charge_efficiency) if state.stored_kwh < target - 1e-9 or solar > load else 0.
                expected_discharge = 0. if expected_charge > 0 else min(max(0., load - solar), max(0., state.stored_kwh - target) * home.discharge_efficiency)
                close(requested.charge_kw, expected_charge / home.step_hours)
                close(requested.discharge_kw, expected_discharge / home.step_hours)
            dispatch = load if up else critical
            result = PhysicalSimulator().step(state, requested, dispatch, solar, home, grid_available=bool(up),
                    repair=True, import_price=home.import_prices[row.timestamp.hour])
            for name, value in asdict(result).items():
                if name != "state": close(value, getattr(row, name))
            close(row.stored_kwh, result.state.stored_kwh)
            critical_missing = 0. if up else result.unserved_kwh
            sacrifice = 0. if up else (1 - fraction) * load
            close(row.critical_ens_kwh, critical_missing)
            close(row.noncritical_shed_kwh, sacrifice)
            close(row.critical_served_kwh + row.noncritical_served_kwh + row.total_unserved_kwh, load)
            if not up:
                close(result.grid_import_kwh, 0.)
                cut_steps.append(t)
                if critical_missing <= config["ens_tolerance_kwh"]: full_hours += 1
                elif first_fail is None: first_fail = len(cut_steps) - 1
            if t and row.timestamp.hour == 0: midnights += 1
            state, ens, cost, imported, shed = result.state, ens + critical_missing, cost + result.cost, imported + result.grid_import_kwh, shed + sacrifice
            actual_steps += 1
        close(metric["critical_ens_kwh"], ens)
        close(metric["gross_cost_sim"], cost)
        close(metric["grid_import_kwh"], imported)
        close(metric["noncritical_shed_kwh"], shed)
        close(metric["final_kwh"], state.stored_kwh)
        close(metric["adjusted_cost_sim"], cost + config["terminal_value"] * (home.initial_soc * home.battery_capacity_kwh - state.stored_kwh))
        assert metric["violations"] == 0
        if cut_steps:
            close(metric["critical_full_hours"], full_hours)
            close(metric["initial_autonomy_hours"], first_fail if first_fail is not None else len(cut_steps))
            assert bool(metric["autonomy_right_censored"]) == (first_fail is None)
            assert bool(metric["no_critical_deficit"]) == (ens <= config["ens_tolerance_kwh"])
        else:
            assert pd.isna(metric["initial_autonomy_hours"]) and pd.isna(metric["no_critical_deficit"])
        ledgers[(metric["case_id"], metric["policy"], metric["event_id"])] = ledger
        if metric["policy"] == "reactive" and not metric["duration_hours"]:
            old = pd.read_csv(Path(design["config"]["paired_reference"]) / "cases" / metric["case_id"] / "ledger.csv")
            old = old[old.strategy == "reactive"]
            for field in ("stored_kwh", "grid_import_kwh", "cost", "curtailed_kwh", "conversion_loss_kwh"):
                np.testing.assert_allclose(ledger[field], old[field], rtol=0, atol=1e-8)
    prefix_cols = ["stored_kwh", "requested_charge_kw", "requested_discharge_kw", "critical_ens_kwh", "noncritical_shed_kwh"]
    for metric in metrics[(metrics.status == "completed") & (metrics.duration_hours > 0)].to_dict("records"):
        key = (metric["case_id"], metric["policy"])
        ledger = ledgers[(*key, metric["event_id"])]
        control = ledgers.get((*key, "no_outage"))
        onset = int(metric["onset_step"])
        if control is not None:
            np.testing.assert_allclose(ledger[prefix_cols].iloc[:onset], control[prefix_cols].iloc[:onset], rtol=0, atol=1e-8)
            prefix_steps += onset
        shorter = metrics[(metrics.case_id == metric["case_id"]) & (metrics.policy == metric["policy"]) &
                          (metrics.onset_step == onset) & metrics.duration_hours.lt(metric["duration_hours"]) & (metrics.status == "completed")]
        for row in shorter.itertuples():
            end = onset + int(row.duration_hours)
            np.testing.assert_allclose(ledger[prefix_cols].iloc[:end], ledgers[(*key, row.event_id)][prefix_cols].iloc[:end], rtol=0, atol=1e-8)
            restoration_steps += end
    paired = pd.read_csv(root / "paired_metrics.csv")
    for row in pd.read_csv(root / "paired_differences.csv").itertuples():
        a = paired[(paired.pair_id == row.pair_id) & (paired.policy == row.candidate)].iloc[0]
        b = paired[(paired.pair_id == row.pair_id) & (paired.policy == row.reference)].iloc[0]
        close(row.avoided_critical_kwh, b.critical_ens_kwh - a.critical_ens_kwh)
        close(row.delta_adjusted_cost_sim, a.adjusted_cost_sim - b.adjusted_cost_sim)
        close(row.delta_grid_import_kwh, a.grid_import_kwh - b.grid_import_kwh)
    for row in pd.read_csv(root / "comparison_summary.csv").itertuples():
        difference = pd.read_csv(root / "paired_differences.csv")
        selected = difference[(difference.comparison == row.comparison) &
                              ((difference.duration_hours > 0) if row.scope == "outage_all" else (difference.duration_hours == 0))]
        assert len(selected) == row.cases
        close(row.avoided_critical_mean_kwh, selected.avoided_critical_kwh.mean())
        close(row.avoided_critical_worst_kwh, selected.avoided_critical_kwh.min())
        ref = selected.critical_ens_reference_kwh.sum()
        if ref > 1e-6: close(row.aggregate_reduction_pct, 100 * selected.avoided_critical_kwh.sum() / ref)
    summary = {"status": "passed", "stage": design["stage"], "replays": len(metrics),
        "failed_replays": int((metrics.status != "completed").sum()),
        "actual_steps_replayed": actual_steps, "new_unique_plans_physically_checked": new_plans,
        "unique_plans_reused_from_verified_receipt": reused_plans, "new_predicted_steps_checked": predicted_steps,
        "midnight_continuity_checks": midnights, "causal_preoutage_prefix_steps": prefix_steps,
        "unknown_restoration_prefix_steps": restoration_steps, "protected_files": len(protected),
        "test_access": False, "neural_training": False, "verifier_sha256": sha256(Path(__file__)),
        "analysis_sha256": sha256(Path("scripts/analyze_resilience.py")), "plan_sha256": plan_hashes,
        "artifact_sha256": {str(p): sha256(p) for p in root.rglob("*") if p.is_file() and p.name != "verification.json"}}
    write_json(root / "verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("artifact_sha256", "plan_sha256")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    parser.add_argument("--verified-cache", type=Path)
    args = parser.parse_args()
    verify(args.run_dir, args.data_dir, args.verified_cache)
