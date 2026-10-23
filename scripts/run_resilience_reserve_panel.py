"""Fixed 25/75% reserve sensitivity and equal-SOC emergency counterfactual."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.planner import PlannerConfig
from bekantor_energy.energy.resilience import ResilienceConfig
from bekantor_energy.energy.resilience_experiment import PlanStore, load_bases, replay_policy, scenarios
from bekantor_energy.energy.simulator import PhysicalSimulator, SelfConsumptionStrategy


def run(reference, root, data_dir):
    if root.exists(): raise ValueError("New sensitivity directory required")
    receipt = json.loads((reference / "verification.json").read_text())
    expected = {**receipt["artifact_sha256"], str(reference / "verification.json"): sha256(reference / "verification.json")}
    assert all(sha256(Path(p)) == h for p, h in expected.items())
    design = json.loads((reference / "design.json").read_text())
    bases = load_bases(design["config"]["paired_reference"], data_dir)
    original = pd.read_csv(reference / "paired_metrics.csv")
    primary = ResilienceConfig(**design["config"]["resilience"])
    planner = PlannerConfig(**design["config"]["planner"])
    root.mkdir(parents=True)
    write_json(root / "design.json", {"parent": str(reference), "reserve_levels": [.25, .75],
        "panel": "eight 22:00/6h outages and eight no-outage controls", "fraction": primary.critical_fraction,
        "initial_soc_counterfactual": .5, "script_sha256": sha256(Path(__file__))})
    rows, counterfactuals, verified_steps = [], [], 0
    for base in bases:
        case = base["case"]
        home = HomeConfig(**case["home"])
        for reserve in (.25, .75):
            settings = replace(primary, operational_soc=reserve)
            for event in scenarios(case["hours"], "panel", case["tariff_mode"]):
                folder = root / "replays" / f'{case["case_id"]}__reserve{int(100*reserve)}__{event[0]}'
                metric = replay_policy(base, event, "fixed_reserve", planner, settings, PlanStore(root / "unused_plans"), folder)
                if metric["status"] == "completed":
                    ledger = pd.read_csv(folder / "ledger.csv")
                    state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
                    for row in ledger.itertuples():
                        assert abs(row.initial_kwh - state.stored_kwh) < 1e-8
                        step = PhysicalSimulator().step(state, EnergyAction(row.requested_charge_kw, row.requested_discharge_kw),
                            row.demand_kwh, row.solar_kwh, home, grid_available=bool(row.grid_available),
                            repair=True, import_price=row.tariff_sim_per_kwh)
                        assert abs(step.state.stored_kwh - row.stored_kwh) < 1e-8
                        assert abs(step.grid_import_kwh - row.grid_import_kwh) < 1e-8
                        assert abs(row.noncritical_shed_kwh - ((1 - settings.critical_fraction) * row.original_demand_kwh if not row.grid_available else 0.)) < 1e-8
                        state = step.state
                        verified_steps += 1
                rows.append({**metric, "operational_soc": reserve})
        # Independent diagnostic: same onset energy implies same emergency service.
        onset, duration = 94, 6
        trajectories = []
        for label in ("reactive", "fixed_reserve", "adaptive_mlp", "astar_persistence", "astar_mlp", "astar_oracle"):
            state, trace = EnergyState(0, .5 * home.battery_capacity_kwh), []
            for t in range(onset, onset + duration):
                critical = primary.critical_fraction * case["actual_demand"][t]
                result = PhysicalSimulator().step(state, SelfConsumptionStrategy().action(critical, case["solar"][t], home),
                    critical, case["solar"][t], home, grid_available=False, repair=True)
                state = result.state
                trace.append((state.stored_kwh, result.unserved_kwh))
            trajectories.append(trace)
            counterfactuals.append({"case_id": case["case_id"], "policy_label": label,
                "critical_ens_kwh": sum(x[1] for x in trace), "cut_initial_kwh": 2.5,
                "cut_final_kwh": state.stored_kwh, "not_a_continuous_replay": True})
        for trace in trajectories[1:]: np.testing.assert_allclose(trace, trajectories[0], rtol=0, atol=0)
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "metrics.csv", index=False)
    pd.DataFrame(counterfactuals).to_csv(root / "equal_soc_counterfactual.csv", index=False)
    summary = []
    for reserve, group in frame[frame.duration_hours > 0].groupby("operational_soc"):
        baseline = original[(original.policy == "fixed_reserve") & (original.event_id == "hour22_duration06")].set_index("case_id")
        summary.append({"operational_soc": reserve, "cases": len(group),
            "critical_ens_mean_kwh": float(group.critical_ens_kwh.mean()),
            "adjusted_cost_mean_sim": float(group.adjusted_cost_sim.mean()),
            "grid_import_mean_kwh": float(group.grid_import_kwh.mean()),
            "mean_avoided_critical_kwh_vs_50": float((group.case_id.map(baseline.critical_ens_kwh) - group.critical_ens_kwh).mean()),
            "mean_delta_adjusted_cost_vs_50": float((group.adjusted_cost_sim - group.case_id.map(baseline.adjusted_cost_sim)).mean())})
    pd.DataFrame(summary).to_csv(root / "summary.csv", index=False)
    assert all(sha256(Path(p)) == h for p, h in expected.items())
    write_json(root / "verification.json", {"status": "passed", "replays": len(rows),
        "failed_replays": int((frame.status != "completed").sum()), "physical_steps_replayed": verified_steps,
        "counterfactual_events": len(bases), "common_emergency_traces_equal": True,
        "parent_artifacts_unchanged": True, "script_sha256": sha256(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    args = parser.parse_args()
    run(args.reference, args.run_dir, args.data_dir)
