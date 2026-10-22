"""Recheck saved energy ledgers and causal audit without neural training/inference."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.evaluation import realized_metrics
from bekantor_energy.energy.planner import AStarPlanner, PlannerConfig
from bekantor_energy.energy.simulator import PhysicalSimulator


def verify(run_dir):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config = HomeConfig(**metadata["home"])
    settings = PlannerConfig(**metadata["planner"])
    ledger = pd.read_csv(run_dir / "ledger.csv", parse_dates=["timestamp"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    preserved = json.loads((run_dir / "preservation.json").read_text(encoding="utf-8"))
    for path, expected in preserved["original_artifacts_before"].items():
        if Path(path).name.startswith("test") or sha256(Path(path)) != expected:
            raise ValueError("Previous artifact changed or held-out path in allowlist")
    groups = {name: group.reset_index(drop=True) for name, group in ledger.groupby("strategy")}
    a, b = groups["astar_mlp"], groups["self_consumption"]
    for column in ("timestamp", "demand_kwh", "solar_kwh", "tariff_sim_per_kwh", "grid_available"):
        np.testing.assert_array_equal(a[column], b[column])
    for name, group in groups.items():
        state = EnergyState(0, config.initial_soc * config.battery_capacity_kwh)
        for _, row in group.iterrows():
            r = PhysicalSimulator().step(state, EnergyAction(row.requested_charge_kw, row.requested_discharge_kw),
                    row.demand_kwh, row.solar_kwh, config, repair=True, grid_available=bool(row.grid_available),
                    import_price=row.tariff_sim_per_kwh)
            for field in ("grid_import_kwh", "curtailed_kwh", "cost", "charge_kwh", "discharge_kwh",
                          "unserved_kwh", "conversion_loss_kwh", "balance_residual_kwh", "storage_residual_kwh"):
                np.testing.assert_allclose(getattr(r, field), row[field], atol=1e-9, rtol=1e-9)
            np.testing.assert_allclose(r.state.stored_kwh, row.stored_kwh, atol=1e-9)
            state = r.state
        for key, value in realized_metrics(group.to_dict("records"), config).items():
            if value is None:
                assert metrics[name][key] is None
            else:
                np.testing.assert_allclose(value, metrics[name][key], atol=1e-9, rtol=1e-9)
    audit = json.loads((run_dir / "causality_audit.json").read_text(encoding="utf-8"))
    assert len(audit) == 24 and all(pd.Timestamp(row["last_history_timestamp"]) < pd.Timestamp(row["origin"]) for row in audit)
    first = pd.read_csv(run_dir / "first_plan.csv", parse_dates=["timestamp"])
    forecast = LoadForecast(first.timestamp.iloc[0].to_pydatetime(), tuple(first.timestamp.dt.to_pydatetime()),
                            tuple(first.predicted_demand_kwh), metadata["model_id"])
    inputs = PlanningInputs(forecast, tuple(first.solar_kwh), config, "saved_24h_forecast_reference")
    ucs = AStarPlanner(replace(settings, heuristic="zero")).plan(inputs, EnergyState(0, config.initial_soc * config.battery_capacity_kwh))
    initial_search = pd.read_csv(run_dir / "planning.csv").iloc[0]
    np.testing.assert_allclose(ucs.predicted_cost, initial_search.predicted_cost_next_24h, atol=1e-9)
    np.testing.assert_allclose(first.predicted_cost.sum(), ucs.predicted_cost, atol=1e-9)
    hashes = {str(p): sha256(p) for p in run_dir.glob("*.csv")}
    hashes.update({str(p): sha256(p) for p in (run_dir / "plots").glob("*.png")})
    for name in ("config.json", "metrics.json", "reproducibility.json", "preservation.json", "causality_audit.json", "completed.json"):
        hashes[str(run_dir / name)] = sha256(run_dir / name)
    result = {"status": "passed", "realized_ledgers_replayed": True, "same_conditions": True,
              "causal_forecast_history_verified": True, "original_artifacts_unchanged": True,
              "protected_files": len(preserved["original_artifacts_before"]),
              "initial_24h_astar_equals_ucs": True, "initial_plan_predicted_cost": ucs.predicted_cost,
              "ucs_expanded_nodes": ucs.expanded_nodes, "astar_expanded_nodes": int(initial_search.expanded_nodes),
              "ucs_seconds": ucs.elapsed_seconds, "artifact_sha256": hashes}
    result["final_rendering_source_sha256"] = sha256(Path("src/bekantor_energy/energy/plots.py"))
    write_json(run_dir / "verification.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run_dir)
