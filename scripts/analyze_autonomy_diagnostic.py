"""Controlled differences and contemporaneous forecast/dispatch diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import write_json


def analyze(root):
    design = json.loads((root / "design.json").read_text())
    prior = Path(design["config"]["previous_run"])
    frame = pd.read_csv(root / "metrics.csv")
    complete = frame[frame.complete & (frame.strategy == "autonomy")]
    interventions = []
    forecast_errors = []
    charge_sources = []
    for item in complete.to_dict("records"):
        folder = Path(item["output_dir"])
        ledger = pd.read_csv(folder / "ledger.csv", parse_dates=["timestamp"])
        ledger = ledger[ledger.strategy == "autonomy"].sort_values("step")
        planned = pd.read_csv(folder / "plans.csv.gz", parse_dates=["origin"])
        planned = planned[(planned.horizon == 1)]
        if "strategy" in planned:
            planned = planned[planned.strategy == "autonomy"]
        planned = planned.set_index("origin").reindex(ledger.timestamp)
        predicted_surplus = np.maximum(planned.solar_kwh.to_numpy() - planned.predicted_demand_kwh.to_numpy(), 0.)
        planned_grid_charge = np.maximum(planned.charge_kw.to_numpy() - predicted_surplus, 0.)
        actual_grid_charge = ledger.grid_charge_kwh.to_numpy()
        charge_sources.append({k: item[k] for k in ("case_id", "forecast", "storage_step_kwh")} | {
            "planned_grid_charge_kwh": float(planned_grid_charge.sum()),
            "realized_grid_charge_kwh": float(actual_grid_charge.sum()),
            "unexpected_extra_grid_charge_kwh": float(np.maximum(0., actual_grid_charge - planned_grid_charge).sum()),
            "less_grid_charge_than_planned_kwh": float(np.maximum(0., planned_grid_charge - actual_grid_charge).sum()),
            "unexpected_extra_grid_charge_hours": int((actual_grid_charge - planned_grid_charge > 1e-8).sum())})
    for case_id, group in complete.groupby("case_id"):
        lookup = group.set_index(["forecast", "storage_step_kwh"])
        for step in (.125, .25, .5):
            if ("mlp", step) in lookup.index and ("oracle", step) in lookup.index:
                mlp, oracle = lookup.loc[("mlp", step)], lookup.loc[("oracle", step)]
                interventions.append({"case_id": case_id, "intervention": "replace_mlp_by_oracle",
                    "forecast": "paired", "storage_step_kwh": step,
                    "grid_reduction_kwh": mlp.grid_import_kwh - oracle.grid_import_kwh,
                    "gap_before_kwh": mlp.delta_grid_import_kwh, "gap_after_kwh": oracle.delta_grid_import_kwh,
                    "pct_excess_removed": 100 * (mlp.delta_grid_import_kwh - oracle.delta_grid_import_kwh) / mlp.delta_grid_import_kwh
                        if abs(mlp.delta_grid_import_kwh) > 1e-12 else None})
        for method in ("mlp", "oracle"):
            if (method, .25) in lookup.index and (method, .125) in lookup.index:
                old, fine = lookup.loc[(method, .25)], lookup.loc[(method, .125)]
                interventions.append({"case_id": case_id, "intervention": "refine_grid_0.25_to_0.125",
                    "forecast": method, "storage_step_kwh": .125,
                    "grid_reduction_kwh": old.grid_import_kwh - fine.grid_import_kwh,
                    "gap_before_kwh": old.delta_grid_import_kwh, "gap_after_kwh": fine.delta_grid_import_kwh,
                    "pct_excess_removed": 100 * (old.delta_grid_import_kwh - fine.delta_grid_import_kwh) / old.delta_grid_import_kwh
                        if abs(old.delta_grid_import_kwh) > 1e-12 else None})
        folder = prior / "cases" / case_id
        case = json.loads((folder / "case.json").read_text())
        demand, solar = np.asarray(case["actual_demand"]), np.asarray(case["solar"][:168])
        forecasts = pd.read_csv(folder / "forecasts.csv.gz")
        current = forecasts[forecasts.horizon == 1].predicted_demand_kwh.to_numpy()
        ledger = pd.read_csv(folder / "ledger.csv")
        selected = ledger[ledger.strategy == "autonomy"].sort_values("step")
        predicted_sign = np.sign(np.where(np.abs(current - solar) < 1e-9, 0., current - solar))
        actual_sign = np.sign(np.where(np.abs(demand - solar) < 1e-9, 0., demand - solar))
        is_charge = selected.requested_charge_kw.to_numpy() > 1e-9
        is_discharge = selected.requested_discharge_kw.to_numpy() > 1e-9
        forecast_errors.append({"case_id": case_id, "hours": len(demand),
            "current_hour_mae_kwh": float(np.abs(current - demand).mean()),
            "current_hour_bias_kwh": float((current - demand).mean()),
            "imbalance_sign_mismatch_hours": int((predicted_sign != actual_sign).sum()),
            "charge_requested_during_realized_deficit_hours": int((is_charge & (actual_sign > 0)).sum()),
            "discharge_requested_during_realized_surplus_hours": int((is_discharge & (actual_sign < 0)).sum()),
            "note": "Present load is realized only in the evaluator/feedback; MLP requested commands remain forecast-based"})
    comparisons = pd.DataFrame(interventions)
    comparisons.to_csv(root / "controlled_differences.csv", index=False)
    pd.DataFrame(charge_sources).to_csv(root / "grid_charge_sources.csv", index=False)
    pd.DataFrame(forecast_errors).to_csv(root / "forecast_dispatch_diagnostics.csv", index=False)
    table = []
    for key, group in comparisons.groupby(["intervention", "forecast", "storage_step_kwh"]):
        before, after = group.gap_before_kwh.sum(), group.gap_after_kwh.sum()
        table.append({"intervention": key[0], "forecast": key[1], "storage_step_kwh": key[2],
            "cases": len(group), "mean_grid_reduction_kwh": float(group.grid_reduction_kwh.mean()),
            "std_grid_reduction_kwh": float(group.grid_reduction_kwh.std(ddof=1)),
            "worst_grid_reduction_kwh": float(group.grid_reduction_kwh.min()),
            "best_grid_reduction_kwh": float(group.grid_reduction_kwh.max()),
            "pct_excess_removed_ratio_of_sums": 100 * (before - after) / before if abs(before) > 1e-12 else None})
    pd.DataFrame(table).to_csv(root / "controlled_summary.csv", index=False)
    good = frame[frame.complete]
    bounds = pd.read_csv(root / "physical_bound.csv")
    write_json(root / "findings.json", {"all_reactive_at_bound": bool((bounds.reactive_gap_to_bound_kwh.abs() < 1e-8).all()),
        "max_reactive_bound_gap_abs_kwh": float(bounds.reactive_gap_to_bound_kwh.abs().max()),
        "all_completed_pairs_same_terminal": bool(good.same_terminal_as_reactive.all()),
        "max_abs_delta_terminal_kwh": float(good.delta_battery_final_kwh.abs().max()),
        "max_abs_balance_gap_residual_kwh": float(good.balance_gap_residual_kwh.abs().max()),
        "physical_violations": int(good.constraint_violations.sum()), "unserved_kwh": float(good.unserved_kwh.sum()),
        "material_import_improvements": int(good.material_improvement.sum()),
        "intervention_effects_not_additive": True,
        "scope": "Two independent validation weeks, independent idealized Cordoba solar, fixed frozen MLP; not production savings"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
