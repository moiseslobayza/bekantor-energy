"""Derived paired diagnostics; no policy changes, neural fitting or test access."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import write_json


def analyze(run_dir):
    root = Path(run_dir)
    paired = pd.read_csv(root / "paired_results.csv")
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))
    valuation = design["configuration"]["valuation_sim_per_stored_kwh"]
    daily, terminal_diagnostics = [], []
    for case_id, group in paired.groupby("case_id"):
        ledger = pd.read_csv(root / "cases" / case_id / "ledger.csv", parse_dates=["timestamp"])
        ledger["date"] = ledger.timestamp.dt.strftime("%Y-%m-%d")
        searches = pd.read_csv(root / "cases" / case_id / "searches.csv")
        original = ledger[ledger.strategy == "mlp_mse"].reset_index(drop=True)
        terminal = ledger[ledger.strategy == "mlp_mse_terminal"].reset_index(drop=True)
        requested = np.abs(original[["requested_charge_kw", "requested_discharge_kw"]].to_numpy()
                           - terminal[["requested_charge_kw", "requested_discharge_kw"]].to_numpy())
        flows = np.abs(original[["charge_kwh", "discharge_kwh"]].to_numpy()
                       - terminal[["charge_kwh", "discharge_kwh"]].to_numpy())
        required = design["configuration"]["terminal_min_soc"] * design["home_reference"]["battery_capacity_kwh"]
        terminal_diagnostics.append({"case_id": case_id,
            "hours_with_different_requested_actions": int((requested > 1e-9).any(axis=1).sum()),
            "hours_with_different_realized_battery_flows": int((flows > 1e-9).any(axis=1).sum()),
            "original_forecast_endings_below_terminal_target": int((searches.loc[searches.strategy == "mlp_mse", "predicted_terminal_kwh"] < required - 1e-9).sum()),
            "variant_min_predicted_terminal_kwh": float(searches.loc[searches.strategy == "mlp_mse_terminal", "predicted_terminal_kwh"].min()),
            "variant_realized_hours_below_terminal_target": int((terminal.stored_kwh < required - 1e-9).sum()),
            "original_planning_seconds": float(searches.loc[searches.strategy == "mlp_mse", "seconds"].sum()),
            "variant_planning_seconds": float(searches.loc[searches.strategy == "mlp_mse_terminal", "seconds"].sum())})
        for (name, date), rows in ledger.groupby(["strategy", "date"]):
            daily.append({"case_id": case_id, "strategy": name, "date": date,
                "initial_stored_kwh": rows.initial_kwh.iloc[0], "final_stored_kwh": rows.stored_kwh.iloc[-1],
                "gross_cost_sim": rows.cost.sum(), "adjusted_cost_sim": rows.cost.sum() + valuation * (rows.initial_kwh.iloc[0] - rows.stored_kwh.iloc[-1]),
                "grid_import_kwh": rows.grid_import_kwh.sum(), "solar_kwh": rows.solar_kwh.sum()})
        for name, scores in group.set_index("strategy").iterrows():
            rows = [r for r in daily if r["case_id"] == case_id and r["strategy"] == name]
            np.testing.assert_allclose(sum(r["adjusted_cost_sim"] for r in rows), scores.adjusted_cost_sim, atol=1e-10)
    pd.DataFrame(daily).to_csv(root / "daily_metrics.csv", index=False)
    pd.DataFrame(terminal_diagnostics).to_csv(root / "terminal_diagnostics.csv", index=False)
    summaries = []
    for tariff, group in [("all", paired), *paired.groupby("tariff_mode")]:
        reactive = group[group.strategy == "reactive"]
        mse = group[group.strategy == "mlp_mse"].set_index("case_id")
        for name, rows in group.groupby("strategy"):
            row = {"tariff_mode": tariff, "strategy": name, "cases": len(rows)}
            for field in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
                total, baseline = rows[field].sum(), reactive[field].sum()
                row[f"{field}_mean"] = float(rows[field].mean())
                row[f"{field}_std"] = float(rows[field].std())
                row[f"improvement_pct_{field}_ratio_of_sums"] = float(100 * (baseline - total) / baseline) if baseline else None
            row.update({"battery_initial_mean_kwh": float(rows.battery_initial_kwh.mean()),
                "battery_final_mean_kwh": float(rows.battery_final_kwh.mean()),
                "solar_utilization_mean_pct": float(rows.solar_utilization_pct.mean()),
                "gross_cost_wins": int((rows.delta_import_cost_sim_vs_reactive < -1e-9).sum()),
                "adjusted_cost_wins": int((rows.delta_adjusted_cost_sim_vs_reactive < -1e-9).sum()),
                "adjusted_cost_losses": int((rows.delta_adjusted_cost_sim_vs_reactive > 1e-9).sum()),
                "gross_to_adjusted_sign_reversals": int(((rows.delta_import_cost_sim_vs_reactive * rows.delta_adjusted_cost_sim_vs_reactive) < -1e-12).sum()),
                "cost_down_grid_up_cases": int(((rows.delta_adjusted_cost_sim_vs_reactive < -1e-9) & (rows.delta_grid_import_kwh_vs_reactive > 1e-9)).sum())})
            if name == "mlp_mse_terminal":
                row["delta_adjusted_cost_vs_original_mean_sim"] = float((rows.set_index("case_id").adjusted_cost_sim - mse.adjusted_cost_sim).mean())
                row["wins_vs_original_adjusted"] = int((rows.delta_adjusted_cost_sim_vs_original_mse < -1e-9).sum())
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "economic_summary.csv", index=False)
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    sensitivity.groupby(["valuation_sim_per_stored_kwh", "tariff_mode", "strategy"]).agg(
        cases=("case_id", "size"), adjusted_cost_mean_sim=("adjusted_cost_sim", "mean"),
        delta_vs_reactive_mean_sim=("delta_vs_reactive_sim", "mean"),
        delta_vs_reactive_min_sim=("delta_vs_reactive_sim", "min"),
        delta_vs_reactive_max_sim=("delta_vs_reactive_sim", "max")).reset_index().to_csv(root / "sensitivity_summary.csv", index=False)
    break_even = []
    for row in paired[paired.strategy != "reactive"].itertuples():
        reference = paired[(paired.case_id == row.case_id) & (paired.strategy == "reactive")].iloc[0]
        delta_final = row.battery_final_kwh - reference.battery_final_kwh
        break_even.append({"case_id": row.case_id, "strategy": row.strategy,
            "delta_gross_sim": row.delta_import_cost_sim_vs_reactive, "delta_final_stored_kwh": delta_final,
            "break_even_valuation_sim_per_stored_kwh": row.delta_import_cost_sim_vs_reactive / delta_final if abs(delta_final) > 1e-9 else np.nan,
            "interpretation": "analytical crossing of accounting deltas; negative value outside nonnegative valuation domain possible"})
    pd.DataFrame(break_even).to_csv(root / "valuation_break_even.csv", index=False)
    planning = pd.concat([pd.read_csv(root / "cases" / c / "searches.csv") for c in paired.case_id.unique()])
    findings = {"valuation_sim_per_stored_kwh": valuation,
        "total_realized_steps": int(paired.hours_completed.sum()), "full_plans": len(planning),
        "planned_transitions": len(planning) * 24, "max_search_seconds": float(planning.seconds.max()),
        "max_expanded_nodes_per_search": int(planning.expanded_nodes.max()),
        "zero_physical_violations": bool(paired.constraint_violations.eq(0).all()),
        "total_unserved_kwh": float(paired.unserved_kwh.sum()),
        "summary": [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in summary.to_dict("records")]}
    write_json(root / "findings.json", findings)
    print(json.dumps(findings, indent=2))
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
