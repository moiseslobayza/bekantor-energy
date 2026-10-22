"""Descriptive mode comparisons; no new policy weights, fitting or test access."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import write_json
from bekantor_energy.energy.mode_experiment import MODE_STRATEGIES


def analyze(run_dir):
    root = Path(run_dir)
    paired = pd.read_csv(root / "paired_results.csv")
    coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    if paired.empty:
        write_json(root / "findings.json", {"status": "no_complete_pairs", "coverage": coverage})
        return
    design = json.loads((root / "design.json").read_text(encoding="utf-8"))
    valuation = design["previous_design"]["configuration"]["valuation_sim_per_stored_kwh"]
    economic = paired[paired.strategy == "economic"].set_index("case_id")
    paired["avoided_grid_vs_economic_kwh"] = paired.case_id.map(economic.grid_import_kwh) - paired.grid_import_kwh
    paired["additional_adjusted_cost_vs_economic_sim"] = paired.adjusted_cost_sim - paired.case_id.map(economic.adjusted_cost_sim)
    avoided = paired.avoided_grid_vs_economic_kwh
    paired["additional_sim_per_avoided_grid_kwh"] = paired.additional_adjusted_cost_vs_economic_sim / avoided.where(avoided > 1e-9)
    paired.to_csv(root / "mode_tradeoffs.csv", index=False)
    reference = paired[paired.strategy == "reactive"].set_index("case_id")
    decomposition = paired[["case_id", "block_id", "solar_profile", "tariff_mode", "strategy"]].copy()
    for field in ("grid_import_kwh", "curtailed_kwh", "conversion_loss_kwh", "battery_initial_kwh", "battery_final_kwh"):
        decomposition[f"delta_{field}"] = paired[field] - paired.case_id.map(reference[field])
    decomposition["balance_residual_kwh"] = (decomposition.delta_grid_import_kwh - decomposition.delta_curtailed_kwh
        - decomposition.delta_conversion_loss_kwh - decomposition.delta_battery_final_kwh + decomposition.delta_battery_initial_kwh)
    # Same realized demand/solar, no export or outages: remaining grid gap is
    # exactly extra curtailment + conversion losses + terminal stock change.
    np.testing.assert_allclose(decomposition.balance_residual_kwh, 0., atol=1e-8)
    decomposition.to_csv(root / "grid_gap_decomposition.csv", index=False)
    summaries = []
    fields = ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh", "solar_utilization_pct", "curtailed_kwh",
              "min_soc", "mean_soc", "final_soc", "battery_initial_kwh", "battery_final_kwh", "unserved_kwh",
              "constraint_violations", "planning_seconds", "expanded_nodes")
    for tariff, group in [("all", paired), *paired.groupby("tariff_mode")]:
        reference = group[group.strategy == "reactive"]
        econ = group[group.strategy == "economic"]
        for name, rows in group.groupby("strategy"):
            record = {"tariff_mode": tariff, "strategy": name, "cases": len(rows)}
            for field in fields:
                record[f"{field}_mean"] = float(rows[field].mean())
                record[f"{field}_std"] = float(rows[field].std())
            for field in ("import_cost_sim", "adjusted_cost_sim", "grid_import_kwh"):
                for label, base in (("reactive", reference), ("economic", econ)):
                    total = base[field].sum()
                    record[f"reduction_pct_{field}_vs_{label}"] = float(100 * (total - rows[field].sum()) / total) if abs(total) > 1e-9 else None
            for field in ("adjusted_cost_sim", "grid_import_kwh"):
                delta = rows[f"delta_{field}_vs_reactive"]
                record[f"wins_{field}_vs_reactive"] = int((delta < -1e-9).sum())
                record[f"losses_{field}_vs_reactive"] = int((delta > 1e-9).sum())
                record[f"ties_{field}_vs_reactive"] = int((delta.abs() <= 1e-9).sum())
            record["cost_down_grid_up_cases"] = int(((rows.delta_adjusted_cost_sim_vs_reactive < -1e-9) & (rows.delta_grid_import_kwh_vs_reactive > 1e-9)).sum())
            summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "mode_summary.csv", index=False)
    daily, observed_frontier = [], []
    for case_id, group in paired.groupby("case_id"):
        ledger = pd.read_csv(root / "cases" / case_id / "ledger.csv", parse_dates=["timestamp"])
        ledger["date"] = ledger.timestamp.dt.strftime("%Y-%m-%d")
        for (name, day), rows in ledger.groupby(["strategy", "date"]):
            daily.append({"case_id": case_id, "strategy": name, "date": day,
                "initial_stored_kwh": rows.initial_kwh.iloc[0], "final_stored_kwh": rows.stored_kwh.iloc[-1],
                "gross_cost_sim": rows.cost.sum(), "adjusted_cost_sim": rows.cost.sum() + valuation * (rows.initial_kwh.iloc[0] - rows.stored_kwh.iloc[-1]),
                "grid_import_kwh": rows.grid_import_kwh.sum(), "mean_soc": float(((rows.initial_kwh + rows.stored_kwh) / (2 * design["previous_design"]["home_reference"]["battery_capacity_kwh"])).mean())})
        for name, row in group.set_index("strategy").iterrows():
            dominated = group[(group.adjusted_cost_sim <= row.adjusted_cost_sim + 1e-9) &
                (group.grid_import_kwh <= row.grid_import_kwh + 1e-9) &
                ((group.adjusted_cost_sim < row.adjusted_cost_sim - 1e-9) | (group.grid_import_kwh < row.grid_import_kwh - 1e-9))]
            observed_frontier.append({"case_id": case_id, "strategy": name, "dominated_by_observed_policies": ",".join(dominated.strategy),
                "nondominated_among_four": dominated.empty, "scope": "four observed realized options, not a complete Pareto frontier"})
            day_values = [v for v in daily if v["case_id"] == case_id and v["strategy"] == name]
            np.testing.assert_allclose(sum(v["adjusted_cost_sim"] for v in day_values), row.adjusted_cost_sim, atol=1e-9)
    pd.DataFrame(daily).to_csv(root / "daily_metrics.csv", index=False)
    pd.DataFrame(observed_frontier).to_csv(root / "observed_dominance.csv", index=False)
    sensitivity = pd.read_csv(root / "sensitivity.csv")
    sensitivity.groupby(["valuation_sim_per_stored_kwh", "tariff_mode", "strategy"]).agg(
        cases=("case_id", "size"), adjusted_cost_mean_sim=("adjusted_cost_sim", "mean"),
        delta_vs_reactive_mean_sim=("delta_vs_reactive_sim", "mean")).reset_index().to_csv(root / "sensitivity_summary.csv", index=False)
    searches = pd.concat([pd.read_csv(root / "cases" / c / "searches.csv") for c in paired.case_id.unique()])
    findings = {"coverage": coverage, "hours_realized": int(paired.hours_completed.sum()), "full_plans": len(searches),
        "planned_transitions": 24 * len(searches), "max_search_seconds": float(searches.seconds.max()),
        "max_expanded_nodes_per_search": int(searches.expanded_nodes.max()),
        "physical_violations": int(paired.constraint_violations.sum()), "unserved_kwh": float(paired.unserved_kwh.sum()),
        "valuation_sim_per_stored_kwh": valuation, "fixed_objectives": design["objectives"],
        "summary": [{k: None if pd.isna(v) else v for k, v in row.items()} for row in summary.to_dict("records")]}
    write_json(root / "findings.json", findings)
    columns = ["tariff_mode", "strategy", "adjusted_cost_sim_mean", "grid_import_kwh_mean",
               "reduction_pct_adjusted_cost_sim_vs_reactive", "reduction_pct_grid_import_kwh_vs_reactive",
               "wins_grid_import_kwh_vs_reactive", "losses_grid_import_kwh_vs_reactive"]
    print(summary[columns].to_string(index=False))
    print(json.dumps({k: v for k, v in findings.items() if k not in ("summary", "fixed_objectives")}, indent=2))
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
