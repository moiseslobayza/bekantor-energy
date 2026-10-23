"""Paired service/cost analysis, including failures and adverse cases."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import write_json
from bekantor_energy.energy.resilience import POLICIES

COMPARISONS = (("A_reserve", "fixed_reserve", "reactive"),
    ("B_adaptive", "adaptive_mlp", "fixed_reserve"),
    ("C_search", "astar_persistence", "adaptive_mlp"),
    ("D_neural", "astar_mlp", "astar_persistence"),
    ("E_primary", "astar_mlp", "fixed_reserve"),
    *(("F_oracle_vs_" + p, "astar_oracle", p) for p in POLICIES if p != "astar_oracle"))


def analyze(root):
    root = Path(root)
    design = json.loads((root / "design.json").read_text())
    metrics = pd.read_csv(root / "metrics.csv")
    metrics["pair_id"] = metrics.case_id + "__" + metrics.event_id
    status = metrics.pivot(index="pair_id", columns="policy", values="status").reindex(columns=POLICIES)
    good_ids = status.index[(status == "completed").all(axis=1)]
    paired = metrics[metrics.pair_id.isin(good_ids)].copy()
    paired.to_csv(root / "paired_metrics.csv", index=False)
    write_json(root / "coverage.json", {"expected_replays": design["expected_replays"], "recorded_replays": len(metrics),
        "expected_cases": design["expected_replays"] // len(POLICIES), "complete_six_way_cases": len(good_ids),
        "failed_replays": int((metrics.status != "completed").sum()),
        "incomplete_pair_ids": list(status.index[~(status == "completed").all(axis=1)]),
        "aggregation": "Only complete six-policy pairs; partial/failed replays remain in metrics and ledgers"})
    if paired.empty:
        return
    summaries = []
    cuts = paired[paired.duration_hours > 0]
    for scope, selection, dimension in (("outage_all", cuts, None), ("no_outage", paired[paired.duration_hours == 0], None),
            *(("by_" + d, cuts, d) for d in ("duration_hours", "onset_step", "solar_profile", "tariff_mode", "block_id"))):
        keys = ["policy"] + ([dimension] if dimension else [])
        for group_key, group in selection.groupby(keys):
            if not isinstance(group_key, tuple): group_key = (group_key,)
            result = {"scope": scope, "policy": group_key[0], "dimension": dimension,
                      "value": str(group_key[1]) if dimension else None, "cases": len(group)}
            for field in ("critical_ens_kwh", "critical_ens_fraction", "critical_full_hours", "initial_autonomy_hours",
                          "noncritical_shed_kwh", "grid_import_kwh", "gross_cost_sim", "adjusted_cost_sim",
                          "outage_initial_kwh", "outage_final_kwh", "outage_min_soc", "final_kwh", "mean_soc",
                          "loss_kwh", "solar_used_kwh", "solar_utilization_pct", "grid_charge_kwh",
                          "planning_seconds", "expanded_nodes", "search_seconds_unique", "expanded_nodes_unique"):
                values = group[field].dropna()
                result[field + "_mean"] = float(values.mean()) if len(values) else None
                result[field + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
                result[field + "_worst_max"] = float(values.max()) if len(values) else None
            result["critical_ens_kwh_sum"] = float(group.critical_ens_kwh.sum())
            result["critical_service_complete_cases"] = int(group.no_critical_deficit.fillna(False).astype(bool).sum())
            result["complete_service_pct"] = 100 * result["critical_service_complete_cases"] / len(group) if scope != "no_outage" else None
            result["autonomy_censored_cases"] = int(group.autonomy_right_censored.fillna(False).astype(bool).sum())
            summaries.append(result)
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    differences = []
    fields = ("critical_ens_kwh", "critical_full_hours", "initial_autonomy_hours", "grid_import_kwh",
              "gross_cost_sim", "adjusted_cost_sim", "outage_initial_kwh", "final_kwh")
    for label, candidate, reference in COMPARISONS:
        left = paired[paired.policy == candidate].set_index("pair_id")
        right = paired[paired.policy == reference].set_index("pair_id")
        for pair_id in left.index.intersection(right.index):
            a, b = left.loc[pair_id], right.loc[pair_id]
            row = {k: a[k] for k in ("case_id", "event_id", "duration_hours", "onset_step", "solar_profile", "tariff_mode", "block_id")}
            row.update(comparison=label, candidate=candidate, reference=reference, pair_id=pair_id,
                       critical_ens_reference_kwh=b.critical_ens_kwh, critical_ens_candidate_kwh=a.critical_ens_kwh,
                       avoided_critical_kwh=b.critical_ens_kwh - a.critical_ens_kwh,
                       avoided_critical_pct=100 * (b.critical_ens_kwh - a.critical_ens_kwh) / b.critical_ens_kwh if b.critical_ens_kwh > 1e-6 else None)
            for field in fields:
                row["delta_" + field] = a[field] - b[field]
            avoided = row["avoided_critical_kwh"]
            row["incremental_adjusted_sim_per_critical_kwh_recovered"] = row["delta_adjusted_cost_sim"] / avoided if avoided > 1e-6 else None
            differences.append(row)
    diff = pd.DataFrame(differences)
    diff.to_csv(root / "paired_differences.csv", index=False)
    aggregates = []
    for scope, group_frame in (("outage_all", diff[diff.duration_hours > 0]), ("no_outage", diff[diff.duration_hours == 0])):
        for comparison, group in group_frame.groupby("comparison"):
            reference_sum = group.critical_ens_reference_kwh.sum()
            result = {"scope": scope, "comparison": comparison, "candidate": group.candidate.iloc[0],
                      "reference": group.reference.iloc[0], "cases": len(group),
                      "avoided_critical_mean_kwh": float(group.avoided_critical_kwh.mean()),
                      "avoided_critical_std_kwh": float(group.avoided_critical_kwh.std(ddof=1)),
                      "avoided_critical_median_kwh": float(group.avoided_critical_kwh.median()),
                      "avoided_critical_worst_kwh": float(group.avoided_critical_kwh.min()),
                      "aggregate_reduction_pct": float(100 * group.avoided_critical_kwh.sum() / reference_sum) if reference_sum > 1e-6 else None,
                      "improved_cases": int((group.avoided_critical_kwh > 1e-6).sum()),
                      "worsened_cases": int((group.avoided_critical_kwh < -1e-6).sum()),
                      "tied_cases": int((group.avoided_critical_kwh.abs() <= 1e-6).sum())}
            for field in ("grid_import_kwh", "gross_cost_sim", "adjusted_cost_sim", "initial_autonomy_hours", "outage_initial_kwh", "final_kwh"):
                values = group["delta_" + field].dropna()
                result["delta_" + field + "_mean"] = float(values.mean()) if len(values) else None
            aggregates.append(result)
    aggregate = pd.DataFrame(aggregates)
    aggregate.to_csv(root / "comparison_summary.csv", index=False)
    diff[diff.duration_hours.gt(0) & diff.avoided_critical_kwh.lt(-1e-6)].to_csv(root / "adverse_cases.csv", index=False)
    sensitivity = []
    for value in design["config"]["valuation_sensitivity"]:
        for row in paired.to_dict("records"):
            sensitivity.append({"pair_id": row["pair_id"], "policy": row["policy"], "stored_kwh_value_sim": value,
                "adjusted_cost_sim": row["gross_cost_sim"] + value * (row["initial_kwh"] - row["final_kwh"])})
    pd.DataFrame(sensitivity).to_csv(root / "terminal_sensitivity.csv", index=False)
    periods = []
    for row in paired.to_dict("records"):
        ledger = pd.read_csv(Path(row["output_dir"]) / "ledger.csv")
        if row["duration_hours"]:
            start, end = int(row["onset_step"]), int(row["onset_step"] + row["duration_hours"])
            masks = {"preparation": ledger.step < start, "outage": (ledger.step >= start) & (ledger.step < end), "recovery": ledger.step >= end}
        else:
            masks = {"whole_no_outage": np.ones(len(ledger), bool)}
        for label, mask in masks.items():
            part = ledger[mask]
            periods.append({"pair_id": row["pair_id"], "policy": row["policy"], "period": label,
                "hours": len(part), "gross_cost_sim": float(part.cost.sum()), "grid_import_kwh": float(part.grid_import_kwh.sum()),
                "critical_ens_kwh": float(part.critical_ens_kwh.sum()), "noncritical_shed_kwh": float(part.noncritical_shed_kwh.sum())})
    pd.DataFrame(periods).to_csv(root / "period_metrics.csv", index=False)
    primary = aggregate[(aggregate.scope == "outage_all") & (aggregate.comparison == "E_primary")]
    complete = len(good_ids) == design["expected_replays"] // len(POLICIES)
    if len(primary):
        result = primary.iloc[0]
        material = (complete and result.avoided_critical_mean_kwh >= design["config"]["material_mean_avoided_critical_kwh"]
                    and pd.notna(result.aggregate_reduction_pct) and result.aggregate_reduction_pct >= design["config"]["material_aggregate_ens_reduction_pct"])
        write_json(root / "findings.json", {"complete_factorial": bool(complete and design["stage"] == "full"),
            "threshold_met_in_evaluated_stage": bool(material), "primary_mean_avoided_kwh": float(result.avoided_critical_mean_kwh),
            "primary_aggregate_reduction_pct": float(result.aggregate_reduction_pct) if pd.notna(result.aggregate_reduction_pct) else None,
            "primary_worsened_cases": int(result.worsened_cases),
            "critical_fraction_is_synthetic": True, "solar_forecast_is_idealized": True,
            "no_test_or_training": True, "risk_not_equal_realized_ens": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
