"""Same-forecast search attribution and no-outage preparation accounting."""
import argparse
from pathlib import Path

import pandas as pd


def analyze(root):
    frame = pd.read_csv(root / "paired_metrics.csv")
    candidate = frame[frame.policy == "astar_mlp"].set_index("pair_id")
    reference = frame[frame.policy == "adaptive_mlp"].set_index("pair_id")
    rows = []
    for pair_id in candidate.index.intersection(reference.index):
        a, b = candidate.loc[pair_id], reference.loc[pair_id]
        rows.append({"pair_id": pair_id, "case_id": a.case_id, "event_id": a.event_id,
            "duration_hours": a.duration_hours, "tariff_mode": a.tariff_mode,
            "avoided_critical_kwh_same_mlp": b.critical_ens_kwh - a.critical_ens_kwh,
            "reference_ens_kwh": b.critical_ens_kwh,
            "delta_adjusted_cost_sim": a.adjusted_cost_sim - b.adjusted_cost_sim,
            "delta_grid_import_kwh": a.grid_import_kwh - b.grid_import_kwh})
    result = pd.DataFrame(rows)
    result.to_csv(root / "same_forecast_search.csv", index=False)
    summaries = []
    for label, group in (("outage_all", result[result.duration_hours > 0]), ("no_outage", result[result.duration_hours == 0])):
        total = group.reference_ens_kwh.sum()
        summaries.append({"scope": label, "cases": len(group),
            "mean_critical_kwh_avoided": float(group.avoided_critical_kwh_same_mlp.mean()),
            "aggregate_ens_reduction_pct": 100 * group.avoided_critical_kwh_same_mlp.sum() / total if total > 1e-6 else None,
            "improved_cases": int((group.avoided_critical_kwh_same_mlp > 1e-6).sum()),
            "worsened_cases": int((group.avoided_critical_kwh_same_mlp < -1e-6).sum()),
            "delta_adjusted_cost_sim_mean": float(group.delta_adjusted_cost_sim.mean()),
            "delta_grid_import_kwh_mean": float(group.delta_grid_import_kwh.mean())})
    pd.DataFrame(summaries).to_csv(root / "same_forecast_search_summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
