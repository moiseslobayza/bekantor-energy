"""Reconcile grid changes when delivered service and final stock differ."""
import argparse
import json
from pathlib import Path

import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json


def analyze(reference, output):
    if output.exists(): raise ValueError("New accounting directory required")
    receipt = json.loads((reference / "verification.json").read_text())
    assert receipt["status"] == "passed"
    frame = pd.read_csv(reference / "paired_metrics.csv")
    pairs = pd.read_csv(reference / "paired_differences.csv")
    rows = []
    for pair in pairs.itertuples():
        a = frame[(frame.pair_id == pair.pair_id) & (frame.policy == pair.candidate)].iloc[0]
        b = frame[(frame.pair_id == pair.pair_id) & (frame.policy == pair.reference)].iloc[0]
        delta_grid = a.grid_import_kwh - b.grid_import_kwh
        avoided = b.critical_ens_kwh - a.critical_ens_kwh
        delta_shed = a.noncritical_shed_kwh - b.noncritical_shed_kwh
        delta_curtail = a.curtailed_kwh - b.curtailed_kwh
        delta_loss = a.loss_kwh - b.loss_kwh
        delta_final = a.final_kwh - b.final_kwh
        delta_initial = a.initial_kwh - b.initial_kwh
        residual = delta_grid - (avoided - delta_shed + delta_curtail + delta_loss + delta_final - delta_initial)
        assert abs(residual) < 1e-8
        rows.append({"pair_id": pair.pair_id, "comparison": pair.comparison, "duration_hours": pair.duration_hours,
            "delta_grid_kwh": delta_grid, "critical_service_recovered_kwh": avoided,
            "delta_noncritical_shed_kwh": delta_shed, "delta_curtail_kwh": delta_curtail,
            "delta_loss_kwh": delta_loss, "delta_final_kwh": delta_final, "residual_kwh": residual})
    output.mkdir(parents=True)
    result = pd.DataFrame(rows)
    result.to_csv(output / "energy_balance.csv", index=False)
    result[result.duration_hours > 0].groupby("comparison").mean(numeric_only=True).to_csv(output / "mean_energy_balance.csv")
    write_json(output / "verification.json", {"status": "passed", "pairs_checked": len(rows),
        "max_abs_residual_kwh": float(result.residual_kwh.abs().max()),
        "parent_receipt_sha256": sha256(reference / "verification.json"), "source_sha256": sha256(Path(__file__)),
        "identity": "deltaG = critical_service_recovered - delta_noncritical_shed + delta_curtail + delta_loss + delta_final - delta_initial"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.reference, args.output)
