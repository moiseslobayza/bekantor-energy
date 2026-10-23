"""Compare only matched supplemental cases; never pool them with the factorial."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json


def analyze(reference, supplemental, output):
    if output.exists():
        raise ValueError("New grid comparison directory required")
    receipts = {}
    for root in (reference, supplemental):
        receipt = json.loads((root / "verification.json").read_text())
        assert receipt["status"] == "passed"
        assert all(sha256(Path(p)) == h for p, h in receipt["artifact_sha256"].items())
        receipts[str(root)] = sha256(root / "verification.json")
    original = json.loads((reference / "design.json").read_text())["config"]
    fine = json.loads((supplemental / "design.json").read_text())["config"]
    for key in ("resilience", "valuation_sensitivity"):
        assert original[key] == fine[key]
    assert original["planner"]["storage_step_kwh"] == .25
    assert fine["planner"]["storage_step_kwh"] == .125
    assert {k: v for k, v in original["planner"].items() if k != "storage_step_kwh"} == {
        k: v for k, v in fine["planner"].items() if k != "storage_step_kwh"}
    coarse = pd.read_csv(reference / "paired_metrics.csv")
    refined = pd.read_csv(supplemental / "paired_metrics.csv")
    keys = ["case_id", "event_id", "policy"]
    merged = refined.merge(coarse, on=keys, suffixes=("_fine", "_coarse"), validate="one_to_one")
    assert len(merged) == len(refined)
    fields = ("critical_ens_kwh", "grid_import_kwh", "adjusted_cost_sim", "gross_cost_sim",
              "outage_initial_kwh", "final_kwh", "expanded_nodes", "planning_seconds")
    for field in fields:
        merged["fine_minus_coarse_" + field] = merged[field + "_fine"] - merged[field + "_coarse"]
    for field in ("noncritical_shed_kwh", "initial_kwh", "critical_outage_required_kwh"):
        np.testing.assert_allclose(merged[field + "_fine"], merged[field + "_coarse"], rtol=0, atol=1e-9)
    for policy in ("reactive", "fixed_reserve", "adaptive_mlp"):
        same = merged[merged.policy == policy]
        np.testing.assert_allclose(same.fine_minus_coarse_critical_ens_kwh, 0, rtol=0, atol=1e-9)
        np.testing.assert_allclose(same.fine_minus_coarse_grid_import_kwh, 0, rtol=0, atol=1e-9)
    rows = []
    for scope, selection in (("outage_panel", merged[merged.duration_hours_fine > 0]),
                             ("no_outage_panel", merged[merged.duration_hours_fine == 0])):
        for policy, group in selection.groupby("policy"):
            delta = group.fine_minus_coarse_critical_ens_kwh
            row = {"scope": scope, "policy": policy, "cases": len(group),
                   "fine_ens_improved_cases": int((delta < -1e-6).sum()),
                   "fine_ens_worsened_cases": int((delta > 1e-6).sum()),
                   "fine_ens_tied_cases": int((delta.abs() <= 1e-6).sum()),
                   "fine_ens_worst_increase_kwh": float(delta.max())}
            for field in fields:
                for suffix in ("coarse", "fine"):
                    row[field + "_" + suffix + "_mean"] = group[field + "_" + suffix].mean()
                row["fine_minus_coarse_" + field + "_mean"] = group["fine_minus_coarse_" + field].mean()
            rows.append(row)
    output.mkdir(parents=True)
    merged.to_csv(output / "matched_grid_cases.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "grid_summary.csv", index=False)
    write_json(output / "verification.json", {"status": "passed", "parent_receipt_sha256": receipts,
        "matched_replays": len(merged), "complete_outage_cases": int(merged[merged.duration_hours_fine > 0].pair_id_fine.nunique()),
        "supplemental_coverage": json.loads((supplemental / "coverage.json").read_text()),
        "interpretation": "Matched 22:00/6h panel only; no changes to primary .25 kWh results",
        "source_sha256": sha256(Path(__file__)),
        "artifact_sha256": {str(p): sha256(p) for p in output.glob("*.csv")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.reference, args.supplemental, args.output)
