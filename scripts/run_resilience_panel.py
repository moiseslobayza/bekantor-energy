"""Predeclared supplemental panel; never overwrites the primary configuration."""
import argparse
import json
from pathlib import Path

from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.energy.resilience_experiment import run_experiment
from analyze_resilience import analyze
from analyze_resilience_incremental import analyze as incremental
from verify_resilience import verify


def run(reference, destination, variant, data_dir):
    reference, destination = Path(reference), Path(destination)
    if destination.exists(): raise ValueError("New supplemental directory required")
    receipt = json.loads((reference / "verification.json").read_text())
    assert receipt["status"] == "passed"
    historical = {**receipt["artifact_sha256"], str(reference / "verification.json"): sha256(reference / "verification.json")}
    assert all(sha256(Path(p)) == h for p, h in historical.items())
    config = json.loads((reference / "design.json").read_text())["config"]
    config["previous_run"] = str(reference)
    if variant == "fine_grid": config["planner"]["storage_step_kwh"] = .125
    elif variant == "critical_020": config["resilience"]["critical_fraction"] = .2
    elif variant == "critical_060": config["resilience"]["critical_fraction"] = .6
    else: raise ValueError("Only predeclared sensitivities accepted")
    config_path = destination.parent / (destination.name + ".config.json")
    write_json(config_path, config)
    run_experiment(config_path, destination, "panel", data_dir, reference)
    analyze(destination)
    incremental(destination)
    verify(destination, data_dir, reference)
    assert all(sha256(Path(p)) == h for p, h in historical.items())
    write_json(destination / "parent_preservation.json", {"verified": True, "files": len(historical),
        "parent_receipt_sha256": historical[str(reference / "verification.json")], "variant": variant})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("fine_grid", "critical_020", "critical_060"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    args = parser.parse_args()
    run(args.reference, args.run_dir, args.variant, args.data_dir)
