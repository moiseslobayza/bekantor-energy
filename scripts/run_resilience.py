"""Stage-gated resilience experiment with frozen demand forecasts, never test."""
import argparse
from pathlib import Path
from bekantor_energy.energy.resilience_experiment import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/energy.resilience.json"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "full", "panel"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/uci"))
    parser.add_argument("--cache-source", type=Path)
    args = parser.parse_args()
    run_experiment(args.config, args.run_dir, args.stage, args.data_dir, args.cache_source)
