"""Derive descriptive paired findings from completed validation simulations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import write_json


def analyze(run_dir):
    root = Path(run_dir)
    frame = pd.read_csv(root / "paired_results.csv")
    measures = ["import_cost_sim", "grid_import_kwh", "final_soc", "solar_utilization_pct"]
    index = ["case_id", "episode_id", "demand_stratum", "solar_level", "tariff_mode", "initial_soc"]
    wide = frame.pivot(index=index, columns="strategy", values=measures)
    wide.columns = [f"{metric}__{strategy}" for metric, strategy in wide.columns]
    wide = wide.reset_index()
    wide["weighted_minus_mse_cost_sim"] = wide.import_cost_sim__mlp_weighted - wide.import_cost_sim__mlp_mse
    wide["weighted_minus_mse_grid_kwh"] = wide.grid_import_kwh__mlp_weighted - wide.grid_import_kwh__mlp_mse
    wide["weighted_minus_mse_final_soc"] = wide.final_soc__mlp_weighted - wide.final_soc__mlp_mse
    wide["same_terminal_soc_models"] = wide.weighted_minus_mse_final_soc.abs() <= 1e-9
    wide["weighted_cheaper"] = wide.weighted_minus_mse_cost_sim < -1e-9
    wide["weighted_more_expensive"] = wide.weighted_minus_mse_cost_sim > 1e-9
    wide.to_csv(root / "model_paired_diagnostics.csv", index=False)
    grouped = []
    for tariff, subset in [("all", wide), *list(wide.groupby("tariff_mode"))]:
        delta = subset.weighted_minus_mse_cost_sim
        denominator = subset.import_cost_sim__mlp_mse.sum()
        grouped.append({"tariff_mode": tariff, "n_cases": len(subset), "weighted_cost_wins": int(subset.weighted_cheaper.sum()),
            "weighted_cost_losses": int(subset.weighted_more_expensive.sum()),
            "weighted_cost_ties": int((~subset.weighted_cheaper & ~subset.weighted_more_expensive).sum()),
            "weighted_cost_change_ratio_of_sums_pct": float(100 * delta.sum() / denominator) if denominator else None,
            "weighted_minus_mse_cost_mean": float(delta.mean()), "weighted_minus_mse_cost_std": float(delta.std()),
            "weighted_minus_mse_grid_mean_kwh": float(subset.weighted_minus_mse_grid_kwh.mean()),
            "equal_terminal_soc_cases": int(subset.same_terminal_soc_models.sum()),
            "weighted_wins_with_equal_terminal_soc": int((subset.weighted_cheaper & subset.same_terminal_soc_models).sum()),
            "weighted_wins_with_lower_terminal_soc": int((subset.weighted_cheaper & (subset.weighted_minus_mse_final_soc < -1e-9)).sum())})
    pd.DataFrame(grouped).to_csv(root / "model_comparison_summary.csv", index=False)
    per_day = wide.groupby("episode_id").agg(
        weighted_minus_mse_cost_mean=("weighted_minus_mse_cost_sim", "mean"),
        weighted_minus_mse_cost_min=("weighted_minus_mse_cost_sim", "min"),
        weighted_minus_mse_cost_max=("weighted_minus_mse_cost_sim", "max"),
        weighted_cost_wins=("weighted_cheaper", "sum"), weighted_cost_losses=("weighted_more_expensive", "sum"))
    per_day.to_csv(root / "model_comparison_by_day.csv")
    predictive = pd.read_csv(root / "predictive_metrics.csv")
    predictive_by_day = predictive.pivot(index="episode_id", columns="strategy", values="rmse_kwh")
    overall_predictive = {name: float(np.sqrt(np.mean(predictive.loc[predictive.strategy == name, "rmse_kwh"] ** 2)))
                          for name in ("mlp_mse", "mlp_weighted")}
    # Equal numbers of origin/horizon pairs per day: average MSE then root, not mean RMSE.
    predictive_by_day["weighted_rmse_worse"] = predictive_by_day.mlp_weighted > predictive_by_day.mlp_mse
    per_day = per_day.join(predictive_by_day)
    per_day.to_csv(root / "prediction_vs_energy_by_day.csv")
    tradeoffs = []
    for (strategy, tariff), subset in frame[frame.strategy != "reactive"].groupby(["strategy", "tariff_mode"]):
        tradeoffs.append({"strategy": strategy, "tariff_mode": tariff, "n_cases": len(subset),
                          "cost_down_grid_up": int(subset.cost_down_grid_up_vs_reactive.sum()),
                          "cost_increase_cases": int((subset.delta_import_cost_sim_vs_reactive > 1e-9).sum())})
    pd.DataFrame(tradeoffs).to_csv(root / "tradeoff_counts.csv", index=False)
    oracle_comparison = {}
    for strategy in ("mlp_mse", "mlp_weighted", "reactive"):
        oracle_comparison[strategy] = int((wide[f"import_cost_sim__{strategy}"] < wide.import_cost_sim__oracle - 1e-9).sum())
    findings = {"predictive_rmse_equal_origin_horizon_pairs_kwh": overall_predictive,
                "weighted_predictive_rmse_worse_days": int(predictive_by_day.weighted_rmse_worse.sum()),
                "weighted_cheaper_on_daily_scenario_mean_days": int((per_day.weighted_minus_mse_cost_mean < -1e-9).sum()),
                "cases_cheaper_than_rolling_oracle": oracle_comparison,
                "oracle_warning": "Rolling 24h discrete oracle is a perfect-information reference, not a guaranteed lower bound on the reported realized day",
                "statistical_caveat": "14 unique selected demand days; 112 factorial repetitions are not independent demand-day observations",
                "weighted_vs_mse": grouped}
    write_json(root / "findings.json", findings)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.run_dir)
