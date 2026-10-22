from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.energy.comparison import (ComparisonConfig, FrozenForecastProvider, OracleForecastProvider,
    STRATEGIES, run_paired_episode, select_episodes, summarize_results)
from bekantor_energy.energy.planner import AStarPlanner, PlannerConfig, PlanningLimitError


def fixture_case():
    return {"case_id": "fixture", "episode_id": "day_01", "start": "2009-09-21T00:00:00",
            "demand_stratum": "low", "solar_level": "low", "tariff_mode": "flat", "initial_soc": .5,
            "home": asdict(HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.5)),
            "planner": asdict(PlannerConfig(storage_step_kwh=.5)),
            "actual_demand": [1.] * 24, "solar": [0.] * 48,
            "predictions": {"mlp_mse": [[.8] * 24] * 24, "mlp_weighted": [[1.1] * 24] * 24},
            "oracle_predictions": [[1.] * 24] * 24,
            "model_ids": {"mlp_mse": "frozen_mse_fixture", "mlp_weighted": "frozen_weighted_fixture"}}


class EnergyComparisonTests(unittest.TestCase):
    def test_config_fixes_scenarios_and_seed(self):
        config = ComparisonConfig.load(Path(__file__).parents[1] / "configs/energy.comparison.json")
        self.assertEqual(config.days_per_demand_stratum, 7)
        for values in ({"seed": 2028}, {"workers": 0}, {"solar_factors": {"low": 2, "high": 1}},
                       {"initial_soc_levels": (.5, .5)}, {"flat_tariff_sim_per_kwh": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(config, **values)

    def test_selection_is_nonoverlapping_temporally_distributed_and_cost_free(self):
        times = pd.date_range("2009-09-21", periods=24 * 60, freq="h")
        demand = np.repeat(np.arange(1, 61, dtype=float), 24)
        validation = pd.DataFrame({"load_kwh": demand, "is_observed_complete": True}, index=times)
        selected, candidates, meta = select_episodes(times[:-23], validation, 7)
        self.assertEqual(len(selected), 14)
        self.assertEqual(selected.demand_stratum.value_counts().to_dict(), {"low": 7, "high": 7})
        self.assertTrue((selected.start.diff().dropna() >= pd.Timedelta(hours=24)).all())
        self.assertFalse(selected.start.duplicated().any())
        self.assertTrue(meta["retrospective_design"])

    def test_neural_provider_has_no_realized_demand_dependency_or_oracle_marker(self):
        times = pd.date_range("2009-09-21", periods=24, freq="h")
        predicted = np.ones((24, 24))
        provider = FrozenForecastProvider(times, predicted, "frozen")
        oracle = OracleForecastProvider(times, predicted * 100, "ORACLE")
        self.assertEqual(provider(times[0]).demand_kwh, (1.,) * 24)
        self.assertEqual(oracle(times[0]).demand_kwh, (100.,) * 24)
        self.assertFalse(hasattr(provider, "actual_demand"))
        with self.assertRaises(ValueError):
            FrozenForecastProvider(times, predicted * np.nan, "bad")

    def test_four_strategies_share_conditions_full_plan_feasibility_and_energy_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = fixture_case()
            result = run_paired_episode(case, root / "case")
            self.assertEqual(len(result["metrics"]), 4)
            self.assertFalse(result["failures"])
            ledger = pd.read_csv(root / "case" / "ledger.csv")
            for strategy, group in ledger.groupby("strategy"):
                self.assertEqual(len(group), 24)
                np.testing.assert_array_equal(group.demand_kwh, np.ones(24))
                self.assertEqual(group.constraint_violations.sum(), 0)
                self.assertGreaterEqual(group.soc.min(), .2)
            audit = json.loads((root / "case" / "plan_audit.json").read_text())
            for values in audit.values():
                self.assertEqual(values["checked_plans"], 24)
                self.assertEqual(values["checked_transitions"], 24 * 24)
            for values in result["metrics"]:
                self.assertIn("charge_kwh", values)
                self.assertIn("discharge_kwh", values)
                self.assertEqual(values["is_oracle"], values["strategy"] == "oracle")

    def test_future_realization_changes_only_oracle_initial_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = fixture_case()
            second = fixture_case()
            second["actual_demand"][1:] = [10.] * 23
            future_truth = [1.] + [10.] * 46
            second["oracle_predictions"] = [future_truth[t:t + 24] for t in range(24)]
            run_paired_episode(first, root / "first")
            run_paired_episode(second, root / "second")
            a = pd.read_csv(root / "first" / "ledger.csv").groupby("strategy").first()
            b = pd.read_csv(root / "second" / "ledger.csv").groupby("strategy").first()
            for name in ("reactive", "mlp_mse", "mlp_weighted"):
                self.assertEqual(a.loc[name, "requested_charge_kw"], b.loc[name, "requested_charge_kw"])
                self.assertEqual(a.loc[name, "requested_discharge_kw"], b.loc[name, "requested_discharge_kw"])
            forecasts = pd.read_csv(root / "second" / "forecasts.csv.gz")
            self.assertTrue(forecasts.loc[forecasts.strategy == "oracle", "is_oracle"].all())

    def test_search_failures_are_retained_without_fallback_or_silent_exclusion(self):
        class FailingPlanner:
            def __init__(self, settings):
                pass
            def plan(self, *args):
                raise PlanningLimitError("fixture resource budget")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_paired_episode(fixture_case(), root / "case", planner_factory=FailingPlanner)
            self.assertEqual(len(result["failures"]), 3)
            self.assertEqual([v["status"] for v in result["metrics"]], ["completed", "failed", "failed", "failed"])
            rows = [{"case_id": "fixture", **m} for m in result["metrics"]]
            summarize_results(rows, root)
            coverage = json.loads((root / "coverage.json").read_text())
            self.assertEqual(coverage["four_way_complete"], 0)
            self.assertEqual(coverage["failed_cases"], 1)
            self.assertEqual(len(pd.read_csv(root / "results_by_case.csv")), 4)

    def test_paired_percentages_dispersion_zero_denominator_and_grid_tradeoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for case, costs in (("case_1", (1., .8, .7, .6)), ("case_2", (0., .2, .1, 0.))):
                for i, name in enumerate(STRATEGIES):
                    rows.append({"case_id": case, "episode_id": case, "strategy": name, "status": "completed",
                                 "solar_level": "high", "tariff_mode": "flat", "initial_soc": .5, "demand_stratum": "low",
                                 "import_cost_sim": costs[i], "grid_import_kwh": float(i + 1)})
            summarize_results(rows, root)
            paired = pd.read_csv(root / "paired_results.csv")
            self.assertTrue(paired.loc[paired.case_id == "case_2", "improvement_pct_import_cost_sim_vs_reactive"].isna().all())
            mse = paired[(paired.case_id == "case_1") & (paired.strategy == "mlp_mse")].iloc[0]
            self.assertAlmostEqual(mse.improvement_pct_import_cost_sim_vs_reactive, 20)
            self.assertTrue(mse.cost_down_grid_up_vs_reactive)
            aggregate = pd.read_csv(root / "aggregate.csv").set_index("strategy")
            self.assertAlmostEqual(aggregate.loc["mlp_mse", "import_cost_sim__std"], np.std([.8, .2], ddof=1))
