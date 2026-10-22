"""Continuous energy fixtures; no real held-out partition or neural fitting."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.continuous import (ContinuousConfig, STRATEGIES, adjusted_cost,
    run_continuous_case, select_continuous_blocks, summarize_continuous, validate_case)
from bekantor_energy.energy.planner import AStarPlanner, BatterySearchProblem, PlannerConfig, PlanningLimitError


def fixture(hours=48):
    return {"case_id": "fixture", "block_id": "early", "solar_profile": "fixture", "tariff_mode": "flat",
        "start": "2009-09-21T00:00:00", "hours": hours,
        "home": asdict(HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.5)),
        "planner": asdict(PlannerConfig(storage_step_kwh=.25)), "terminal_min_soc": .5,
        "valuation_sim_per_stored_kwh": .1425, "actual_demand": [1.] * hours,
        "solar": [0.] * (hours + 23), "predictions": [[.8] * 24 for _ in range(hours)], "model_id": "frozen_fixture"}


class ContinuousEnergyTests(unittest.TestCase):
    def test_predeclared_configuration_and_invalid_values(self):
        settings = ContinuousConfig.load(Path(__file__).parents[1] / "configs/energy.continuous.json")
        self.assertEqual(settings.terminal_min_soc, .5)
        self.assertEqual(settings.valuation_sim_per_stored_kwh, .15 * .95)
        for values in ({"days_per_block": 6}, {"terminal_min_soc": float("nan")}, {"workers": 0},
                       {"seed": 2028}, {"solar_sources": {}}, {"valuation_sensitivity": (0.,)},
                       {"valuation_sensitivity": (.1425, .1425)}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(settings, **values)

    def test_first_complete_block_in_each_half_without_overlapping_tails(self):
        times = pd.date_range("2009-09-01", periods=24 * 60, freq="h")
        values = pd.DataFrame({"load_kwh": 1., "is_observed_complete": True}, index=times)
        origins = times[:-23]
        selected, candidates = select_continuous_blocks(origins, values)
        self.assertEqual(list(selected.block_id), ["early", "late"])
        self.assertEqual(selected.start.iloc[0], times[0])
        self.assertGreater(selected.start.iloc[1], selected.forecast_tail_end.iloc[0])
        values.loc[times[100], "is_observed_complete"] = False
        changed, _ = select_continuous_blocks(origins, values)
        self.assertGreater(changed.start.iloc[0], times[100])
        self.assertFalse(candidates.empty)

    def test_terminal_accounting_sign_units_and_telescoping(self):
        self.assertAlmostEqual(adjusted_cost(10, 2, 3, .1425), 9.8575)
        self.assertAlmostEqual(adjusted_cost(10, 3, 2, .1425), 10.1425)
        self.assertEqual(adjusted_cost(10, 2, 2, .38), 10)
        self.assertAlmostEqual(adjusted_cost(5, 2, 3, .1425) + adjusted_cost(5, 3, 1, .1425),
                               adjusted_cost(10, 2, 1, .1425))
        for values in ((10, 2, 3, -1), (float("nan"), 2, 3, 1), (1, -2, 3, 1)):
            with self.assertRaises(ValueError):
                adjusted_cost(*values)

    def test_no_midnight_reset_strict_plan_goals_and_realized_energy(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "case"
            result = run_continuous_case(fixture(), output)
            self.assertFalse(result["failures"])
            ledger = pd.read_csv(output / "ledger.csv")
            for name, rows in ledger.groupby("strategy"):
                self.assertEqual(len(rows), 48)
                self.assertEqual(rows.demand_kwh.sum(), 48)
                self.assertAlmostEqual(rows.initial_kwh.iloc[0], .5)
                np.testing.assert_allclose(rows.initial_kwh.iloc[1:], rows.stored_kwh.iloc[:-1], atol=1e-12)
                self.assertAlmostEqual(rows.initial_kwh.iloc[24], rows.stored_kwh.iloc[23])
                self.assertEqual(rows.constraint_violations.sum(), 0)
                self.assertGreaterEqual(rows.soc.min(), .2)
            original = ledger[ledger.strategy == "mlp_mse"]
            self.assertNotAlmostEqual(original.initial_kwh.iloc[24], .5)
            searches = pd.read_csv(output / "searches.csv")
            self.assertEqual(len(searches), 96)
            self.assertTrue((searches.predicted_terminal_kwh >= searches.required_terminal_kwh - 1e-9).all())
            self.assertEqual(len(pd.read_csv(output / "plans.csv.gz")), 96 * 24)
            self.assertAlmostEqual(result["metrics"][0]["adjusted_cost_sim"],
                result["metrics"][0]["import_cost_sim"] + .1425 * (.5 - .2))

    def test_unknown_future_realization_cannot_change_first_requested_action(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = fixture(2), fixture(2)
            second["actual_demand"][1] = 50.
            run_continuous_case(first, root / "first")
            run_continuous_case(second, root / "second")
            a = pd.read_csv(root / "first/ledger.csv").groupby("strategy").first()
            b = pd.read_csv(root / "second/ledger.csv").groupby("strategy").first()
            for name in STRATEGIES:
                np.testing.assert_array_equal(a.loc[name, ["requested_charge_kw", "requested_discharge_kw"]],
                                              b.loc[name, ["requested_charge_kw", "requested_discharge_kw"]])

    def test_missing_solar_tail_and_bad_physical_contracts_are_rejected(self):
        for key, value in (("solar", [0.] * 48), ("terminal_min_soc", .1), ("predictions", [[1.] * 23] * 48),
                           ("actual_demand", [float("nan")] * 48)):
            case = fixture()
            case[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_case(case)

    def test_search_failure_is_not_replaced_and_empty_coverage_is_explicit(self):
        class FailingPlanner:
            def __init__(self, settings):
                pass
            def plan(self, *args):
                raise PlanningLimitError("fixture budget")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_continuous_case(fixture(2), root / "case", planner_factory=FailingPlanner)
            self.assertEqual(set(result["failures"]), {"mlp_mse", "mlp_mse_terminal"})
            self.assertEqual([r["hours_completed"] for r in result["metrics"]], [2, 0, 0])
            summarize_continuous([{"case_id": "fixture", **r} for r in result["metrics"]], root, (.1425,))
            self.assertEqual(json.loads((root / "coverage.json").read_text())["complete_three_way_cases"], 0)

    def test_paired_metrics_and_sensitivity_keep_terminal_energy_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for name, cost, final in (("reactive", 10., 1.), ("mlp_mse", 9.9, 0.), ("mlp_mse_terminal", 10.1, 2.)):
                rows.append({"case_id": "c", "block_id": "early", "solar_profile": "fixture", "tariff_mode": "flat",
                    "strategy": name, "status": "completed", "battery_initial_kwh": 1., "battery_final_kwh": final,
                    "import_cost_sim": cost, "adjusted_cost_sim": adjusted_cost(cost, 1, final, .2), "grid_import_kwh": cost})
            summarize_continuous(rows, root, (0., .2))
            sensitivity = pd.read_csv(root / "sensitivity.csv")
            mse = sensitivity[sensitivity.strategy == "mlp_mse"].set_index("valuation_sim_per_stored_kwh")
            self.assertLess(mse.loc[0., "delta_vs_reactive_sim"], 0)
            self.assertGreater(mse.loc[.2, "delta_vs_reactive_sim"], 0)
            self.assertEqual(len(pd.read_csv(root / "results_by_case.csv")), 3)


class TerminalSearchGuarantees(unittest.TestCase):
    def inputs(self, demand, solar, prices, max_charge_kw=2.5):
        n = len(demand)
        times = tuple(pd.date_range("2009-01-01", periods=n, freq="h").to_pydatetime())
        config = HomeConfig(battery_capacity_kwh=2, min_soc=.25, initial_soc=.5, horizon_steps=n,
                            import_tariff_per_kwh=tuple(prices), max_charge_kw=max_charge_kw)
        return PlanningInputs(LoadForecast(times[0], times, tuple(demand), "fixture"), tuple(solar), config, "fixture")

    def test_relaxed_astar_matches_ucs_with_terminal_constraints_randomized(self):
        rng = np.random.default_rng(2026)
        for i in range(24):
            data = self.inputs(rng.uniform(0, 2, 4), rng.uniform(0, 2, 4), rng.choice([0., .08, .18, .4], 4))
            settings = PlannerConfig(storage_step_kwh=.5, terminal_min_soc=(.5, .9, 1.)[i % 3])
            plan = AStarPlanner(settings).plan(data, EnergyState(0, 1.))
            ucs = AStarPlanner(replace(settings, heuristic="zero")).plan(data, EnergyState(0, 1.))
            self.assertAlmostEqual(plan.predicted_cost, ucs.predicted_cost, places=10)
            self.assertGreaterEqual(plan.states[-1].stored_kwh, settings.terminal_min_soc * 2 - 1e-9)

    def test_bound_is_admissible_from_intermediate_states_and_infeasible_goals_fail(self):
        data = self.inputs([2., .2, 1.], [0., 2., 0.], [.4, .08, .18])
        settings = PlannerConfig(storage_step_kwh=.5, terminal_min_soc=1.)
        problem = BatterySearchProblem(data, EnergyState(0, 1.), settings)
        for t in range(3):
            for energy in problem.levels:
                state = EnergyState(t, energy)
                exact = AStarPlanner(replace(settings, heuristic="zero"))
                zero_problem = BatterySearchProblem(data, EnergyState(0, 1.), exact.settings)
                try:
                    remaining = exact.search(zero_problem, state).predicted_cost
                except ValueError:
                    continue
                self.assertLessEqual(problem.heuristic(state), remaining + 1e-10)
        impossible = self.inputs([0.], [0.], [.1], max_charge_kw=.01)
        with self.assertRaisesRegex(ValueError, "No feasible plan"):
            AStarPlanner(settings).plan(impossible, EnergyState(0, .5))
