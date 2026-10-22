"""Synthetic objective proofs and continuous replay; no neural fit or held-out data."""
from dataclasses import asdict, replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.continuous import run_continuous_case, summarize_continuous
from bekantor_energy.energy.mode_experiment import MODE_STRATEGIES, ModeComparisonConfig, run_mode_case
from bekantor_energy.energy.modes import ModePlanner, ModeSearchProblem, ObjectiveConfig
from bekantor_energy.energy.planner import AStarPlanner, BatterySearchProblem, PlannerConfig, PlanningLimitError
from bekantor_energy.energy.search import LexicographicArithmetic, astar_search


def inputs(demand=(1., 1., 1.), solar=None, prices=None, **kwargs):
    n = len(demand)
    times = tuple(pd.date_range("2009-01-01", periods=n, freq="h").to_pydatetime())
    home = HomeConfig(battery_capacity_kwh=2, min_soc=.25, initial_soc=.5, horizon_steps=n,
                      import_tariff_per_kwh=tuple(prices) if prices is not None else (.15,), **kwargs)
    return PlanningInputs(LoadForecast(times[0], times, tuple(demand), "fixture"),
                          tuple(solar) if solar is not None else (0.,) * n, home, "fixture")


def mode_fixture(hours=2):
    return {"case_id": "fixture", "block_id": "early", "solar_profile": "fixture", "tariff_mode": "flat",
        "start": "2009-09-21T00:00:00", "hours": hours,
        "home": asdict(HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.5)),
        "planner": asdict(PlannerConfig(storage_step_kwh=.25)), "terminal_min_soc": .2,
        "valuation_sim_per_stored_kwh": .1425, "actual_demand": [1.] * hours,
        "solar": [0.] * (hours + 23), "predictions": [[.8] * 24 for _ in range(hours)],
        "model_id": "frozen_fixture", "objectives": {name: asdict(ObjectiveConfig(name)) for name in MODE_STRATEGIES[1:]}}


def independent_exhaustive(data, settings, objective):
    graph = BatterySearchProblem(data, EnergyState(0, 1.), settings)
    lex = objective.mode == "autonomy"
    def visit(state):
        if graph.is_goal(state):
            return (Fraction(0), Fraction(0)) if lex else 0.
        candidates = []
        for _, nxt, result in graph.physical_successors(state):
            future = visit(nxt)
            if future is None:
                continue
            if lex:
                q = lambda v: Fraction.from_float(float(v))
                if nxt.stored_kwh >= state.stored_kwh:
                    ac_charge = (q(nxt.stored_kwh) - q(state.stored_kwh)) / q(data.config.charge_efficiency)
                    ac_discharge = Fraction(0)
                else:
                    ac_charge = Fraction(0)
                    ac_discharge = (q(state.stored_kwh) - q(nxt.stored_kwh)) * q(data.config.discharge_efficiency)
                imported = max(Fraction(0), q(data.forecast.demand_kwh[state.step]) + ac_charge
                               - q(data.solar_kwh[state.step]) - ac_discharge)
                value = (imported + future[0], imported * q(data.config.import_prices[state.step]) + future[1])
            elif objective.mode == "economic":
                value = result.cost + future
            else:
                value = objective.cost_weight * result.cost / objective.cost_scale_sim + (1 - objective.cost_weight) * result.grid_import_kwh / objective.grid_scale_kwh + future
            candidates.append(value)
        return min(candidates) if candidates else None
    return visit(EnergyState(0, 1.))


class ObjectiveTests(unittest.TestCase):
    def test_parameters_units_normalization_and_invalid_values(self):
        fixed = ModeComparisonConfig.load(Path(__file__).parents[1] / "configs/energy.modes.json")
        self.assertEqual(fixed.objectives()["balanced"], ObjectiveConfig("balanced", .5, 1., 1.))
        objective = ObjectiveConfig("balanced", .5, 2., 4.)
        self.assertAlmostEqual(objective.edge_value(2., 1.2), .55)
        converted = ObjectiveConfig("balanced", .5, 200., 4.)
        self.assertEqual(objective.edge_value(2., 1.2), converted.edge_value(2., 120.))
        self.assertEqual(ObjectiveConfig("autonomy").units, ("kWh", "SIM"))
        for values in ({"mode": "resilience"}, {"cost_weight": -.1}, {"cost_weight": float("nan")},
                       {"cost_scale_sim": 0}, {"grid_scale_kwh": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ObjectiveConfig(**values)

    def test_autonomy_exact_primary_and_cost_tie_without_scalar_epsilon(self):
        class Graph:
            def heuristic(self, state):
                return LexicographicArithmetic.zero
            def successors(self, state):
                if state.step:
                    return []
                return [("least_import", EnergyState(1, 1), (Fraction(1), Fraction(1000000))),
                        ("cheap_but_more_grid", EnergyState(1, 2), (Fraction(1) + Fraction(1, 10**15), Fraction(0))),
                        ("same_import_cheaper", EnergyState(1, 3), (Fraction(1), Fraction(1)))]
            def is_goal(self, state):
                return state.step == 1
        result = astar_search(Graph(), EnergyState(0, 0), PlannerConfig(), PlanningLimitError, LexicographicArithmetic())
        self.assertEqual(result.actions, ("same_import_cheaper",))
        self.assertEqual(result.value, (Fraction(1), Fraction(1)))

    def test_exact_accumulation_recognizes_real_sum_ties(self):
        # Same edge multiset, different summation order; exact binary edge sums tie.
        ops = LexicographicArithmetic()
        a = [ObjectiveConfig("autonomy").edge_value(v, 1.) for v in (.1, .2, .3)]
        one, two = ops.zero, ops.zero
        for value in a:
            one = ops.add(one, value)
        for value in reversed(a):
            two = ops.add(two, value)
        self.assertEqual(one, two)

    def test_real_battery_autonomy_tie_prefers_lower_cost(self):
        data = inputs((1., 1.), prices=(.4, .08), charge_efficiency=1., discharge_efficiency=1.)
        plan = ModePlanner(PlannerConfig(storage_step_kwh=.5), ObjectiveConfig("autonomy")).plan(data, EnergyState(0, 1.))
        self.assertAlmostEqual(plan.predicted_grid_import_kwh, 1.5)
        self.assertAlmostEqual(plan.predicted_cost, .28)
        self.assertAlmostEqual(plan.actions[0].discharge_kw, .5)
        self.assertEqual(plan.heuristic_name, "zero")

    def test_realistic_efficiency_tie_is_not_broken_by_rounded_import_edges(self):
        data = inputs((1., 1.), prices=(.4, .08))  # eta=.95; .25+.25 discharge ties .5+0.
        plan = ModePlanner(PlannerConfig(storage_step_kwh=.25), ObjectiveConfig("autonomy")).plan(data, EnergyState(0, 1.))
        self.assertAlmostEqual(plan.predicted_grid_import_kwh, 1.525)
        self.assertAlmostEqual(plan.predicted_cost, .29)
        self.assertAlmostEqual(plan.actions[0].discharge_kw, .475)
        self.assertEqual(plan.actions[1].discharge_kw, 0.)

    def test_economic_mode_preserves_reference_paths_and_monetary_contract(self):
        data = inputs((1., 1., 2.), solar=(0., 1.5, 0.), prices=(.08, .18, .4))
        settings = PlannerConfig(storage_step_kwh=.5)
        old = AStarPlanner(settings).plan(data, EnergyState(0, 1.))
        new = ModePlanner(settings, ObjectiveConfig("economic")).plan(data, EnergyState(0, 1.))
        self.assertEqual(old.actions, new.actions)
        self.assertEqual(old.states, new.states)
        self.assertEqual(old.predicted_cost, new.predicted_cost)
        balance = ModePlanner(settings, ObjectiveConfig("balanced")).plan(data, EnergyState(0, 1.))
        self.assertAlmostEqual(balance.objective_primary, .5 * balance.predicted_cost + .5 * balance.predicted_grid_import_kwh)
        self.assertNotEqual(balance.predicted_cost, balance.objective_primary)

    def test_all_modes_match_ucs_and_independent_enumeration(self):
        rng = np.random.default_rng(2026)
        for i in range(12):
            data = inputs(tuple(rng.uniform(0, 2, 3)), tuple(rng.uniform(0, 2, 3)), tuple(rng.choice([0., .08, .18, .4], 3)))
            settings = PlannerConfig(storage_step_kwh=.5, terminal_min_soc=(None, .75)[i % 2])
            for name in MODE_STRATEGIES[1:]:
                objective = ObjectiveConfig(name)
                a = ModePlanner(settings, objective).plan(data, EnergyState(0, 1.))
                b = ModePlanner(replace(settings, heuristic="zero"), objective).plan(data, EnergyState(0, 1.))
                exact = independent_exhaustive(data, settings, objective)
                self.assertAlmostEqual(a.objective_primary, b.objective_primary, places=10)
                self.assertAlmostEqual(a.objective_primary, float(exact[0] if name == "autonomy" else exact), places=10)
                if name == "autonomy":
                    self.assertAlmostEqual(a.objective_secondary, float(exact[1]), places=10)
                    self.assertAlmostEqual(a.objective_secondary, b.objective_secondary, places=10)

    def test_balanced_bound_admissible_at_intermediate_states_and_weight_endpoints(self):
        data = inputs((2., .2, 1.), (0., 2., 0.), (.4, .08, .18))
        for weight in (0., .25, .5, 1.):
            objective = ObjectiveConfig("balanced", weight, 2., .5)
            settings = PlannerConfig(storage_step_kwh=.5, terminal_min_soc=.75)
            graph = ModeSearchProblem(data, EnergyState(0, 1.), settings, objective)
            zero = ModeSearchProblem(data, EnergyState(0, 1.), replace(settings, heuristic="zero"), objective)
            for t in range(3):
                for energy in graph.levels:
                    state = EnergyState(t, energy)
                    try:
                        exact = astar_search(zero, state, settings, PlanningLimitError).value
                    except ValueError:
                        continue
                    self.assertLessEqual(graph.heuristic(state), exact + 1e-10)

    def test_unproven_search_is_failed_for_every_objective(self):
        for mode in MODE_STRATEGIES[1:]:
            with self.subTest(mode=mode), self.assertRaises(PlanningLimitError):
                ModePlanner(PlannerConfig(max_expanded=1), ObjectiveConfig(mode)).plan(inputs(), EnergyState(0, 1.))

    def test_no_export_or_outage_capability_is_implied(self):
        data = inputs()
        for mode in MODE_STRATEGIES[1:]:
            with self.assertRaises(ValueError):
                ModePlanner(objective=ObjectiveConfig(mode)).plan(replace(data, grid_available=(True, False, True)), EnergyState(0, 1.))


class ModeReplayTests(unittest.TestCase):
    def test_four_modes_use_shared_continuous_evaluator_and_physical_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = mode_fixture(26)
            result = run_mode_case(case, root / "case")
            self.assertFalse(result["failures"])
            self.assertEqual([m["strategy"] for m in result["metrics"]], list(MODE_STRATEGIES))
            ledger = pd.read_csv(root / "case/ledger.csv")
            for _, group in ledger.groupby("strategy"):
                self.assertEqual(len(group), 26)
                np.testing.assert_allclose(group.initial_kwh.iloc[1:], group.stored_kwh.iloc[:-1])
                self.assertEqual(group.constraint_violations.sum(), 0)
                self.assertEqual(group.demand_kwh.sum(), 26)
            self.assertTrue(all("mean_soc" in row for row in result["metrics"]))
            searches = pd.read_csv(root / "case/searches.csv")
            self.assertEqual(len(searches), 3 * 26)
            self.assertEqual(set(searches.objective_mode), set(MODE_STRATEGIES[1:]))

    def test_future_actual_demand_cannot_change_initial_mode_decisions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, changed = mode_fixture(), mode_fixture()
            changed["actual_demand"][1] = 50.
            run_mode_case(first, root / "a")
            run_mode_case(changed, root / "b")
            a = pd.read_csv(root / "a/ledger.csv").groupby("strategy").first()
            b = pd.read_csv(root / "b/ledger.csv").groupby("strategy").first()
            np.testing.assert_array_equal(a[["requested_charge_kw", "requested_discharge_kw"]], b[["requested_charge_kw", "requested_discharge_kw"]])

    def test_mean_soc_uses_hourly_trapezoids_not_final_or_maximum_soc(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_mode_case(mode_fixture(), Path(temporary) / "case")
            reactive = result["metrics"][0]
            self.assertAlmostEqual(reactive["mean_soc"], .275)
            self.assertNotEqual(reactive["mean_soc"], reactive["final_soc"])

    def test_failed_mode_has_no_fallback_and_is_visible_in_coverage(self):
        class Fail:
            def plan(self, *args):
                raise PlanningLimitError("fixture failure")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_continuous_case(mode_fixture(), root / "case", planning_policies={name: Fail() for name in MODE_STRATEGIES[1:]})
            self.assertEqual(set(result["failures"]), set(MODE_STRATEGIES[1:]))
            summarize_continuous([{"case_id": "fixture", **m} for m in result["metrics"]], root, (.1425,),
                                 strategies=MODE_STRATEGIES, reference_strategy="economic")
            coverage = json.loads((root / "coverage.json").read_text())
            self.assertEqual(coverage["complete_four_way_cases"], 0)
            self.assertEqual(coverage["failed_strategy_runs"], 3)
