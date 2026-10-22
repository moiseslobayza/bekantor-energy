from dataclasses import replace
from datetime import datetime, timedelta
from functools import lru_cache
import math
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.evaluation import evaluate_receding
from bekantor_energy.energy.planner import AStarPlanner, BatterySearchProblem, PlannerConfig, PlanningLimitError
from bekantor_energy.energy.simulator import PhysicalSimulator, SelfConsumptionStrategy


def inputs(demand, solar=None, prices=None, **home):
    n = len(demand)
    config = HomeConfig(horizon_steps=n, import_tariff_per_kwh=tuple(prices) if prices is not None else (.2,), **home)
    starts = tuple(datetime(2020, 1, 1) + timedelta(hours=i * config.step_hours) for i in range(n))
    return PlanningInputs(LoadForecast(starts[0], starts, tuple(demand), "synthetic"),
                          tuple(solar) if solar is not None else (0.,) * n, config, "synthetic_fixture")


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.sim, self.c = PhysicalSimulator(), HomeConfig()

    def test_empty_at_reserve_cannot_create_energy(self):
        state = EnergyState(0, 1.)
        with self.assertRaises(ValueError):
            self.sim.step(state, EnergyAction(discharge_kw=1), 2, 0, self.c)
        result = self.sim.step(state, EnergyAction(discharge_kw=1), 2, 0, self.c, repair=True)
        self.assertEqual(result.state.stored_kwh, 1)
        self.assertEqual(result.grid_import_kwh, 2)
        self.assertTrue(result.action_adjusted)

    def test_full_battery_curtails_solar_without_double_efficiency(self):
        result = self.sim.step(EnergyState(0, 5), EnergyAction(charge_kw=2), 1, 6, self.c, repair=True)
        self.assertEqual(result.pv_direct_kwh, 1)
        self.assertEqual(result.curtailed_kwh, 5)
        self.assertEqual(result.state.stored_kwh, 5)
        self.assertEqual(result.grid_import_kwh, 0)

    def test_round_trip_losses_and_grid_charging(self):
        c = replace(self.c, min_soc=0, initial_soc=0)
        charged = self.sim.step(EnergyState(0, 0), EnergyAction(charge_kw=1), 0, 0, c)
        self.assertAlmostEqual(charged.state.stored_kwh, .95)
        self.assertEqual(charged.grid_charge_kwh, 1)
        discharged = self.sim.step(charged.state, EnergyAction(discharge_kw=.95 ** 2), 1, 0, c)
        self.assertAlmostEqual(discharged.state.stored_kwh, 0)
        self.assertAlmostEqual(charged.conversion_loss_kwh + discharged.conversion_loss_kwh, 1 - .95 ** 2)

    def test_kw_and_kwh_with_half_hour_step(self):
        c = replace(self.c, step_hours=.5)
        result = self.sim.step(EnergyState(0, 2), EnergyAction(charge_kw=2), 0, 0, c)
        self.assertEqual(result.charge_kwh, 1)
        self.assertAlmostEqual(result.state.stored_kwh, 2.95)

    def test_outage_serves_load_before_charging_and_reports_unserved(self):
        result = self.sim.step(EnergyState(0, 1), EnergyAction(charge_kw=2), 3, 1, self.c,
                               grid_available=False, repair=True)
        self.assertEqual(result.grid_import_kwh, 0)
        self.assertEqual(result.charge_kwh, 0)
        self.assertEqual(result.served_kwh, 1)
        self.assertEqual(result.unserved_kwh, 2)
        self.assertEqual(result.state.stored_kwh, 1)

    def test_no_simultaneous_charge_discharge_even_in_repair(self):
        for repair in (False, True):
            with self.assertRaises(ValueError):
                self.sim.step(EnergyState(0, 2), EnergyAction(1, 1), 2, 2, self.c, repair=repair)

    def test_no_battery_discharge_to_export_or_curtail(self):
        result = self.sim.step(EnergyState(0, 5), EnergyAction(discharge_kw=2), 0, 0, self.c, repair=True)
        self.assertEqual(result.discharge_kwh, 0)
        self.assertEqual(result.state.stored_kwh, 5)

    def test_power_caps_and_storage_caps(self):
        result = self.sim.step(EnergyState(0, 2), EnergyAction(charge_kw=100), 0, 0, self.c, repair=True)
        self.assertEqual(result.charge_kwh, 2.5)
        result = self.sim.step(EnergyState(0, 1.1), EnergyAction(discharge_kw=100), 8, 0, self.c, repair=True)
        self.assertAlmostEqual(result.discharge_kwh, .095)
        self.assertAlmostEqual(result.state.stored_kwh, 1)

    def test_optional_export_only_solar_surplus_and_not_during_outage(self):
        c = replace(self.c, export_tariff_per_kwh=(.05,))
        result = self.sim.step(EnergyState(0, 2), EnergyAction(), 1, 4, c, allow_export=True)
        self.assertEqual(result.grid_export_kwh, 3)
        self.assertEqual(result.grid_import_kwh, 0)
        self.assertAlmostEqual(result.cost, -.15)
        off = self.sim.step(EnergyState(0, 2), EnergyAction(), 1, 4, c, allow_export=True, grid_available=False)
        self.assertEqual(off.grid_export_kwh, 0)
        self.assertEqual(off.curtailed_kwh, 3)

    def test_invalid_inputs_rejected(self):
        for bad in (-1, math.inf, math.nan):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.sim.step(EnergyState(0, 2), EnergyAction(), bad, 0, self.c)
        for action in (EnergyAction(-1), EnergyAction(discharge_kw=math.nan)):
            with self.assertRaises(ValueError):
                self.sim.step(EnergyState(0, 2), action, 1, 0, self.c)
        for state in (EnergyState(0, .99), EnergyState(0, 5.1), EnergyState(-1, 2)):
            with self.assertRaises(ValueError):
                self.sim.step(state, EnergyAction(), 1, 0, self.c)

    def test_seeded_random_energy_balances_and_limits(self):
        rng = np.random.default_rng(2026)
        for i in range(1000):
            c = replace(self.c, charge_efficiency=float(rng.uniform(.5, 1)),
                        discharge_efficiency=float(rng.uniform(.5, 1)), step_hours=float(rng.choice([.25, .5, 1.])))
            stored = float(rng.uniform(1, 5))
            action = EnergyAction(charge_kw=float(rng.uniform(0, 10))) if i % 2 else EnergyAction(discharge_kw=float(rng.uniform(0, 10)))
            r = self.sim.step(EnergyState(i, stored), action, float(rng.uniform(0, 8)),
                               float(rng.uniform(0, 8)), c, grid_available=bool(i % 3), repair=True)
            self.assertLess(abs(r.balance_residual_kwh), 1e-9)
            self.assertLess(abs(r.storage_residual_kwh), 1e-9)
            self.assertGreaterEqual(r.state.stored_kwh, 1)
            self.assertLessEqual(r.state.stored_kwh, 5)
            self.assertEqual(r.constraint_violations, 0)
            self.assertAlmostEqual(stored + r.grid_import_kwh + r.solar_kwh,
                                   r.state.stored_kwh + r.served_kwh + r.curtailed_kwh +
                                   r.grid_export_kwh + r.conversion_loss_kwh, places=9)


class SearchTests(unittest.TestCase):
    def test_tariff_arbitrage_charges_then_discharges(self):
        data = inputs([0, 1], prices=[.1, 1], battery_capacity_kwh=2, min_soc=0, initial_soc=.25,
                      charge_efficiency=1, discharge_efficiency=1)
        plan = AStarPlanner(PlannerConfig(storage_step_kwh=.5)).plan(data, EnergyState(0, .5))
        self.assertAlmostEqual(plan.predicted_cost, .05)
        self.assertGreater(plan.actions[0].charge_kw, 0)
        self.assertGreater(plan.actions[1].discharge_kw, 0)

    def test_changed_demand_forecast_changes_plan(self):
        kwargs = dict(prices=[.1, 1], battery_capacity_kwh=2, min_soc=0, initial_soc=.25,
                      charge_efficiency=1, discharge_efficiency=1)
        planner = AStarPlanner(PlannerConfig(storage_step_kwh=.5))
        high = planner.plan(inputs([0, 1], **kwargs), EnergyState(0, .5))
        low = planner.plan(inputs([0, 0], **kwargs), EnergyState(0, .5))
        self.assertNotEqual(high.actions, low.actions)
        self.assertNotEqual(high.predicted_cost, low.predicted_cost)

    def test_off_grid_initial_state_and_boundaries_are_not_rounded(self):
        data = inputs([1, 2], min_soc=.2)
        initial = EnergyState(0, 2.123456)
        problem = BatterySearchProblem(data, initial, PlannerConfig())
        self.assertIn(initial.stored_kwh, problem.levels)
        self.assertIn(1, problem.levels)
        self.assertIn(5, problem.levels)
        for action, next_state, cost in problem.successors(initial):
            r = PhysicalSimulator().step(initial, action, 1, 0, data.config)
            self.assertAlmostEqual(r.state.stored_kwh, next_state.stored_kwh, places=10)

    def test_astar_matches_ucs_and_exhaustive_recursion_and_admissibility(self):
        rng = np.random.default_rng(97)
        for case in range(30):
            data = inputs(list(rng.uniform(0, 2, 3)), list(rng.uniform(0, 3, 3)),
                          list(rng.choice([0., .1, .5], 3)), battery_capacity_kwh=2,
                          initial_soc=.5, min_soc=.25, max_charge_kw=1, max_discharge_kw=1,
                          charge_efficiency=.8, discharge_efficiency=.9)
            initial = EnergyState(0, 1.)
            settings = PlannerConfig(storage_step_kwh=.5)
            problem = BatterySearchProblem(data, initial, settings)

            @lru_cache(None)
            def exhaustive(state):
                if problem.is_goal(state):
                    return 0.
                return min((cost + exhaustive(next_state) for _, next_state, cost in problem.successors(state)), default=math.inf)

            astar = AStarPlanner(settings).plan(data, initial)
            ucs = AStarPlanner(replace(settings, heuristic="zero")).plan(data, initial)
            self.assertAlmostEqual(astar.predicted_cost, ucs.predicted_cost, places=10)
            self.assertAlmostEqual(astar.predicted_cost, exhaustive(initial), places=10)
            for t in range(4):
                for stored in problem.levels:
                    state = EnergyState(t, stored)
                    self.assertLessEqual(problem.heuristic(state), exhaustive(state) + 1e-9, (case, state))

    def test_terminal_constraint_and_zero_heuristic(self):
        data = inputs([0, 0], prices=[.1, .2], battery_capacity_kwh=2, min_soc=0, initial_soc=.25)
        settings = PlannerConfig(storage_step_kwh=.5, terminal_min_soc=.75, heuristic="zero")
        plan = AStarPlanner(settings).plan(data, EnergyState(0, .5))
        self.assertGreaterEqual(plan.states[-1].stored_kwh, 1.5)
        self.assertGreater(plan.predicted_cost, 0)
        self.assertEqual(plan.heuristic_name, "zero")

    def test_search_limit_never_returns_unproven_optimum(self):
        with self.assertRaises(PlanningLimitError):
            AStarPlanner(PlannerConfig(max_expanded=1)).plan(inputs([1, 1]), EnergyState(0, 2.5))

    def test_negative_revenue_contract_is_rejected(self):
        data = inputs([1, 1], export_tariff_per_kwh=(.05,))
        with self.assertRaises(ValueError):
            AStarPlanner().plan(data, EnergyState(0, 2.5))

    def test_bad_forecast_and_grid_contracts(self):
        data = inputs([1, 1])
        for broken in (replace(data, solar_kwh=(1,)), replace(data, grid_available=(False, True)),
                       replace(data, forecast=replace(data.forecast, demand_kwh=(math.nan, 1)))):
            with self.assertRaises(ValueError):
                AStarPlanner().plan(broken, EnergyState(0, 2.5))
        for values in ({"storage_step_kwh": 0}, {"heuristic": "unsafe"}, {"max_seconds": 0}, {"terminal_min_soc": math.nan}):
            with self.assertRaises(ValueError):
                PlannerConfig(**values)


class ReplayTests(unittest.TestCase):
    def provider(self, origin):
        starts = tuple(pd.date_range(origin, periods=24, freq="h").to_pydatetime())
        return LoadForecast(starts[0], starts, (1.,) * 24, "fixed_forecast_fixture")

    def test_realized_demand_is_not_replaced_by_forecast_and_limits_hold(self):
        times = pd.date_range("2009-09-21", periods=3, freq="h")
        result = evaluate_receding([2, 3, 4], times, [0.] * 26, self.provider, HomeConfig())
        self.assertEqual(result["metrics"]["astar_mlp"]["demand_kwh"], 9)
        self.assertEqual(result["metrics"]["self_consumption"]["demand_kwh"], 9)
        self.assertEqual(result["metrics"]["astar_mlp"]["constraint_violations"], 0)
        self.assertEqual(len(result["planning"]), 3)
        self.assertEqual(len(result["first_plan"]), 24)

    def test_future_realizations_cannot_change_first_decision(self):
        times = pd.date_range("2009-09-21", periods=3, freq="h")
        first = evaluate_receding([1, 2, 3], times, [0.] * 26, self.provider, HomeConfig())
        changed = evaluate_receding([1, 50, 60], times, [0.] * 26, self.provider, HomeConfig())
        for result in (first, changed):
            self.assertTrue((result["forecasts"].origin <= result["forecasts"].target_timestamp).all())
        a = first["ledger"].query("strategy == 'astar_mlp'").iloc[0]
        b = changed["ledger"].query("strategy == 'astar_mlp'").iloc[0]
        self.assertEqual(a.requested_charge_kw, b.requested_charge_kw)
        self.assertEqual(a.requested_discharge_kw, b.requested_discharge_kw)
        self.assertEqual(a.stored_kwh, b.stored_kwh)

    def test_outage_replay_preserves_reserve_and_reports_ens(self):
        result = evaluate_receding([10], pd.date_range("2009-09-21", periods=1), [0.] * 24,
                                   self.provider, HomeConfig(), grid_available=[False])
        for scores in result["metrics"].values():
            self.assertGreater(scores["unserved_kwh"], 0)
            self.assertEqual(scores["grid_import_kwh"], 0)
            self.assertGreaterEqual(scores["min_soc"], .2)

    def test_baseline_never_intentionally_charges_from_grid(self):
        policy, c = SelfConsumptionStrategy(), HomeConfig()
        self.assertEqual(policy.action(2, 1, c).charge_kw, 0)
        self.assertEqual(policy.action(1, 2, c).discharge_kw, 0)
