"""Synthetic physics, causality, exact objective and independent UCS/DP checks."""
from dataclasses import asdict, replace
from fractions import Fraction
from functools import lru_cache
import heapq
from itertools import count
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.planner import BatterySearchProblem, PlannerConfig, PlanningLimitError
from bekantor_energy.energy.resilience import (POLICIES, ReadinessRisk, ResilienceConfig, ResiliencePlanner,
    ResilienceProblem, realized_step, resilience_metrics, reserve_action)
from bekantor_energy.energy.resilience_experiment import PlanStore, replay_policy, scenarios
from bekantor_energy.energy.simulator import PhysicalSimulator, SelfConsumptionStrategy


def forecast_inputs(demand, solar, home):
    times = tuple(pd.date_range("2009-10-01", periods=len(demand), freq="h").to_pydatetime())
    return PlanningInputs(LoadForecast(times[0], times, tuple(demand), "synthetic"), tuple(solar), home, "synthetic")


def fixture(hours=6):
    home = HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.5)
    case = {"case_id": "fixture", "block_id": "early", "solar_profile": "fixture", "tariff_mode": "flat",
        "start": "2009-10-01", "hours": hours, "home": asdict(home), "model_id": "frozen_synthetic",
        "actual_demand": [.3] * hours, "solar": [0.] * (hours + 23)}
    return {"case": case, "forecasts": {k: np.full((hours, 24), .3) for k in ("mlp", "oracle", "persistence")}}


class RiskTests(unittest.TestCase):
    def test_invalid_settings_and_no_current_or_out_of_window_start(self):
        for args in ({"critical_fraction": 0}, {"operational_soc": 1.1}, {"risk_hours": 0}):
            with self.assertRaises(ValueError): ResilienceConfig(**args)
        risk = ReadinessRisk([1.] * 24, [0.] * 24, HomeConfig())
        self.assertEqual(risk.starts, 20)
        for start in (0, 21):
            with self.assertRaises(ValueError): risk.phi(start, 2.5)
        with self.assertRaises(ValueError): ReadinessRisk([1.] * 4, [0.] * 4, HomeConfig())

    def test_phi_matches_independent_emergency_simulator_randomly(self):
        rng = np.random.default_rng(2026)
        home = HomeConfig()
        config = ResilienceConfig()
        for _ in range(30):
            demand, solar = rng.uniform(0, 6, (2, 24))
            energy = float(rng.uniform(1, 5))
            risk = ReadinessRisk(demand, solar, home, config)
            state, missing = EnergyState(0, energy), 0.
            for t in range(1, 5):
                critical = config.critical_fraction * demand[t]
                step = PhysicalSimulator().step(state, SelfConsumptionStrategy().action(critical, solar[t], home),
                    critical, solar[t], home, grid_available=False, repair=True)
                state, missing = step.state, missing + step.unserved_kwh
            self.assertAlmostEqual(float(risk.phi(1, energy)), missing, places=9)

    def test_target_is_minimum_zero_risk_energy_and_below_is_insufficient(self):
        risk = ReadinessRisk([1.] * 24, [0.] * 24, HomeConfig())
        target, feasible = risk.target()
        self.assertTrue(feasible)
        self.assertAlmostEqual(target, 1 + 1.6 / .95)
        self.assertLess(float(risk.phi(1, target)), 1e-9)
        self.assertGreater(float(risk.phi(1, target - .01)), 0.)

    def test_power_or_capacity_shortage_cannot_be_hidden_by_high_soc(self):
        power = ReadinessRisk([10.] * 24, [0.] * 24, HomeConfig())
        self.assertEqual(power.target(), (5., False))
        self.assertGreater(float(power.phi(1, 5.)), 0.)
        capacity = ReadinessRisk([1.] * 24, [0.] * 24, HomeConfig(battery_capacity_kwh=1))
        self.assertEqual(capacity.target(), (1., False))

    def test_solar_can_reduce_required_reserve_without_creating_energy(self):
        risk = ReadinessRisk([1.] * 24, [2.] * 24, HomeConfig())
        self.assertEqual(risk.target(), (1., True))
        self.assertEqual(risk.phi(1, 1.), Fraction(0))


class PhysicalAndMetricTests(unittest.TestCase):
    def test_fixed_operational_reserve_is_released_but_floor_is_preserved(self):
        home, config = HomeConfig(), ResilienceConfig()
        state = EnergyState(0, 2.5)
        action = reserve_action(2.5, 2.5, 1., 0., home)
        up, _ = realized_step(state, action, 1., 0., home, True, .15, config)
        down, row = realized_step(state, action, 1., 0., home, False, .15, config)
        self.assertEqual(up.state.stored_kwh, 2.5)
        self.assertLess(down.state.stored_kwh, 2.5)
        self.assertGreaterEqual(down.state.stored_kwh, 1.)
        self.assertAlmostEqual(row["noncritical_shed_kwh"], .6)

    def test_grid_replenishment_and_r0_matches_conventional_physical_dispatch(self):
        home = HomeConfig()
        action = reserve_action(1., 2.5, 1., 0., home)
        result = PhysicalSimulator().step(EnergyState(0, 1.), action, 1., 0., home, repair=True)
        self.assertAlmostEqual(result.state.stored_kwh, 2.5)
        self.assertGreater(result.grid_charge_kwh, 0.)
        for stored, demand, solar in ((1., 2., 0.), (3., .1, 4.), (5., 2., 0.)):
            a = PhysicalSimulator().step(EnergyState(0, stored), reserve_action(stored, 1., demand, solar, home), demand, solar, home, repair=True)
            b = PhysicalSimulator().step(EnergyState(0, stored), SelfConsumptionStrategy().action(demand, solar, home), demand, solar, home, repair=True)
            self.assertAlmostEqual(a.state.stored_kwh, b.state.stored_kwh)
            self.assertAlmostEqual(a.grid_import_kwh, b.grid_import_kwh)

    def test_all_requests_have_identical_emergency_and_explicit_sacrifice(self):
        rows = []
        for command in (EnergyAction(), EnergyAction(charge_kw=100), EnergyAction(discharge_kw=100)):
            result, row = realized_step(EnergyState(0, 1.), command, 2., 0., HomeConfig(), False, .4, ResilienceConfig())
            rows.append(row)
            self.assertEqual(result.grid_import_kwh, 0.)
            self.assertAlmostEqual(row["critical_ens_kwh"], .8)
            self.assertAlmostEqual(row["noncritical_shed_kwh"], 1.2)
            self.assertAlmostEqual(row["critical_served_kwh"] + row["noncritical_served_kwh"] + row["total_unserved_kwh"], 2.)
        self.assertEqual(rows[0], rows[1])
        self.assertEqual(rows[1], rows[2])

    def test_zero_outage_import_solar_charging_no_simultaneous_flows(self):
        result, row = realized_step(EnergyState(0, 2.5), EnergyAction(discharge_kw=99), 1., 3., HomeConfig(), False, .4, ResilienceConfig())
        self.assertEqual(result.grid_import_kwh, 0.)
        self.assertEqual(result.discharge_kwh, 0.)
        self.assertGreater(result.pv_charge_kwh, 0.)
        self.assertEqual(row["critical_ens_kwh"], 0.)

    def test_censoring_and_no_outage_not_invented_as_backup(self):
        home, config = HomeConfig(), ResilienceConfig()
        state, rows = EnergyState(0, 5.), []
        for _ in range(2):
            result, row = realized_step(state, EnergyAction(), .1, 0., home, False, .15, config)
            state = result.state
            rows.append(row)
        metric = resilience_metrics(rows, home, config, 2)
        self.assertEqual(metric["initial_autonomy_hours"], 2)
        self.assertTrue(metric["autonomy_right_censored"])
        self.assertTrue(metric["no_critical_deficit"])
        for row in rows: row["grid_available"] = True
        self.assertIsNone(resilience_metrics(rows, home, config, 2)["initial_autonomy_hours"])
        self.assertIsNone(resilience_metrics(rows, home, config, 2)["no_critical_deficit"])


class SearchGuarantees(unittest.TestCase):
    def test_exact_astar_matches_independent_ucs_and_backward_dynamic_programming(self):
        rng = np.random.default_rng(2026)
        for _ in range(12):
            n = 5
            home = HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.5, horizon_steps=n,
                import_tariff_per_kwh=tuple(rng.choice([.08, .4], n)))
            data = forecast_inputs(rng.uniform(0, 2, n), rng.uniform(0, 1, n), home)
            settings, config = PlannerConfig(storage_step_kwh=.4), ResilienceConfig(risk_hours=2)
            problem = ResilienceProblem(data, EnergyState(0, .5), settings, config)
            plan = ResiliencePlanner(settings, config).plan(data, EnergyState(0, .5))
            serial = count()
            zero = (Fraction(0), Fraction(0))
            queue, best = [(zero, next(serial), EnergyState(0, .5))], {EnergyState(0, .5): zero}
            while queue:
                cost, _, state = heapq.heappop(queue)
                if cost != best[state]: continue
                if problem.is_goal(state): break
                for _, successor, edge in problem.successors(state):
                    candidate = (cost[0] + edge[0], cost[1] + edge[1])
                    if successor not in best or candidate < best[successor]:
                        best[successor] = candidate
                        heapq.heappush(queue, (candidate, next(serial), successor))
            @lru_cache(None)
            def optimum(state):
                if problem.is_goal(state): return zero
                return min((edge[0] + optimum(nxt)[0], edge[1] + optimum(nxt)[1]) for _, nxt, edge in problem.successors(state))
            exact = optimum(EnergyState(0, .5))
            self.assertEqual(cost, exact)
            self.assertAlmostEqual(plan.objective_primary, float(exact[0]))
            self.assertAlmostEqual(plan.predicted_cost, float(exact[1]))
            self.assertEqual(plan.heuristic_name, "zero")

    def test_true_outage_masks_are_rejected_by_search_contract(self):
        home = HomeConfig(horizon_steps=5)
        data = forecast_inputs([1.] * 5, [0.] * 5, home)
        with self.assertRaises(ValueError):
            ResiliencePlanner(resilience=ResilienceConfig(risk_hours=2)).plan(replace(data, grid_available=(True, False, True, True, True)), EnergyState(0, 2.5))

    def test_unproven_plan_raises_instead_of_silent_fallback(self):
        home = HomeConfig(horizon_steps=5)
        data = forecast_inputs([1.] * 5, [0.] * 5, home)
        with self.assertRaises(PlanningLimitError):
            ResiliencePlanner(PlannerConfig(max_expanded=1), ResilienceConfig(risk_hours=2)).plan(data, EnergyState(0, 2.5))


class IntegrationCausalityTests(unittest.TestCase):
    def test_predeclared_matrix_sizes_and_bounds(self):
        self.assertEqual(len(scenarios(168, "full", "flat")), 10)
        self.assertEqual(len(scenarios(168, "pilot", "flat")), 0)
        self.assertEqual(len(scenarios(168, "pilot", "time_of_use")), 2)
        self.assertEqual(len(POLICIES), 6)

    def test_future_outages_and_restoration_cannot_change_prior_actions_all_policies(self):
        base, config = fixture(), ResilienceConfig()
        settings = PlannerConfig(storage_step_kwh=.4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = PlanStore(root / "plans")
            for policy in POLICIES:
                replay_policy(base, ("short", 2, 1), policy, settings, config, store, root / f"{policy}_short")
                replay_policy(base, ("long", 2, 3), policy, settings, config, store, root / f"{policy}_long")
                replay_policy(base, ("future", 5, 1), policy, settings, config, store, root / f"{policy}_future")
                a = pd.read_csv(root / f"{policy}_short" / "ledger.csv")
                b = pd.read_csv(root / f"{policy}_long" / "ledger.csv")
                c = pd.read_csv(root / f"{policy}_future" / "ledger.csv")
                cols = ["stored_kwh", "requested_charge_kw", "requested_discharge_kw"]
                np.testing.assert_allclose(a[cols].iloc[:2], c[cols].iloc[:2])
                np.testing.assert_allclose(a[cols].iloc[:3], b[cols].iloc[:3])
                self.assertTrue((a[a.grid_available == False].grid_import_kwh == 0).all())

    def test_failed_or_incomplete_run_is_explicit_not_zero_ens_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metric = replay_policy(fixture(), ("cut", 2, 1), "astar_mlp", PlannerConfig(max_expanded=1), ResilienceConfig(), PlanStore(root / "plans"), root / "failed")
            self.assertEqual(metric["status"], "failed")
            self.assertEqual(metric["hours_completed"], 0)
            self.assertIsNotNone(json.loads((root / "failed" / "failure.json").read_text()))

    def test_energy_persists_over_midnight_without_reset(self):
        base = fixture(26)
        base["case"]["actual_demand"] = [.01] * 26
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            replay_policy(base, ("none", None, 0), "reactive", PlannerConfig(), ResilienceConfig(), PlanStore(root / "plans"), root / "run")
            frame = pd.read_csv(root / "run" / "ledger.csv")
            self.assertAlmostEqual(frame.initial_kwh.iloc[24], frame.stored_kwh.iloc[23])
            self.assertNotAlmostEqual(frame.initial_kwh.iloc[24], .5)


if __name__ == "__main__":
    unittest.main()
