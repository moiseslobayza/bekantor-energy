"""Independent import-bound and diagnostic regressions; synthetic data only."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyState, LoadForecast, PlanningInputs
from bekantor_energy.energy.autonomy_diagnostics import (
    import_lower_bound, ledger_diagnostics, oracle_windows, replay_diagnostic, stock_adjusted_import,
)
from bekantor_energy.energy.modes import ModePlanner, ObjectiveConfig
from bekantor_energy.energy.planner import PlannerConfig
from bekantor_energy.energy.simulator import PhysicalSimulator, SelfConsumptionStrategy


def reactive(demand, solar, home):
    simulator, policy = PhysicalSimulator(), SelfConsumptionStrategy()
    state, rows = EnergyState(0, home.initial_soc * home.battery_capacity_kwh), []
    for demand_t, solar_t in zip(demand, solar):
        result = simulator.step(state, policy.action(demand_t, solar_t, home), demand_t, solar_t, home, repair=True)
        row = asdict(result)
        row.update(initial_kwh=state.stored_kwh, stored_kwh=result.state.stored_kwh)
        rows.append(row)
        state = result.state
    return pd.DataFrame(rows)


class ImportBoundTests(unittest.TestCase):
    def test_no_storage_supply_and_known_efficiency_conversion(self):
        home = HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.2)
        self.assertAlmostEqual(import_lower_bound([1., 1.], [0., 0.], home)["import_lower_bound_kwh"], 2.)
        bound = import_lower_bound([0., 1.], [.5, 0.], home)
        self.assertAlmostEqual(bound["import_lower_bound_kwh"], 1 - .5 * .95 ** 2)
        self.assertAlmostEqual(bound["max_deliverable_kwh"], bound["min_cut_kwh"])

    def test_chronology_prevents_future_solar_serving_past_load(self):
        home = HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.2)
        self.assertEqual(import_lower_bound([1., 0.], [0., 1.], home)["import_lower_bound_kwh"], 1.)

    def test_capacity_charge_and_discharge_power_are_independent_limits(self):
        base = HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.2,
                          charge_efficiency=1., discharge_efficiency=1.)
        self.assertAlmostEqual(import_lower_bound([0., 2.], [3., 0.], base)["import_lower_bound_kwh"], 1.2)
        self.assertAlmostEqual(import_lower_bound([0., 2.], [3., 0.], replace(base, max_charge_kw=.1))["import_lower_bound_kwh"], 1.9)
        self.assertAlmostEqual(import_lower_bound([0., 2.], [3., 0.], replace(base, max_discharge_kw=.2))["import_lower_bound_kwh"], 1.8)

    def test_initial_energy_and_nonunit_step_hours(self):
        home = HomeConfig(battery_capacity_kwh=2, min_soc=.25, initial_soc=.75,
                          discharge_efficiency=.8, step_hours=.5, max_discharge_kw=.2)
        self.assertAlmostEqual(import_lower_bound([1., 1.], [0., 0.], home)["import_lower_bound_kwh"], 1.8)

    def test_independent_reactive_replay_matches_bound_on_random_scenarios(self):
        rng = np.random.default_rng(2026)
        for _ in range(120):
            home = HomeConfig(battery_capacity_kwh=float(rng.uniform(.5, 5)), min_soc=.2,
                initial_soc=float(rng.uniform(.2, 1)), max_charge_kw=float(rng.uniform(.1, 3)),
                max_discharge_kw=float(rng.uniform(.1, 3)), charge_efficiency=float(rng.uniform(.7, 1)),
                discharge_efficiency=float(rng.uniform(.7, 1)))
            demand, solar = rng.uniform(0, 3, (2, 36))
            actual = reactive(demand, solar, home)
            bound = import_lower_bound(demand, solar, home)
            self.assertAlmostEqual(actual.grid_import_kwh.sum(), bound["import_lower_bound_kwh"], places=8)
            self.assertAlmostEqual(bound["flow_cut_residual_kwh"], 0., places=8)

    def test_invalid_series_and_initial_reserve_are_rejected(self):
        home = HomeConfig()
        for demand, solar in (([], []), ([1.], []), ([float("nan")], [0.]), ([-1.], [0.])):
            with self.assertRaises(ValueError):
                import_lower_bound(demand, solar, home)
        with self.assertRaises(ValueError):
            import_lower_bound([1.], [0.], replace(home, initial_soc=.1, min_soc=.2))

    def test_perfect_forecast_grid_can_force_grid_charging_and_lose_to_feedback(self):
        home = HomeConfig(battery_capacity_kwh=1, min_soc=.2, initial_soc=.2, horizon_steps=2)
        demand, solar = [0., 1.], [.1, 0.]
        times = tuple(pd.date_range("2009-01-01", periods=2, freq="h").to_pydatetime())
        inputs = PlanningInputs(LoadForecast(times[0], times, tuple(demand), "ORACLE_FIXTURE"), tuple(solar), home, "synthetic")
        actual = reactive(demand, solar, home)
        imports, grid_charges = [], []
        for spacing in (.25, .125):
            plan = ModePlanner(PlannerConfig(storage_step_kwh=spacing), ObjectiveConfig("autonomy")).plan(inputs, EnergyState(0, .2))
            state, total, grid_charge = EnergyState(0, .2), 0., 0.
            for t, action in enumerate(plan.actions):
                result = PhysicalSimulator().step(state, action, demand[t], solar[t], home)
                total += result.grid_import_kwh
                grid_charge += result.grid_charge_kwh
                state = result.state
            imports.append(total)
            grid_charges.append(grid_charge)
        self.assertGreater(grid_charges[0], 0.)
        self.assertGreater(imports[0], actual.grid_import_kwh.sum() + .01)
        self.assertLess(imports[1], imports[0])
        self.assertGreaterEqual(imports[1] + 1e-9, import_lower_bound(demand, solar, home)["import_lower_bound_kwh"])


class DiagnosticBoundaryTests(unittest.TestCase):
    def test_oracle_targets_are_exact_and_need_observed_tail(self):
        times = pd.date_range("2009-01-01", periods=25, freq="h")
        frame = pd.DataFrame({"load_kwh": np.arange(25.), "is_observed_complete": True}, index=times)
        values = oracle_windows(frame, times[0], 2)
        np.testing.assert_array_equal(values[0], np.arange(24.))
        np.testing.assert_array_equal(values[1], np.arange(1., 25.))
        frame.iloc[-1, frame.columns.get_loc("is_observed_complete")] = False
        with self.assertRaises(ValueError):
            oracle_windows(frame, times[0], 2)

    def test_stock_accounting_sign_and_identical_terminal_energy(self):
        self.assertEqual(stock_adjusted_import(10., 2., 1., .95), 10.95)
        self.assertEqual(stock_adjusted_import(10., 2., 3., .95), 9.05)
        for value in (0., .95, 1 / .95):
            self.assertAlmostEqual(stock_adjusted_import(12., 2.5, 1., value) - stock_adjusted_import(10., 2.5, 1., value), 2.)
        with self.assertRaises(ValueError):
            stock_adjusted_import(10., 2., 1., -1.)

    def test_reactive_has_no_grid_charging_or_missed_instant_opportunities(self):
        home = HomeConfig()
        ledger = reactive([1., .2, 3.], [0., 4., 0.], home)
        metrics = ledger_diagnostics(ledger, home)
        self.assertAlmostEqual(metrics["grid_charge_kwh"], 0.)
        self.assertAlmostEqual(metrics["unused_instant_discharge_kwh"], 0.)
        self.assertAlmostEqual(metrics["unused_instant_pv_charge_kwh"], 0.)

    def test_wrapper_only_injects_labelled_forecasts_and_resolution(self):
        from test_energy_modes import mode_fixture
        case = mode_fixture()
        frozen = json.dumps(case, sort_keys=True)
        with patch("bekantor_energy.energy.autonomy_diagnostics.run_continuous_case", return_value={"ok": True}) as run:
            with tempfile.TemporaryDirectory() as temporary:
                result = replay_diagnostic(case, [[.7] * 24] * 2, .125, "oracle", Path(temporary))
            modified = run.call_args.args[0]
            policy = run.call_args.kwargs["planning_policies"]["autonomy"]
            self.assertEqual(result, {"ok": True})
            self.assertEqual(modified["model_id"], "ORACLE_DIAGNOSTIC_NOT_PRODUCTION")
            self.assertEqual(modified["predictions"][0][0], .7)
            self.assertEqual(policy.objective, ObjectiveConfig("autonomy"))
            self.assertEqual(policy.settings.max_seconds, case["planner"]["max_seconds"])
            self.assertEqual(policy.settings.storage_step_kwh, .125)
        self.assertEqual(json.dumps(case, sort_keys=True), frozen)
        with self.assertRaises(ValueError):
            replay_diagnostic(case, [], .25, "unknown", Path("unused"))


if __name__ == "__main__":
    unittest.main()
