from dataclasses import asdict, replace
from datetime import timedelta
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.config import HomeConfig
from bekantor_energy.contracts import EnergyAction, EnergyState, LoadForecast
from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.data.pvgis import PVGISQuery
from bekantor_energy.energy.experiment import CausalForecastProvider
from bekantor_energy.energy.planner import AStarPlanner, PlannerConfig
from bekantor_energy.energy.scenarios import first_complete_midnight, prepare_solar_scenario
from bekantor_energy.energy.simulator import PhysicalSimulator


class ScenarioAndCausalityTests(unittest.TestCase):
    def fixture(self, root, count=48):
        query = PVGISQuery(latitude=-31.4201, longitude=-64.1888, start="2023-09-21T03:00:00",
                           end=(pd.Timestamp("2023-09-21T03:00:00") + pd.Timedelta(hours=count - 1)).isoformat())
        times = pd.date_range(*query.bounds, freq="h")
        path = root / "synthetic_pvgis_fixture.json"
        write_json(path, {"power": [1000.] * count, "timestamps": [t.isoformat() for t in times],
                          "peak_power_unit": "kWp", "peak_power": 3.})
        write_json(path.with_suffix(".metadata.json"), {"sha256": sha256(path), "query": asdict(query),
                   "source_kind": "synthetic_fixture_not_downloaded"})
        return path

    def test_solar_source_preserves_both_clocks_units_and_no_double_efficiency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self.fixture(root)
            frame, metadata = prepare_solar_scenario(raw, HomeConfig(), root / "scenario")
            self.assertEqual(len(frame), 48)
            self.assertEqual(frame.pvgis_timestamp_cordoba.iloc[0].hour, 0)
            self.assertEqual(frame.pvgis_timestamp_utc.iloc[0].hour, 3)
            self.assertTrue((frame.solar_kwh == 1).all())  # Not multiplied by 0.86 again.
            self.assertIn("not synchronized", metadata["synchronization"])
            self.assertEqual(metadata["query"]["latitude"], -31.4201)
            self.assertTrue((root / "scenario" / "solar_scenario.csv").exists())
            with self.assertRaises(FileExistsError):
                prepare_solar_scenario(raw, HomeConfig(), root / "scenario")

    def test_solar_checksum_size_and_coverage_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = self.fixture(root)
            with self.assertRaises(ValueError):
                prepare_solar_scenario(raw, replace(HomeConfig(), pv_peak_kwp=4))
            raw.write_text("{}")
            with self.assertRaises(ValueError):
                prepare_solar_scenario(raw, HomeConfig())
            raw = self.fixture(root, 24)
            with self.assertRaises(ValueError):
                prepare_solar_scenario(raw, HomeConfig())

    def test_first_window_filter_uses_completeness_not_demand_size(self):
        times = pd.date_range("2009-09-21", periods=72, freq="h")
        actual = pd.DataFrame({"load_kwh": 1., "is_observed_complete": True}, index=times)
        self.assertEqual(first_complete_midnight(times, actual), times[0])
        actual.iloc[0, actual.columns.get_loc("is_observed_complete")] = False
        self.assertEqual(first_complete_midnight(times, actual), times[24])
        actual.load_kwh = 999.
        self.assertEqual(first_complete_midnight(times, actual), times[24])

    def test_neural_port_never_receives_future_actuals_or_calls_fit(self):
        times = pd.date_range("2009-09-01", periods=240, freq="h")
        frame = pd.DataFrame({"load_kwh": 1., "is_observed_complete": True}, index=times)
        seen = []

        class Predictor:
            def fit(self, *args):
                raise AssertionError("Neural fitting is prohibited in Phase 3")

            def predict(self, history, starts):
                seen.append(history.index[-1])
                self.last = history.load_kwh.iloc[-1]
                return LoadForecast(starts[0], tuple(starts), (float(self.last),) * 24, "fixture")

        model = Predictor()
        provider = CausalForecastProvider(frame, model)
        before = provider(times[200])
        frame.loc[times[200]:, "load_kwh"] = 500.
        after = provider(times[200])
        self.assertEqual(before, after)
        self.assertTrue(all(t < times[200] for t in seen))
        self.assertEqual(pd.Timestamp(provider.audit[0]["last_history_timestamp"]), times[199])

    def test_all_capacity_reserved_is_a_valid_disabled_battery_case(self):
        c = HomeConfig(initial_soc=1, min_soc=1)
        result = PhysicalSimulator().step(EnergyState(0, 5), EnergyAction(discharge_kw=2), 3, 0, c, repair=True)
        self.assertEqual(result.grid_import_kwh, 3)
        self.assertEqual(result.state.stored_kwh, 5)

    def test_zero_demand_and_solar_zero_price_never_creates_energy(self):
        c = HomeConfig(import_tariff_per_kwh=(0.,))
        state = EnergyState(0, 2.5)
        for _ in range(24):
            result = PhysicalSimulator().step(state, EnergyAction(), 0, 0, c)
            self.assertEqual(result.state.stored_kwh, 2.5)
            self.assertEqual(result.cost, 0)
            state = result.state

    def test_unreachable_terminal_requirement_and_excessive_grid_are_rejected(self):
        from test_energy import inputs
        data = inputs([0], battery_capacity_kwh=2, initial_soc=.25, min_soc=0, max_charge_kw=.001)
        with self.assertRaisesRegex(ValueError, "No feasible plan"):
            AStarPlanner(PlannerConfig(terminal_min_soc=1)).plan(data, EnergyState(0, .5))
        with self.assertRaises(ValueError):
            AStarPlanner(PlannerConfig(storage_step_kwh=1e-9)).plan(data, EnergyState(0, .5))
