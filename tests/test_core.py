from dataclasses import replace
import math
from pathlib import Path
import unittest

from bekantor_energy.config import HomeConfig
from bekantor_energy.units import energy_kwh, soc_from_energy, watts_to_kw, wh_to_kwh


class UnitsTests(unittest.TestCase):
    def test_power_energy_and_soc(self):
        self.assertEqual(watts_to_kw(2000), 2)
        self.assertEqual(wh_to_kwh(500), 0.5)
        self.assertEqual(energy_kwh(2, 0.5), 1)
        self.assertEqual(soc_from_energy(1, 5), 0.2)

    def test_impossible_units_rejected(self):
        for value in (-1, math.inf, math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                watts_to_kw(value)
        for stored, capacity in ((2, 1), (-1, 5), (0, 0), (0, math.inf)):
            with self.subTest(stored=stored, capacity=capacity), self.assertRaises(ValueError):
                soc_from_energy(stored, capacity)


class ConfigTests(unittest.TestCase):
    def test_example_load_and_flat_tariffs(self):
        path = Path(__file__).resolve().parents[1] / "configs/home.example.json"
        home = HomeConfig.load(path)
        self.assertEqual(len(home.import_prices), 24)
        self.assertEqual(home.import_prices[0], 0.15)

    def test_time_of_use_tariffs(self):
        home = HomeConfig(horizon_steps=3, import_tariff_per_kwh=(1, 2, 3))
        self.assertEqual(home.import_prices, (1, 2, 3))
        self.assertEqual(home.export_prices, (0, 0, 0))

    def test_invalid_physics_and_tariffs(self):
        cases = ({"battery_capacity_kwh": 0}, {"max_charge_kw": -1},
                 {"discharge_efficiency": 1.1}, {"charge_efficiency": 0},
                 {"initial_soc": 0.1}, {"min_soc": math.nan},
                 {"pv_peak_kwp": math.inf}, {"horizon_steps": 2.5},
                 {"import_tariff_per_kwh": (1, 2)},
                 {"import_tariff_per_kwh": (math.nan,)},
                 {"export_tariff_per_kwh": (1,)}, {"currency": ""})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(HomeConfig(), **kwargs)
