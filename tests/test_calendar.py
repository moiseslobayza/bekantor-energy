from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.forecasting.calendar import calendar_features, with_calendar, CalendarInputNormalizer
from bekantor_energy.forecasting.calendar_experiment import (
    CalendarComparisonConfig, aggregate_metrics, metric_values, peak_thresholds, validation_rows)
from bekantor_energy.forecasting.config import Architecture, ExperimentConfig
from bekantor_energy.forecasting.normalization import DemandNormalizer
from bekantor_energy.forecasting.windows import make_windows, WindowSet


def frame(n=400):
    return pd.DataFrame({"load_kwh": 1 + np.arange(n) / n, "is_observed_complete": True},
                        index=pd.date_range("2007-12-25", periods=n, freq="h", name="timestamp"))


class CalendarTests(unittest.TestCase):
    def test_cycles_align_to_each_future_target_across_boundaries(self):
        origins = pd.DatetimeIndex(["2023-12-31 23:00"])
        values = calendar_features(origins, 24)
        self.assertEqual(values.shape, (1, 24, 6))
        # Jan 1 2024: midnight, Monday, January -> all phases zero.
        np.testing.assert_allclose(values[0, 1], [0, 1, 0, 1, 0, 1], atol=1e-6)
        np.testing.assert_allclose(values[0, 0, :2], [np.sin(23 * np.pi / 12), np.cos(23 * np.pi / 12)], atol=1e-6)
        np.testing.assert_allclose(values[0, 0, 2:4], [np.sin(6 * 2 * np.pi / 7), np.cos(6 * 2 * np.pi / 7)], atol=1e-6)
        np.testing.assert_allclose(values[0, 0, 4:], [-0.5, np.sqrt(3) / 2], atol=1e-6)

    def test_calendar_is_known_independent_of_future_demand(self):
        original = frame()
        changed = original.copy()
        changed.iloc[168:, changed.columns.get_loc("load_kwh")] *= 100
        before, after = with_calendar(make_windows(original)), with_calendar(make_windows(changed))
        np.testing.assert_array_equal(before.x[0], after.x[0])
        self.assertFalse(np.array_equal(before.y[0], after.y[0]))

    def test_calendar_preserves_windows_and_unknown_target_policy(self):
        data = frame()
        data.iloc[210] = [np.nan, False]
        raw = make_windows(data)
        result = with_calendar(raw)
        self.assertEqual(result.x.shape[1], 312)
        np.testing.assert_array_equal(raw.x, result.x[:, :168])
        np.testing.assert_array_equal(raw.y, result.y)
        self.assertTrue(raw.origins.equals(result.origins))
        self.assertFalse(np.isnan(result.x).any())
        self.assertEqual(with_calendar(make_windows(frame(10))).x.shape, (0, 312))

    def test_normalization_changes_only_past_demand(self):
        data = frame()
        demand = DemandNormalizer.fit(data.iloc[:250])
        windows = with_calendar(make_windows(data))
        normalizer = CalendarInputNormalizer(demand, 168, 24)
        result = normalizer.inputs(windows.x)
        np.testing.assert_array_equal(result[:, :168], demand.inputs(windows.x[:, :168]))
        np.testing.assert_array_equal(result[:, 168:], windows.x[:, 168:])
        np.testing.assert_allclose(normalizer.inverse_targets(normalizer.targets(windows.y)), windows.y, atol=1e-6)
        with self.assertRaises(ValueError):
            normalizer.inputs(windows.x[:, :-1])
        bad = windows.x.copy()
        bad[0, -1] = 2
        with self.assertRaises(ValueError):
            normalizer.inputs(bad)

    def test_invalid_calendar_origins(self):
        for origin in (pd.DatetimeIndex([pd.NaT]), pd.DatetimeIndex(["2007-01-01 00:01"])):
            with self.assertRaises(ValueError):
                calendar_features(origin)
        with self.assertRaises(ValueError):
            calendar_features(pd.DatetimeIndex([]), 0)

    def test_thresholds_use_only_unique_observed_training_hours(self):
        data = frame()
        data.iloc[-1] = [999, False]
        result = peak_thresholds(data, (0.9, 0.95))
        self.assertAlmostEqual(result["q90"], np.quantile(data.load_kwh.iloc[:-1], 0.9))
        self.assertAlmostEqual(result["q95"], np.quantile(data.load_kwh.iloc[:-1], 0.95))

    def test_physical_metrics_sign_strict_peak_and_target_hour(self):
        windows = WindowSet(np.zeros((1, 168)), np.array([[1., 3., 4.]]),
                            pd.DatetimeIndex(["2007-01-01 23:00"]), {})
        predictions = np.array([[2., 2., 4.]])
        rows = pd.DataFrame(validation_rows(windows, predictions, {"q90": 3.}, 1, "reference"))
        high = rows[rows.scope == "high_consumption"]
        self.assertTrue((high.n_pairs == 1).all())  # Equality at 3 excluded.
        self.assertEqual(high[high.metric == "underestimate_pct"].value.iloc[0], 0)
        hour_zero = rows[(rows.scope == "hour") & (rows.group == "0") & (rows.metric == "bias_kwh")]
        self.assertEqual(hour_zero.value.iloc[0], -1)
        values = metric_values(windows.y, predictions)
        self.assertAlmostEqual(values["mae_kwh"], 2 / 3)
        self.assertAlmostEqual(values["rmse_kwh"], np.sqrt(2 / 3))
        self.assertAlmostEqual(values["underestimate_pct"], 100 / 3)

    def test_seed_aggregation_and_paired_deltas(self):
        rows = [{"seed": seed, "variant": variant, "scope": "global", "group": "all",
                 "metric": "mae_kwh", "value": seed + delta, "n_pairs": 5}
                for seed in (1, 2, 3) for variant, delta in (("reference", 0), ("calendar", -0.5))]
        _, aggregate, paired = aggregate_metrics(rows)
        self.assertEqual(aggregate.loc[aggregate.variant == "reference", "mean"].iloc[0], 2)
        self.assertEqual(aggregate.loc[aggregate.variant == "reference", "std"].iloc[0], 1)
        np.testing.assert_array_equal(paired.delta_calendar_minus_reference, [-0.5] * 3)
        with self.assertRaises(ValueError):
            aggregate_metrics(rows[:-1])

    def test_config_fixes_ablation_and_requires_three_distinct_seeds(self):
        config = CalendarComparisonConfig.load(Path(__file__).parents[1] / "configs/mlp.calendar-comparison.json")
        self.assertEqual(config.training.architectures[0].hidden_units, (64, 32))
        for seeds in ((1, 2), (1, 1, 2), (1, 2, -1)):
            with self.assertRaises(ValueError):
                replace(config, seeds=seeds)
        for arch in (Architecture("small", (128, 32)), Architecture("small", (64, 32), dropout=0.1)):
            with self.assertRaises(ValueError):
                replace(config, training=replace(config.training, architectures=(arch,)))
