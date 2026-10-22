from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.forecasting.baselines import seasonal_persistence
from bekantor_energy.forecasting.config import Architecture, ExperimentConfig
from bekantor_energy.forecasting.metrics import demand_metrics
from bekantor_energy.forecasting.normalization import DemandNormalizer
from bekantor_energy.forecasting.windows import make_windows, read_hourly, validate_hourly


def hourly_frame(n=900):
    return pd.DataFrame({"load_kwh": np.arange(n, dtype=float), "is_observed_complete": True},
                         index=pd.date_range("2007-01-01", periods=n, freq="h", name="timestamp"))


class WindowTests(unittest.TestCase):
    def test_exact_168_to_24_and_no_future_input(self):
        windows = make_windows(hourly_frame(500))
        self.assertEqual(windows.x.shape, (309, 168))
        self.assertEqual(windows.y.shape, (309, 24))
        np.testing.assert_array_equal(windows.x[0], np.arange(168))
        np.testing.assert_array_equal(windows.y[0], np.arange(168, 192))
        self.assertEqual(windows.origins[0], pd.Timestamp("2007-01-08"))

    def test_validation_context_and_targets_stay_in_partition(self):
        frame = hourly_frame()
        windows = make_windows(frame.iloc[500:700], history=frame.iloc[:500])
        self.assertEqual(len(windows.x), 177)
        np.testing.assert_array_equal(windows.x[0], np.arange(332, 500))
        self.assertEqual(windows.y.min(), 500)
        self.assertEqual(windows.y.max(), 699)

    def test_imputed_targets_excluded_but_known_past_can_be_input(self):
        frame = hourly_frame(500)
        frame.iloc[200, frame.columns.get_loc("is_observed_complete")] = False
        windows = make_windows(frame)
        self.assertTrue(((windows.y < 200) | (windows.y > 200)).all())
        self.assertTrue((windows.x == 200).any())
        self.assertEqual(windows.report["unknown_targets"], 24)

    def test_unknown_energy_invalidates_inputs_and_targets_without_fill(self):
        frame = hourly_frame(500)
        frame.iloc[200] = [np.nan, False]
        windows = make_windows(frame)
        self.assertFalse(np.isnan(windows.x).any())
        self.assertFalse(np.isnan(windows.y).any())
        self.assertEqual(windows.report["unknown_inputs"], 168)
        self.assertEqual(windows.report["unknown_targets"], 24)

    def test_missing_timestamp_does_not_compress_time(self):
        full = hourly_frame(500)
        frame = full.drop(full.index[200])
        windows = make_windows(frame)
        self.assertEqual(windows.report["unknown_inputs"], 168)
        self.assertEqual(windows.report["unknown_targets"], 24)
        for row in windows.x:
            np.testing.assert_array_equal(np.diff(row), np.ones(167))

    def test_history_overlap_rejected(self):
        frame = hourly_frame()
        with self.assertRaises(ValueError):
            make_windows(frame.iloc[500:], history=frame.iloc[:501])

    def test_changes_after_origin_do_not_change_its_inputs(self):
        original = hourly_frame(500)
        changed = original.copy()
        changed.loc[changed.index[200]:, "load_kwh"] *= 1000
        before, after = make_windows(original), make_windows(changed)
        keep = before.origins <= original.index[200]
        np.testing.assert_array_equal(before.x[keep], after.x[keep])

    def test_small_partition_and_stride(self):
        self.assertEqual(make_windows(hourly_frame(10)).x.shape, (0, 168))
        windows = make_windows(hourly_frame(500), stride=24)
        self.assertTrue((np.diff(windows.origins.to_numpy()) == np.timedelta64(24, "h")).all())

    def test_bad_flags_and_times_rejected(self):
        frame = hourly_frame(500)
        frame["is_observed_complete"] = "False"
        with self.assertRaises(ValueError):
            validate_hourly(frame)
        for value in (hourly_frame().iloc[::-1], pd.concat([hourly_frame(), hourly_frame()])):
            with self.assertRaises(ValueError):
                validate_hourly(value)

    def test_csv_boolean_false_is_not_truthy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            frame = hourly_frame(500)
            frame.iloc[200] = [np.nan, False]
            frame.to_csv(path)
            loaded = read_hourly(path)
            self.assertFalse(loaded.is_observed_complete.iloc[200])
            self.assertTrue(np.isnan(loaded.load_kwh.iloc[200]))


class NormalizationTests(unittest.TestCase):
    def test_fitted_only_on_unique_observed_train_values(self):
        train = hourly_frame(500)
        train.iloc[100] = [10000, False]
        scaler = DemandNormalizer.fit(train)
        expected = np.delete(np.arange(500), 100)
        self.assertAlmostEqual(scaler.mean_kwh, expected.mean())
        self.assertAlmostEqual(scaler.scale_kwh, expected.std())
        self.assertEqual(scaler.fitted_hours, 499)
        self.assertLess(pd.Timestamp(scaler.fitted_end), hourly_frame(900).index[500])

    def test_inverse_targets_and_persistence_roundtrip(self):
        scaler = DemandNormalizer.fit(hourly_frame())
        data = np.array([[0, 1, 10]], dtype=float)
        np.testing.assert_allclose(scaler.inverse_targets(scaler.targets(data)), data, atol=1e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "normalization.json"
            scaler.save(path)
            self.assertEqual(DemandNormalizer.load(path), scaler)

    def test_constant_and_no_observed_training_data(self):
        frame = hourly_frame(500)
        frame.load_kwh = 2.0
        scaler = DemandNormalizer.fit(frame)
        self.assertEqual(scaler.scale_kwh, 1)
        np.testing.assert_array_equal(scaler.inputs(np.array([2.])), np.array([0.]))
        frame.is_observed_complete = False
        with self.assertRaises(ValueError):
            DemandNormalizer.fit(frame)


class BaselineMetricConfigTests(unittest.TestCase):
    def test_exact_seasonal_lag_alignment(self):
        history = np.arange(168)[None, :]
        np.testing.assert_array_equal(seasonal_persistence(history, lag=24)[0], np.arange(144, 168))
        np.testing.assert_array_equal(seasonal_persistence(history, lag=168)[0], np.arange(24))
        with self.assertRaises(ValueError):
            seasonal_persistence(history, lag=12)

    def test_metrics_physical_units_and_horizon(self):
        actual = np.array([[1., 2.], [3., 4.]])
        scores = demand_metrics(actual, actual + np.array([1., 2.]))
        self.assertEqual(scores["mae_kwh"], 1.5)
        self.assertAlmostEqual(scores["rmse_kwh"], np.sqrt(2.5))
        self.assertEqual(scores["by_horizon"][1]["mae_kwh"], 2)
        self.assertEqual(scores["evaluated_pairs"], 4)
        with self.assertRaises(ValueError):
            demand_metrics(actual, actual * np.nan)

    def test_reference_and_invalid_configs(self):
        config = ExperimentConfig.load(Path(__file__).parents[1] / "configs/mlp.reference.json")
        self.assertEqual([a.hidden_units for a in config.architectures], [(64, 32), (128, 64), (256, 128, 64)])
        self.assertEqual(ExperimentConfig(**config.to_dict()), config)
        for values in ({"learning_rate": float("nan")}, {"lookback": 24}, {"horizon": 25}, {"epochs": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(config, **values)
        with self.assertRaises(ValueError):
            Architecture("../unsafe", (32,))
