"""Loss algebra and small deterministic CPU fits, only train and validation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.forecasting.calendar_experiment import CalendarComparisonConfig, run_calendar_comparison
from bekantor_energy.forecasting.config import Architecture, ExperimentConfig
from bekantor_energy.forecasting.mlp import MLPDemandPredictor, build_model, configure_tensorflow
from bekantor_energy.forecasting.weighted_experiment import WeightedComparisonConfig, run_weighted_comparison
from bekantor_energy.forecasting.weighted_loss import make_peak_weighted_mse


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow is absent")
class WeightedTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = configure_tensorflow(2026, 2)

    def test_weights_each_of_24_targets_and_gradients(self):
        loss = make_peak_weighted_mse(3., 2.)
        y = np.ones((1, 24), dtype=np.float32)
        y[0, -1] = 2.  # Physical demand 4, over physical threshold 3.
        predicted = self.tf.Variable(np.zeros((1, 24), dtype=np.float32))
        with self.tf.GradientTape() as tape:
            value = loss(y, predicted)
        gradient = tape.gradient(value, predicted).numpy()
        self.assertAlmostEqual(float(value), (23 + 8) / 24, places=6)
        np.testing.assert_allclose(gradient[0, :-1], -2 / 24, atol=1e-7)
        self.assertAlmostEqual(float(gradient[0, -1]), -8 / 24, places=6)

    def test_scaled_units_strict_threshold_and_batch_reduction(self):
        loss = make_peak_weighted_mse(3., 2.)
        self.assertEqual(loss.threshold_scaled, 1.5)
        actual_kwh = np.array([[2., 3.], [4., 6.]], dtype=np.float32)
        predicted_kwh = np.array([[3., 4.], [3., 4.]], dtype=np.float32)
        expected_physical = np.mean(np.where(actual_kwh > 3, 2, 1) * (actual_kwh - predicted_kwh) ** 2)
        actual_loss = float(loss(actual_kwh / 2, predicted_kwh / 2))
        self.assertAlmostEqual(actual_loss * 2 ** 2, expected_physical, places=6)
        # Exactly the threshold gets weight 1, determined by actual, not prediction.
        self.assertAlmostEqual(float(loss(np.array([[1.5]], np.float32), np.array([[2.5]], np.float32))), 1)
        for scale in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                make_peak_weighted_mse(3., scale)

    def test_same_initial_weights_and_loss_serialization(self):
        config = ExperimentConfig(architectures=(Architecture("small", (64, 32)),))
        mse = build_model(config, config.architectures[0], input_size=312)
        initial = mse.get_weights()
        loss = make_peak_weighted_mse(2.3914, .9278)
        weighted = build_model(config, config.architectures[0], input_size=312, loss=loss)
        self.assertEqual(weighted.count_params(), 22904)
        for a, b in zip(initial, weighted.get_weights()):
            np.testing.assert_array_equal(a, b)
        serialized = self.tf.keras.losses.serialize(loss)
        restored = self.tf.keras.losses.deserialize(serialized)
        self.assertEqual(restored.get_config(), loss.get_config())
        self.assertAlmostEqual(float(weighted.optimizer.learning_rate), float(mse.optimizer.learning_rate))

    def test_three_seeds_validation_only_repeatability_and_checkpoint_roundtrip(self):
        import bekantor_energy.forecasting.weighted_experiment as experiment
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            n = 360
            frame = pd.DataFrame({"load_kwh": 1 + .5 * np.sin(np.arange(n) * 2 * np.pi / 24),
                                  "is_observed_complete": True},
                                 index=pd.date_range("2007-01-01", periods=n, freq="h", name="timestamp"))
            manifest = {"partitions": {}}
            for name, part in {"train": frame.iloc[:270], "validation": frame.iloc[270:]}.items():
                path = data / f"{name}.csv"
                part.to_csv(path)
                manifest["partitions"][name] = {"sha256": sha256(path), "hours": len(part),
                    "start": part.index[0].isoformat(), "end": part.index[-1].isoformat()}
            write_json(data / "manifest.json", manifest)
            training = ExperimentConfig(epochs=1, patience=1, architectures=(Architecture("small", (64, 32)),))
            reference_run, reference_models = root / "calendar_run", root / "calendar_models"
            run_calendar_comparison(data, CalendarComparisonConfig(training), reference_run, reference_models)
            config = WeightedComparisonConfig()
            with patch.object(experiment, "_checked_partition", wraps=experiment._checked_partition) as reader:
                run_weighted_comparison(data, config, root / "run", root / "models", reference_run, reference_models)
            self.assertEqual([call.args[1] for call in reader.call_args_list], ["train", "validation"])
            run_weighted_comparison(data, config, root / "repeat", root / "repeat_models", reference_run, reference_models)
            first = pd.read_csv(root / "run" / "metrics_by_seed.csv")
            repeated = pd.read_csv(root / "repeat" / "metrics_by_seed.csv")
            np.testing.assert_allclose(first.value, repeated.value, atol=1e-7)
            self.assertTrue(json.loads((root / "run" / "preservation.json").read_text())["verified"])
            directory = root / "models" / "seed_2026" / "weighted_mse"
            loaded = self.tf.keras.models.load_model(directory / "best.keras")
            self.assertEqual(loaded.loss.normal_weight, 1)
            self.assertEqual(loaded.loss.peak_weight, 2)
            history, starts = frame.iloc[:270], frame.index[270:294].to_pydatetime()
            before = MLPDemandPredictor.load(directory / "bundle").predict(history, starts)
            predictor = MLPDemandPredictor.load(directory / "bundle")
            predictor.save(root / "copy")
            after = MLPDemandPredictor.load(root / "copy").predict(history, starts)
            np.testing.assert_allclose(before.demand_kwh, after.demand_kwh, atol=1e-6)
            with self.assertRaises(ValueError):
                run_weighted_comparison(data, config, root / "run", root / "models", reference_run, reference_models)
