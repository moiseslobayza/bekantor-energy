"""Tiny CPU integration fixtures contain only train and validation, never test."""
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


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow is absent")
class CalendarTensorFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = configure_tensorflow(2026, 2)

    def test_fixed_hidden_layers_and_backward_compatible_initialization(self):
        config = ExperimentConfig(architectures=(Architecture("small", (64, 32)),))
        original = build_model(config, config.architectures[0])
        weights = original.get_weights()
        explicit = build_model(config, config.architectures[0], input_size=168)
        for a, b in zip(weights, explicit.get_weights()):
            np.testing.assert_array_equal(a, b)
        calendar = build_model(config, config.architectures[0], input_size=312)
        self.assertEqual([layer.units for layer in calendar.layers], [64, 32, 24])
        self.assertEqual(original.count_params(), 13688)
        self.assertEqual(calendar.count_params(), 22904)
        self.assertEqual(calendar.loss, "mse")
        self.assertIsInstance(calendar.optimizer, self.tf.keras.optimizers.Adam)
        self.assertFalse(self.tf.config.get_visible_devices("GPU"))

    def test_six_paired_fits_validation_only_and_predictor_roundtrip(self):
        import bekantor_energy.forecasting.calendar_experiment as experiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            n = 360
            frame = pd.DataFrame({"load_kwh": 1 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 24),
                                  "is_observed_complete": True},
                                 index=pd.date_range("2007-01-01", periods=n, freq="h", name="timestamp"))
            manifest = {"partitions": {}}
            for name, part in {"train": frame.iloc[:270], "validation": frame.iloc[270:]}.items():
                part.to_csv(data / f"{name}.csv")
                manifest["partitions"][name] = {"sha256": sha256(data / f"{name}.csv"), "hours": len(part),
                    "start": part.index[0].isoformat(), "end": part.index[-1].isoformat()}
            write_json(data / "manifest.json", manifest)
            training = ExperimentConfig(epochs=1, patience=1, architectures=(Architecture("small", (64, 32)),))
            config = CalendarComparisonConfig(training=training)
            with patch.object(experiment, "_checked_partition", wraps=experiment._checked_partition) as reader:
                run_calendar_comparison(data, config, root / "run", root / "models")
            self.assertEqual([call.args[1] for call in reader.call_args_list], ["train", "validation"])
            fits = pd.read_csv(root / "run" / "fits.csv")
            self.assertEqual(len(fits), 6)
            metrics = pd.read_csv(root / "run" / "metrics_by_seed.csv")
            self.assertEqual(metrics[metrics.scope == "global"].n_pairs.nunique(), 1)
            for variant in ("reference", "calendar"):
                bundle = root / "models" / "seed_2026" / variant / "bundle"
                predictor = MLPDemandPredictor.load(bundle)
                before = predictor.predict(frame.iloc[:270], frame.index[270:294].to_pydatetime())
                predictor.save(root / f"copy_{variant}")
                after = MLPDemandPredictor.load(root / f"copy_{variant}").predict(
                    frame.iloc[:270], frame.index[270:294].to_pydatetime())
                np.testing.assert_allclose(before.demand_kwh, after.demand_kwh, atol=1e-6)
                if variant == "reference":
                    metadata = json.loads((bundle / "predictor.json").read_text())
                    del metadata["feature_mode"]
                    write_json(bundle / "predictor.json", metadata)
                    self.assertEqual(MLPDemandPredictor.load(bundle).feature_mode, "consumption_only")
            with self.assertRaises(ValueError):
                run_calendar_comparison(data, config, root / "run", root / "models")
