"""Small synthetic integration tests; never read the real UCI test partition."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from bekantor_energy.data.acquisition import sha256, write_json
from bekantor_energy.forecasting.config import Architecture, ExperimentConfig
from bekantor_energy.forecasting.experiment import evaluate_test_once, train_experiment
from bekantor_energy.forecasting.mlp import MLPDemandPredictor, build_model, configure_tensorflow


@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow dependency not installed")
class TensorFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tf = configure_tensorflow(2026, 2)

    def test_seed_initialization_and_nonnegative_cpu_output(self):
        config = ExperimentConfig(architectures=(Architecture("tiny", (8,)),))
        first = build_model(config, config.architectures[0])
        weights = [w.copy() for w in first.get_weights()]
        second = build_model(config, config.architectures[0])
        for a, b in zip(weights, second.get_weights()):
            np.testing.assert_array_equal(a, b)
        values = second(np.zeros((2, 168), dtype=np.float32)).numpy()
        self.assertEqual(values.shape, (2, 24))
        self.assertTrue((values >= 0).all())
        self.assertFalse(self.tf.config.get_visible_devices("GPU"))

    def test_synthetic_full_training_roundtrip_and_single_test_guard(self):
        import bekantor_energy.forecasting.experiment as experiment
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            n = 1000
            frame = pd.DataFrame({"load_kwh": 1 + 0.5 * np.sin(np.arange(n) * 2 * np.pi / 24),
                                  "is_observed_complete": True},
                                 index=pd.date_range("2007-01-01", periods=n, freq="h", name="timestamp"))
            pieces = {"train": frame.iloc[:500], "validation": frame.iloc[500:750], "test": frame.iloc[750:]}
            manifest = {"source_kind": "synthetic_fixture", "partitions": {}}
            for name, part in pieces.items():
                part.to_csv(data / f"{name}.csv")
                manifest["partitions"][name] = {"sha256": sha256(data / f"{name}.csv"), "hours": len(part),
                                               "start": part.index[0].isoformat(), "end": part.index[-1].isoformat()}
            write_json(data / "manifest.json", manifest)
            run, models = root / "run", root / "models"
            config = ExperimentConfig(epochs=2, patience=1, batch_size=64,
                                       architectures=(Architecture("tiny", (8,)),))
            with patch.object(experiment, "_checked_partition", wraps=experiment._checked_partition) as reader:
                selection = train_experiment(data, config, run, models)
            self.assertEqual([call.args[1] for call in reader.call_args_list], ["train", "validation"])
            self.assertFalse((run / "test_metrics.json").exists())
            self.assertEqual(selection["selected_architecture"], "tiny")
            repeated = train_experiment(data, config, root / "repeat", root / "repeat_models")
            for split in ("train", "validation"):
                self.assertAlmostEqual(selection["architectures"]["tiny"][split]["rmse_kwh"],
                                       repeated["architectures"]["tiny"][split]["rmse_kwh"], places=6)
            predictor = MLPDemandPredictor.load(models / "selected")
            history = frame.iloc[:500]
            starts = frame.index[500:524].to_pydatetime()
            before = predictor.predict(history, starts)
            self.assertEqual(len(before.demand_kwh), 24)
            self.assertEqual(before.issued_at, starts[0])
            predictor.save(root / "copy")
            after = MLPDemandPredictor.load(root / "copy").predict(history, starts)
            np.testing.assert_allclose(before.demand_kwh, after.demand_kwh, atol=1e-6)
            with self.assertRaises(ValueError):
                predictor.predict(history, frame.index[501:525].to_pydatetime())
            with patch.object(experiment, "_checked_partition", wraps=experiment._checked_partition) as reader:
                results = evaluate_test_once(data, run)
            self.assertEqual([call.args[1] for call in reader.call_args_list].count("test"), 1)
            self.assertEqual(set(results["metrics"]), {"selected_mlp", "seasonal_24h", "seasonal_168h"})
            self.assertEqual(json.loads((run / "test_evaluation.json").read_text())["status"], "completed")
            marker_before = (run / "test_evaluation.json").read_bytes()
            with patch.object(experiment, "_checked_partition") as reader:
                with patch.object(experiment, "predict_windows") as inference:
                    experiment.replot_experiment(run)
                reader.assert_not_called()
                inference.assert_not_called()
            self.assertEqual((run / "test_evaluation.json").read_bytes(), marker_before)
            with patch.object(experiment, "_checked_partition") as reader:
                with self.assertRaises(FileExistsError):
                    evaluate_test_once(data, run)
                reader.assert_not_called()
            # Frozen config cannot be changed to silently alter the test protocol.
            modified = config.to_dict()
            modified["stride"] = 24
            write_json(run / "config.json", modified)
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                evaluate_test_once(data, run)
