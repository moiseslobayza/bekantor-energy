"""Small checks of the generated notebook; no TensorFlow fitting or MVP data."""
import ast
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import heapq
from itertools import count
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
from types import ModuleType, SimpleNamespace
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = json.loads((ROOT / "colab/BEKANTOR_Energy_Trabajo_Final.ipynb").read_text(encoding="utf-8"))


def extracted_namespace():
    module = ModuleType("academic_generated")
    sys.modules[module.__name__] = module
    module.__dict__.update(np=np, pd=pd, dataclass=dataclass, SimpleNamespace=SimpleNamespace,
                           math=math, heapq=heapq, count=count, perf_counter=perf_counter)
    for cell in NOTEBOOK["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code": continue
        if source.startswith(("def validate_hourly", "HomeConfig =", "def finite_nonnegative",
                              "class ScalarArithmetic", "class ProblemaBateria")):
            exec(source, module.__dict__)
        elif source.startswith(("def seasonal_persistence", "def home_config")):
            nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name != "replay"]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), "generated-functions", "exec"), module.__dict__)
    return SimpleNamespace(**module.__dict__)


class AcademicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.api = extracted_namespace()

    def test_notebook_is_valid_json_with_portable_code(self):
        self.assertEqual(NOTEBOOK["nbformat"], 4)
        self.assertFalse(NOTEBOOK["metadata"]["bekantor"]["historical_test_used"])
        for cell in NOTEBOOK["cells"]:
            if cell["cell_type"] != "code": continue
            source = "".join(cell["source"])
            ast.parse(source)
            self.assertNotIn("import bekantor_energy", source)
            self.assertIsNone(re.search(r"(?:^|[\s\"'])[A-Za-z]:[\\/]", source))
            self.assertNotIn("load_model(", source)

    def test_assets_are_small_and_provenance_matches(self):
        folder = ROOT / "colab/assets"
        meta = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["uci"]["license"], "CC BY 4.0")
        self.assertTrue(meta["academic_partitions"]["all_within_original_train"])
        for name, checksum in meta["assets"].items():
            path = folder / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
            self.assertLess(path.stat().st_size, 250_000)
        frame = pd.read_csv(folder / "consumption_hourly.csv", index_col="timestamp", parse_dates=True)
        self.assertEqual(len(frame), 4344)
        self.assertLess(frame.index.max(), pd.Timestamp(meta["uci"]["original_train_end"]))
        self.assertTrue(frame.loc[frame.is_observed_complete].load_kwh.notna().all())
        solar = pd.read_csv(folder / "solar_cordoba_48h.csv")
        stamps = pd.to_datetime(solar.source_utc)
        self.assertTrue((stamps.diff().dropna() == pd.Timedelta(hours=1)).all())
        self.assertEqual(len(solar), 48)

    def test_extracted_source_files_match_frozen_core(self):
        sources = NOTEBOOK["metadata"]["bekantor"]["source_sha256"]
        self.assertEqual(len(sources), 7)
        for name, checksum in sources.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), checksum)

    def test_missing_inputs_and_unknown_targets_are_rejected(self):
        frame = pd.DataFrame({"load_kwh": 1., "is_observed_complete": True},
                             index=pd.date_range("2007-01-01", periods=100, freq="h"))
        frame.loc[frame.index[30], "load_kwh"] = np.nan
        frame.loc[frame.index[[30, 50]], "is_observed_complete"] = False
        windows = self.api.make_windows(frame, lookback=8, horizon=4)
        self.assertGreater(windows.report["unknown_inputs"], 0)
        self.assertGreater(windows.report["unknown_targets"], 0)
        for origin in windows.origins:
            labels = frame.loc[origin:origin + pd.Timedelta(hours=3)]
            self.assertTrue(labels.is_observed_complete.all())
        changed = frame.copy()
        changed.loc[changed.index >= frame.index[60], "load_kwh"] = 999.
        later = self.api.make_windows(changed, lookback=8, horizon=4)
        i = windows.origins.get_loc(frame.index[60])
        j = later.origins.get_loc(frame.index[60])
        np.testing.assert_array_equal(windows.x[i], later.x[j])

    def test_chronological_history_and_training_only_normalization(self):
        frame = pd.DataFrame({"load_kwh": np.arange(100, dtype=float)+1, "is_observed_complete": True},
                             index=pd.date_range("2007-01-01", periods=100, freq="h"))
        train, val = frame.iloc[:50], frame.iloc[50:]
        windows = self.api.make_windows(val, history=train, lookback=8, horizon=4)
        self.assertEqual(windows.origins.min(), val.index.min())
        self.assertLessEqual(windows.origins.max()+pd.Timedelta(hours=3), val.index.max())
        normalizer = self.api.DemandNormalizer.fit(train)
        self.assertEqual(normalizer.fitted_hours, 50)
        self.assertEqual(normalizer.mean_kwh, 25.5)
        np.testing.assert_allclose(normalizer.inverse_targets(normalizer.targets(np.array([2., 3.]))), [2., 3.], atol=1e-6)
        with self.assertRaises(ValueError): self.api.make_windows(val, history=frame, lookback=8, horizon=4)

    def test_physical_extremes_and_energy_conservation(self):
        a = self.api
        home = a.home_config([.15]*24)
        for energy, action, demand, sun in ((1., a.EnergyAction(discharge_kw=99), 2., 0.),
                (5., a.EnergyAction(charge_kw=99), .2, 4.), (2.5, a.EnergyAction(charge_kw=1), 1., 2.)):
            result = a.PhysicalSimulator().step(a.EnergyState(0, energy), action, demand, sun, home, repair=True)
            self.assertGreaterEqual(result.state.stored_kwh, 1.)
            self.assertLessEqual(result.state.stored_kwh, 5.)
            self.assertEqual(result.constraint_violations, 0)
            self.assertAlmostEqual(result.storage_residual_kwh, 0.)
            self.assertAlmostEqual(result.balance_residual_kwh, 0.)

    def test_astar_matches_ucs_and_independent_enumeration(self):
        rng = np.random.default_rng(2026)
        for _ in range(8):
            demand, solar = rng.uniform(0, 2, 4), rng.uniform(0, 1, 4)
            args = (demand, solar, [.08, .38, .15, .08], .5)
            astar, problem = self.api.plan(*args, capacity=1.)
            ucs, _ = self.api.plan(*args, capacity=1., heuristic="zero")
            @lru_cache(None)
            def optimum(state):
                if problem.is_goal(state): return 0.
                return min(edge + optimum(nxt) for _, nxt, edge in problem.successors(state))
            self.assertAlmostEqual(astar.value, ucs.value)
            self.assertAlmostEqual(astar.value, optimum(problem.initial))

    def test_forecasts_influence_first_action(self):
        low, _ = self.api.plan([0., 0.], [0., 0.], [.08, .38], .5, capacity=1.)
        high, _ = self.api.plan([0., 1.], [0., 0.], [.08, .38], .5, capacity=1.)
        self.assertNotEqual(low.actions[0], high.actions[0])

    def test_physical_metric_formula_and_seasonal_past(self):
        values = self.api.physical_metrics(np.array([0., 2.]), np.array([1., 4.]))
        self.assertEqual(values["MAE kWh"], 1.5)
        self.assertAlmostEqual(values["RMSE kWh"], math.sqrt(2.5))
        past = np.arange(168, dtype=float)[None]
        np.testing.assert_array_equal(self.api.seasonal_persistence(past, lag=24), past[:, -24:])


if __name__ == "__main__": unittest.main()
