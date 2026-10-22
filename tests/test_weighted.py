from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.forecasting.calendar_experiment import aggregate_metrics
from bekantor_energy.forecasting.weighted_experiment import WeightedComparisonConfig, weighted_validation_rows
from bekantor_energy.forecasting.windows import WindowSet


class WeightedMetricTests(unittest.TestCase):
    def test_fixed_weight_protocol_and_low_consumption_definition(self):
        config = WeightedComparisonConfig.load(Path(__file__).parents[1] / "configs/mlp.weighted-mse.json")
        self.assertEqual((config.normal_weight, config.peak_weight), (1, 2))
        for values in ({"peak_weight": 3}, {"normal_weight": 2}, {"peak_quantile": .95},
                       {"low_quantile": .1}, {"weight_reduction": "divide_by_sum_weights"},
                       {"global_error_tolerance_pct": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                replace(config, **values)

    def test_low_consumption_boundary_and_overprediction_metrics(self):
        actual = np.array([[.25, .5, 3., 4.]])
        predicted = np.array([[.5, .5, 2., 2.]])
        windows = WindowSet(np.empty((1, 168)), actual, pd.DatetimeIndex(["2008-01-01"]), {})
        rows = pd.DataFrame(weighted_validation_rows(windows, predicted, {"q25": .25, "q90": 3., "q95": 3.5}, 1, "mse"))
        low = rows[rows.scope == "low_consumption"].set_index("metric")
        self.assertEqual(low.loc["mae_kwh", "value"], .25)
        self.assertEqual(low.loc["bias_kwh", "value"], .25)
        self.assertEqual(low.loc["overestimate_pct", "value"], 100)
        self.assertEqual(low.n_pairs.max(), 1)
        high = rows[(rows.scope == "high_consumption") & (rows.group == "q90")]
        self.assertEqual(high.n_pairs.max(), 1)  # Strict >, not >=.

    def test_custom_variants_are_paired_and_use_sample_standard_deviation(self):
        rows = [{"seed": seed, "variant": variant, "scope": "global", "group": "all", "metric": "rmse_kwh",
                 "value": seed + delta, "n_pairs": 24}
                for seed in (1, 2, 3) for variant, delta in (("mse", 0), ("weighted_mse", -.2))]
        _, aggregate, paired = aggregate_metrics(rows, ("mse", "weighted_mse"))
        self.assertEqual(aggregate.loc[aggregate.variant == "mse", "std"].iloc[0], 1)
        np.testing.assert_allclose(paired.delta_weighted_mse_minus_mse, -.2)
