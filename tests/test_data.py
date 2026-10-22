from io import StringIO
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from bekantor_energy.data.splits import chronological_split
from bekantor_energy.data.uci import NUMERIC_COLUMNS, hourly_consumption, load_uci


def minute_frame(periods=180):
    frame = pd.DataFrame(1.0, index=pd.date_range("2007-01-01", periods=periods, freq="min"),
                         columns=NUMERIC_COLUMNS)
    frame["Global_active_power"] = 2.0
    frame.index.name = "timestamp"
    return frame


HEADER = "Date;Time;" + ";".join(NUMERIC_COLUMNS) + "\n"
ROW = "01/01/2007;00:00:00;2;0.1;230;9;1;2;3\n"


class LoadingTests(unittest.TestCase):
    def test_fixture_is_synthetic_and_parseable(self):
        path = Path(__file__).parent / "fixtures/uci_synthetic.txt"
        frame = load_uci(path)
        self.assertEqual(len(frame), 719)
        self.assertTrue(frame.index.is_monotonic_increasing)

    def test_missing_markers(self):
        frame = load_uci(StringIO(HEADER + ROW.replace(";2;0.1", ";?;0.1")))
        self.assertTrue(np.isnan(frame.iloc[0].Global_active_power))
        frame = load_uci(StringIO(HEADER + ROW.replace(";2;0.1", ";;0.1")))
        self.assertTrue(np.isnan(frame.iloc[0].Global_active_power))

    def test_invalid_schema_and_values(self):
        cases = ["Date;Time\n01/01/2007;00:00:00\n", HEADER,
                 HEADER + ROW + ROW,
                 HEADER + ROW.replace("01/01/2007", "31/02/2007"),
                 HEADER + ROW.replace("00:00:00", "00:00:30"),
                 HEADER + ROW.replace(";2;0.1", ";bad;0.1"),
                 HEADER + ROW.replace(";2;0.1", ";-2;0.1"),
                 HEADER + ROW.replace(";2;0.1", ";inf;0.1")]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_uci(StringIO(value))


class HourlyTests(unittest.TestCase):
    def test_kw_and_wh_are_integrated_differently(self):
        hourly, quality = hourly_consumption(minute_frame())
        self.assertEqual(hourly.iloc[0].load_kwh, 2.0)
        self.assertEqual(hourly.iloc[0].load_kw, 2.0)
        self.assertEqual(hourly.iloc[0].sub_metering_1_kwh, 0.06)
        self.assertEqual(quality["complete_hours"], 3)

    def test_partial_hour_is_not_extrapolated(self):
        hourly, _ = hourly_consumption(minute_frame(90))
        self.assertTrue(np.isnan(hourly.iloc[1].load_kwh))
        self.assertEqual(hourly.iloc[1].observed_coverage, 0.5)

    def test_missing_timestamp_is_inserted_and_reported(self):
        frame = minute_frame().drop(pd.Timestamp("2007-01-01 00:20"))
        hourly, quality = hourly_consumption(frame)
        self.assertEqual(quality["missing_timestamp_minutes"], 1)
        self.assertEqual(hourly.iloc[0].observed_minutes, 59)
        self.assertEqual(hourly.iloc[0].imputed_minutes, 1)
        self.assertEqual(hourly.iloc[0].load_kwh, 2)

    def test_leading_missing_is_never_backward_filled(self):
        frame = minute_frame()
        frame.iloc[0] = np.nan
        hourly, _ = hourly_consumption(frame)
        self.assertTrue(np.isnan(hourly.iloc[0].load_kwh))
        self.assertEqual(hourly.iloc[0].imputed_minutes, 0)

    def test_long_gap_stays_missing_after_limit(self):
        frame = minute_frame()
        frame.iloc[60:70] = np.nan
        hourly, quality = hourly_consumption(frame, max_ffill_minutes=5)
        self.assertEqual(hourly.iloc[1].imputed_minutes, 5)
        self.assertTrue(np.isnan(hourly.iloc[1].load_kwh))
        self.assertEqual(quality["missing_after"]["Global_active_power"], 5)

    def test_no_fill_option(self):
        frame = minute_frame()
        frame.iloc[1] = np.nan
        hourly, _ = hourly_consumption(frame, max_ffill_minutes=0)
        self.assertTrue(np.isnan(hourly.iloc[0].load_kwh))

    def test_future_values_do_not_change_past(self):
        frame = minute_frame(600)
        frame.iloc[50:52] = np.nan
        before, _ = hourly_consumption(frame)
        changed = frame.copy()
        changed.iloc[180:] *= 1000
        after, _ = hourly_consumption(changed)
        pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])

    def test_empty_and_bad_options_fail(self):
        for frame, limit in ((minute_frame(0), 5), (minute_frame(), -1), (minute_frame(), 1.5)):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                hourly_consumption(frame, max_ffill_minutes=limit)


class SplitTests(unittest.TestCase):
    def test_disjoint_ordered_and_exhaustive(self):
        frame = minute_frame(100)
        splits = chronological_split(frame)
        self.assertEqual([len(v) for v in splits.values()], [70, 15, 15])
        self.assertLess(splits["train"].index.max(), splits["validation"].index.min())
        self.assertLess(splits["validation"].index.max(), splits["test"].index.min())
        pd.testing.assert_frame_equal(pd.concat(splits.values()), frame)

    def test_invalid_splits_fail(self):
        for frame, train, validation in ((minute_frame(3), 0.7, 0.15),
                (minute_frame(), 0.8, 0.2), (minute_frame(), float("nan"), 0.1),
                (minute_frame().iloc[::-1], 0.7, 0.15),
                (pd.concat([minute_frame(), minute_frame()]), 0.7, 0.15)):
            with self.subTest(train=train), self.assertRaises(ValueError):
                chronological_split(frame, train, validation)
