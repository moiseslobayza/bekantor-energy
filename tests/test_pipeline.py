import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from bekantor_energy.cli import main
from bekantor_energy.pipeline import process_uci


class PipelineTests(unittest.TestCase):
    def test_end_to_end_outputs_provenance_and_repeatability(self):
        fixture = Path(__file__).parent / "fixtures/uci_synthetic.txt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = process_uci(fixture, root / "processed", root / "reports", source_kind="synthetic_fixture")
            second = process_uci(fixture, root / "processed", root / "reports", source_kind="synthetic_fixture")
            self.assertEqual(first, second)
            self.assertEqual([p["hours"] for p in first["partitions"].values()], [8, 1, 3])
            summary = json.loads((root / "reports/eda/summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scope"], "train")
            self.assertEqual(summary["hours"], 8)
            self.assertEqual(summary["source_kind"], "synthetic_fixture")
            self.assertEqual(first["quality"]["missing_timestamp_minutes"], 1)
            self.assertGreater(first["quality"]["incomplete_hours"], 0)
            for name in ("hourly_consumption", "distribution", "hour_of_day", "coverage"):
                self.assertEqual((root / f"reports/eda/{name}.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            frame = pd.read_csv(root / "processed/hourly.csv")
            self.assertEqual(len(frame), 12)
            self.assertGreater(frame.load_kwh.isna().sum(), 0)

    def test_cli_rejects_invalid_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"charge_efficiency": 2}', encoding="utf-8")
            with self.assertRaises(SystemExit) as failure:
                main(["check-config", "--input", str(path)])
            self.assertEqual(failure.exception.code, 2)
