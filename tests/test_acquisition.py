from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from zipfile import ZipFile

from bekantor_energy.data.acquisition import UCI_MEMBER, acquire_uci, download, write_json
from bekantor_energy.data.pvgis import PVGISQuery, acquire_pvgis, parse_pvgis

FIXTURES = Path(__file__).parent / "fixtures"


class DownloadTests(unittest.TestCase):
    def test_download_size_limit_removes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "file"
            with patch("bekantor_energy.data.acquisition.urlopen", return_value=BytesIO(b"too big")):
                with self.assertRaises(ValueError):
                    download("https://example.test", target, max_bytes=2)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_name("file.part").exists())

    def test_retries_overload_then_success(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "file"
            error = HTTPError("https://example.test", 529, "busy", {}, None)
            with patch("bekantor_energy.data.acquisition.urlopen", side_effect=[error, BytesIO(b"ok")]) as http:
                with patch("bekantor_energy.data.acquisition.time.sleep"):
                    download("https://example.test", target, max_bytes=20)
            self.assertEqual(http.call_count, 2)
            self.assertEqual(target.read_bytes(), b"ok")

    def test_client_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            error = HTTPError("https://example.test", 400, "bad request", {}, None)
            with patch("bekantor_energy.data.acquisition.urlopen", side_effect=error) as http:
                with self.assertRaises(HTTPError):
                    download("https://example.test", Path(directory) / "file", max_bytes=20)
            self.assertEqual(http.call_count, 1)

    def test_uci_extract_receipt_and_cache_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fake_download(url, destination, **kwargs):
                with ZipFile(destination, "w") as archive:
                    archive.writestr(UCI_MEMBER, b"fixture")
                    archive.writestr("../should-not-extract", b"bad")
                return destination
            with patch("bekantor_energy.data.acquisition.download", side_effect=fake_download) as fetch:
                target = acquire_uci(root)
                self.assertEqual(target.read_bytes(), b"fixture")
                self.assertTrue(target.with_suffix(".metadata.json").exists())
                self.assertEqual(acquire_uci(root), target)
                self.assertEqual(fetch.call_count, 1)
                target.write_text("tampered", encoding="utf-8")
                with self.assertRaises(ValueError):
                    acquire_uci(root)


class PVGISTests(unittest.TestCase):
    def setUp(self):
        self.query = PVGISQuery(start="2023-01-01T00:00:00", end="2023-01-01T03:00:00")
        self.payload = json.loads((FIXTURES / "pvgis_v6_synthetic.json").read_text(encoding="utf-8"))

    def test_watts_and_hourly_energy_estimate(self):
        frame = parse_pvgis(self.payload, self.query)
        self.assertEqual(frame.iloc[1].pv_kw, 1)
        self.assertEqual(frame.iloc[2].pv_kwh_estimate, 2)
        self.assertEqual(str(frame.index.tz), "UTC")
        self.assertEqual(frame.iloc[0].source_kind, "pvgis_modeled_generation")

    def test_current_api_parameters_are_explicit(self):
        self.assertIn("/api/v6/power/broadband?", self.query.url)
        self.assertEqual(self.query.params["timezone"], "UTC")
        self.assertEqual(self.query.params["peak-power"], 3)
        self.assertEqual(self.query.params["surface_orientation"], 0)
        self.assertNotIn("periods", self.query.params)

    def test_unknown_or_misaligned_responses_fail(self):
        cases = [dict(self.payload, power=[1]), dict(self.payload, power=[0, -1, 2, 0]),
                 dict(self.payload, power=[0, float("nan"), 2, 0]),
                 dict(self.payload, timestamps=list(reversed(self.payload["timestamps"]))),
                 dict(self.payload, peak_power_unit="W"), {"outputs": {"hourly": []}}]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                parse_pvgis(payload, self.query)

    def test_bad_query_rejected(self):
        for kwargs in ({"latitude": -91}, {"peak_kwp": 0}, {"system_efficiency": 1.1},
                       {"end": "2007-01-01"}, {"end": "2026-01-01"},
                       {"start": "2023-01-01T00:30:00"}, {"irradiance_source": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                replace(self.query, **kwargs)

    def test_pvgis_query_aware_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "solar.json"
            with patch("bekantor_energy.data.pvgis.download") as fetch:
                fetch.side_effect = lambda url, path, **kwargs: write_json(path, self.payload)
                acquire_pvgis(self.query, target)
                acquire_pvgis(self.query, target)
                self.assertEqual(fetch.call_count, 1)
                with self.assertRaises(ValueError):
                    acquire_pvgis(replace(self.query, latitude=0), target)
