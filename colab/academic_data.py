"""Public UCI acquisition and reduced hourly sample; shared with the notebook.

No local project imports. Calendar selection is fixed before fitting or scoring.
Only the prefix ending June 2007 is parsed, entirely inside the MVP's train.
"""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import hashlib

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ARCHIVE_SHA256 = "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"
SAMPLE_START = "2007-01-01 00:00:00"
SAMPLE_END = "2007-06-30 23:00:00"
ORIGINAL_TRAIN_END = "2009-09-20 12:00:00"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download_archive(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "uci_household.zip"
    if not path.exists():
        request = Request(UCI_URL, headers={"User-Agent": "BEKANTOR-Academic/1.0"})
        with urlopen(request, timeout=120) as response:
            payload = response.read(40 * 1024**2 + 1)
        if len(payload) > 40 * 1024**2:
            raise ValueError("Unexpected UCI download size")
        if digest(payload) != ARCHIVE_SHA256:
            raise ValueError("UCI archive changed; review provenance before continuing")
        path.write_bytes(payload)
    if digest(path.read_bytes()) != ARCHIVE_SHA256:
        raise ValueError("Cached UCI checksum differs")
    return path


def sample_from_archive(path):
    """Equivalent GAP-only extraction of the core's causal hourly aggregation."""
    stop = pd.Timestamp(SAMPLE_END) + pd.Timedelta(minutes=59)
    pieces = []
    with ZipFile(path) as archive, archive.open("household_power_consumption.txt") as stream:
        chunks = pd.read_csv(stream, sep=";", usecols=["Date", "Time", "Global_active_power"],
                             dtype=str, na_values=["?", "", "NA"], chunksize=50_000)
        for raw in chunks:
            stamps = pd.to_datetime(raw.Date + " " + raw.Time, format="%d/%m/%Y %H:%M:%S")
            assert stamps.max() < pd.Timestamp(ORIGINAL_TRAIN_END)
            power = pd.to_numeric(raw.Global_active_power, errors="raise").to_numpy(float)
            pieces.append(pd.Series(power[stamps <= stop], index=stamps[stamps <= stop]))
            if stamps.max() >= stop:
                break
    minute_power = pd.concat(pieces)
    assert minute_power.index.is_monotonic_increasing and not minute_power.index.has_duplicates
    assert not np.isinf(minute_power).any() and not (minute_power < 0).any()
    grid = pd.date_range(minute_power.index.min(), stop, freq="min", name="timestamp")
    observed = minute_power.reindex(grid)
    cleaned = observed.ffill(limit=5)  # Only past values; no interpolation from the future.
    result = pd.DataFrame({
        "load_kwh": cleaned.resample("h").sum(min_count=60) / 60,
        "is_observed_complete": observed.resample("h").count() == 60,
        "observed_minutes": observed.resample("h").count(),
        "imputed_minutes": (observed.isna() & cleaned.notna()).resample("h").sum(),
    }).loc[SAMPLE_START:SAMPLE_END]
    assert result.index.max() <= pd.Timestamp(ORIGINAL_TRAIN_END)
    assert len(result) == 181 * 24
    return result


def canonical_csv(frame):
    return frame.to_csv(index_label="timestamp", float_format="%.9f", lineterminator="\n").encode("utf-8")
