"""Bounded HTTP downloads with retries, atomic writes and provenance receipts."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
UCI_MEMBER = "household_power_consumption.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def download(url: str, destination: Path, *, max_bytes: int,
             timeout: float = 45, attempts: int = 3) -> Path:
    """The caller handles caching. A failed request leaves no partial final file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        for attempt in range(attempts):
            try:
                request = Request(url, headers={"User-Agent": "BEKANTOR-Energy/0.1 (research)"})
                with urlopen(request, timeout=timeout) as response, partial.open("wb") as output:
                    size = 0
                    while block := response.read(1024 * 1024):
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError("Download exceeds configured size limit")
                        output.write(block)
                partial.replace(destination)
                return destination
            except HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504, 529) or attempt == attempts - 1:
                    raise
            except (URLError, TimeoutError):
                if attempt == attempts - 1:
                    raise
            time.sleep(2 ** attempt)
    finally:
        partial.unlink(missing_ok=True)
    raise RuntimeError("No download attempt completed")


def provenance(path: Path, url: str, source_kind: str, **details: object) -> dict:
    return {"url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": sha256(path), "source_kind": source_kind, **details}


def acquire_uci(directory: Path, *, force: bool = False) -> Path:
    directory = Path(directory)
    target = directory / UCI_MEMBER
    receipt = target.with_suffix(".metadata.json")
    if target.exists() and not force:
        if not receipt.exists() or json.loads(receipt.read_text(encoding="utf-8"))["sha256"] != sha256(target):
            raise ValueError("Cached UCI file lacks valid provenance; use another directory or --force")
        return target
    archive = directory / "uci_household.zip"
    download(UCI_URL, archive, max_bytes=40 * 1024**2)
    partial = target.with_name(target.name + ".part")
    try:
        with ZipFile(archive) as zipped:
            info = zipped.getinfo(UCI_MEMBER)
            if info.file_size > 200 * 1024**2:
                raise ValueError("Unexpected UCI uncompressed size")
            # Extract exactly one known member; never trust ZIP paths.
            with zipped.open(info) as source, partial.open("wb") as output:
                shutil.copyfileobj(source, output)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    write_json(receipt, provenance(target, UCI_URL, "measured_household_consumption",
                                  archive_sha256=sha256(archive),
                                  license="CC BY 4.0", doi="10.24432/C58K54",
                                  timezone="unspecified source wall-clock; kept naive"))
    return target
