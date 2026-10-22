"""One reusable processing/EDA workflow for the CLI and future academic export."""

import importlib.metadata
from pathlib import Path
import platform

from . import __version__
from .data.acquisition import sha256, write_json
from .data.splits import chronological_split
from .data.uci import hourly_consumption, load_uci
from .eda import run_eda


def process_uci(input_path: Path, output: Path, reports: Path, *,
                max_ffill_minutes: int = 5, train_fraction: float = 0.7,
                validation_fraction: float = 0.15, source_kind: str = "uci_measured",
                eda_scope: str = "train") -> dict:
    if source_kind not in ("uci_measured", "synthetic_fixture"):
        raise ValueError("Declare uci_measured or synthetic_fixture provenance")
    if eda_scope not in ("train", "all"):
        raise ValueError("EDA scope must be train or all")
    input_path, output, reports = Path(input_path), Path(output), Path(reports)
    hourly, quality = hourly_consumption(load_uci(input_path), max_ffill_minutes=max_ffill_minutes)
    splits = chronological_split(hourly, train_fraction, validation_fraction)
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / "hourly.csv", index_label="timestamp")
    partitions = {}
    for name, frame in splits.items():
        target = output / f"{name}.csv"
        frame.to_csv(target, index_label="timestamp")
        partitions[name] = {"hours": len(frame), "start": frame.index.min().isoformat(),
                            "end": frame.index.max().isoformat(), "sha256": sha256(target),
                            "fully_observed_target_hours": int(frame.is_observed_complete.sum())}
    write_json(reports / "quality.json", quality)
    run_eda(splits["train"] if eda_scope == "train" else hourly, reports / "eda",
            scope=eda_scope, source_kind=source_kind)
    manifest = {"package_version": __version__, "input": str(input_path.resolve()),
                "input_sha256": sha256(input_path), "source_kind": source_kind,
                "python": platform.python_version(),
                "dependencies": {p: importlib.metadata.version(p) for p in ("pandas", "numpy", "matplotlib")},
                "parameters": {"max_ffill_minutes": max_ffill_minutes, "train_fraction": train_fraction,
                               "validation_fraction": validation_fraction, "eda_scope": eda_scope},
                "partitions": partitions, "quality": quality,
                "hourly_sha256": sha256(output / "hourly.csv"),
                "leakage_policy": "causal fill only; no scaling; split before windowing; train-only EDA by default"}
    write_json(output / "manifest.json", manifest)
    return manifest
