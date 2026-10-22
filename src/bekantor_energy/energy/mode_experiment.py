"""Four-policy continuous paired validation using frozen Phase 4.2 conditions."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd

from ..data.acquisition import sha256, write_json
from ..forecasting.windows import read_hourly
from .continuous import run_continuous_case, summarize_continuous
from .modes import ModePlanner, ObjectiveConfig
from .planner import PlannerConfig

MODE_STRATEGIES = ("reactive", "economic", "autonomy", "balanced")


@dataclass(frozen=True)
class ModeComparisonConfig:
    previous_run: str = "experiments/phase4_2/continuous-20261022"
    source_archive: str = "experiments/phase4_3/preparation-20261022/preservation-input.json"
    balanced_cost_weight: float = .5
    balanced_cost_scale_sim: float = 1.
    balanced_grid_scale_kwh: float = 1.
    workers: int = 2
    seed: int = 2026
    description: str = "Fixed objective comparison"

    def __post_init__(self):
        ObjectiveConfig("balanced", self.balanced_cost_weight, self.balanced_cost_scale_sim, self.balanced_grid_scale_kwh)
        if self.seed != 2026 or type(self.workers) is not int or not 1 <= self.workers <= 4:
            raise ValueError("Use frozen seed 2026 and one to four local workers")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def objectives(self):
        return {"economic": ObjectiveConfig("economic"), "autonomy": ObjectiveConfig("autonomy"),
                "balanced": ObjectiveConfig("balanced", self.balanced_cost_weight,
                                            self.balanced_cost_scale_sim, self.balanced_grid_scale_kwh)}


def mode_preservation_snapshot(config):
    """Preserve old results verbatim; authorized source evolution uses archives.

    Historical manifests are NEVER rewritten to pretend the active source hash
    is unchanged. Only the explicitly archived planner source can substitute for
    an old protected Python source; every data/model/result hash still matches.
    """
    manifest_path = Path(config.source_archive)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    previous = Path(config.previous_run)
    if previous.resolve() != Path(manifest["previous_run"]).resolve():
        raise ValueError("Source archive refers to another prior run")
    verification = previous / "verification.json"
    preserved = previous / "preservation.json"
    verified = json.loads(verification.read_text(encoding="utf-8"))
    old = json.loads(preserved.read_text(encoding="utf-8"))
    expected = {**verified["artifact_sha256"], **old["original_artifacts_before"]}
    expected.update({str(verification.resolve()): sha256(verification), str(preserved.resolve()): sha256(preserved)})
    if expected != manifest["protected_before"]:
        # Paths in prior maps can be relative; snapshot keys were resolved.
        resolved = {str(Path(p).resolve()): h for p, h in expected.items()}
        if resolved != manifest["protected_before"]:
            raise ValueError("Historical preservation scope/hash changed since source archiving")
        expected = resolved
    snapshot, changes = {}, []
    archives = manifest["source_archives"]
    allowed_runtime_paths = {str(Path(p).resolve()) for p in (
        "src/bekantor_energy/energy/planner.py", "src/bekantor_energy/energy/continuous.py",
        "src/bekantor_energy/cli.py", "README.md")}
    for original, info in archives.items():
        if original not in allowed_runtime_paths:
            raise ValueError("Unexpected authorized source archive")
        archived = Path(info["archive"])
        if sha256(archived) != info["sha256"]:
            raise ValueError("Historical source archive changed")
        snapshot[str(archived.resolve())] = info["sha256"]
        changes.append({"source": original, "archive": str(archived.resolve()),
                        "before_sha256": info["sha256"], "current_sha256": sha256(Path(original))})
    for value, checksum in expected.items():
        path = Path(value)
        if path.name.startswith("test"):
            raise ValueError("Held-out path in historical allowlist")
        if str(path.resolve()) in archives:
            info = archives[str(path.resolve())]
            if info["sha256"] != checksum:
                raise ValueError("Archived source does not match its historical checksum")
            snapshot[str(Path(info["archive"]).resolve())] = checksum
        else:
            if sha256(path) != checksum:
                raise ValueError(f"Prior model/data/result changed: {path}")
            snapshot[str(path.resolve())] = checksum
    snapshot[str(manifest_path.resolve())] = sha256(manifest_path)
    diagnostic = manifest_path.parent / "preliminary-source" / "numerical-diagnostic.json"
    if diagnostic.exists():
        snapshot[str(diagnostic.resolve())] = sha256(diagnostic)
        audit = json.loads(diagnostic.read_text(encoding="utf-8"))
        preliminary = Path(audit["preliminary_run"])
        for artifact in preliminary.rglob("*"):
            if artifact.is_file():
                snapshot[str(artifact.resolve())] = sha256(artifact)
    return snapshot, changes


def run_mode_case(case, output_dir):
    settings = PlannerConfig(**case["planner"])
    planners = {name: ModePlanner(settings, ObjectiveConfig(**values)) for name, values in case["objectives"].items()}
    return run_continuous_case(case, output_dir, planning_policies=planners)


def verify_economic_regression(root, prior, cases):
    rows = []
    columns = ["initial_kwh", "stored_kwh", "requested_charge_kw", "requested_discharge_kw", "cost",
               "grid_import_kwh", "charge_kwh", "discharge_kwh", "curtailed_kwh", "pv_charge_kwh"]
    for case_id in cases:
        before = pd.read_csv(prior / "cases" / case_id / "ledger.csv", parse_dates=["timestamp"])
        after = pd.read_csv(root / "cases" / case_id / "ledger.csv", parse_dates=["timestamp"])
        before = before[before.strategy == "mlp_mse"].reset_index(drop=True)
        after = after[after.strategy == "economic"].reset_index(drop=True)
        matched = len(after) == len(before) and after.timestamp.equals(before.timestamp)
        difference = float(np.abs(after[columns].to_numpy() - before[columns].to_numpy()).max()) if matched else None
        rows.append({"case_id": case_id, "hours": len(after), "matches": matched and difference < 1e-9,
                     "max_abs_difference": difference})
    write_json(root / "economic_regression.json", {"all_match": all(r["matches"] for r in rows), "cases": rows,
        "reference": str(prior), "scope": "all realized states/actions/physical flows/costs; excludes nondeterministic timings"})
    return all(r["matches"] for r in rows)


def run_mode_experiment(config, run_dir, *, data_dir=Path("data/processed/uci")):
    root, prior = Path(run_dir), Path(config.previous_run)
    if root.exists():
        raise ValueError("New result directory required")
    protected_roots = [prior, Path(data_dir), Path(config.source_archive).parent, Path("models"), Path("data")]
    if any(root.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(root.resolve()) for p in protected_roots):
        raise ValueError("Output must be disjoint from inputs, models and scenarios")
    before, source_changes = mode_preservation_snapshot(config)
    prior_design = json.loads((prior / "design.json").read_text(encoding="utf-8"))
    prior_repro = json.loads((prior / "reproducibility.json").read_text(encoding="utf-8"))
    validation_path = Path(data_dir) / "validation.csv"
    if sha256(validation_path) != prior_repro["data"]["validation"]["sha256"]:
        raise ValueError("Validation checksum differs from the frozen paired reference")
    validation = read_hourly(validation_path)
    bundle = Path(prior_design["model_bundle"])
    predictor = json.loads((bundle / "predictor.json").read_text(encoding="utf-8"))
    if predictor["config"]["seed"] != config.seed or predictor["feature_mode"] != "future_calendar":
        raise ValueError("Expected frozen MSE calendar seed-2026 model")
    with np.load(prior_design["frozen_predictions"], allow_pickle=False) as saved:
        origins = pd.DatetimeIndex(saved["origins"])
        predictions = saved["predicted_kwh"].copy()
    cases = pd.read_csv(prior / "cases.csv")
    objectives = {name: asdict(value) for name, value in config.objectives().items()}
    tasks = []
    for row in cases.itertuples():
        case = json.loads((prior / "cases" / row.case_id / "case.json").read_text(encoding="utf-8"))
        times = pd.date_range(case["start"], periods=case["hours"], freq="h")
        tail = pd.date_range(case["start"], periods=case["hours"] + 23, freq="h")
        indices = origins.get_indexer(times)
        if (indices < 0).any() or not validation.reindex(tail).is_observed_complete.fillna(False).all():
            raise ValueError("Incomplete validation-only continuous forecast block")
        np.testing.assert_allclose(validation.loc[times, "load_kwh"], case["actual_demand"], atol=1e-12, rtol=0)
        case.update({"predictions": predictions[indices].tolist(), "objectives": objectives,
                     "terminal_min_soc": case["home"]["min_soc"], "source_case": str((prior / "cases" / row.case_id).resolve())})
        tasks.append(case)
    root.mkdir(parents=True)
    cases.to_csv(root / "cases.csv", index=False)
    design = {"configuration": asdict(config), "objectives": objectives, "previous_design": prior_design,
        "strategies": MODE_STRATEGIES, "source_evolution": source_changes,
        "scenario_directory": prior_repro["scenario_directory"],
        "terminal_accounting": "same fixed Phase 4.2 values; accounting only, no terminal 50% condition or search cost",
        "mean_soc": "mean of hourly (initial_SOC + final_SOC)/2, trapezoidal within-hour approximation",
        "optimality": "one shared A*; positive normalized scalar bound for economic/balanced; exact rational lexicographic UCS for autonomy",
        "causality": "frozen causal forecasts only, no oracle, test access or neural fitting",
        "solar": "same chronological independent Cordoba paths; ideal scenario solar forecast, not synchronized UCI/PVGIS measurements"}
    write_json(root / "design.json", design)  # Fixed objectives persisted before outcomes.
    sources = [*Path("src/bekantor_energy/energy").glob("*.py"), Path("src/bekantor_energy/contracts.py"), Path("configs/energy.modes.json")]
    write_json(root / "reproducibility.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "source_sha256": {str(p): sha256(p) for p in sources}, "validation_sha256": sha256(validation_path),
        "new_neural_training": False, "evaluation_partition": "validation", "workers": config.workers})
    started, results = perf_counter(), []
    description = cases.set_index("case_id").to_dict("index")

    def record(result):
        results.extend({"case_id": result["case_id"], **description[result["case_id"]], **m} for m in result["metrics"])
        count = len(results) // 4
        failures = sum(r["status"] != "completed" for r in results)
        write_json(root / "progress.json", {"completed_cases": count, "total_cases": len(tasks),
            "failed_strategy_runs": failures, "elapsed_seconds": perf_counter() - started})
        print(f"Completed {count}/{len(tasks)} mode cases; failed strategy runs={failures}", flush=True)

    if config.workers == 1:
        for case in tasks:
            record(run_mode_case(case, root / "cases" / case["case_id"]))
    else:
        with ProcessPoolExecutor(max_workers=config.workers) as pool:
            futures = {pool.submit(run_mode_case, c, root / "cases" / c["case_id"]): c for c in tasks}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    directory = root / "cases" / case["case_id"]
                    directory.mkdir(parents=True, exist_ok=True)
                    write_json(directory / "worker_failure.json", {"exception_type": type(error).__name__, "message": str(error)})
                    result = {"case_id": case["case_id"], "metrics": [{"strategy": n, "status": "worker_failed", "hours_completed": 0} for n in MODE_STRATEGIES]}
                record(result)
    results.sort(key=lambda row: (row["case_id"], MODE_STRATEGIES.index(row["strategy"])))
    summarize_continuous(results, root, prior_design["configuration"]["valuation_sensitivity"],
                         strategies=MODE_STRATEGIES, reference_strategy="economic")
    regression = verify_economic_regression(root, prior, cases.case_id)
    after = {p: sha256(Path(p)) for p in before}
    write_json(root / "preservation.json", {"verified": before == after, "original_artifacts_before": before,
        "original_artifacts_after": after, "authorized_source_evolution": source_changes,
        "meaning": "old result/data/model files unchanged; prior source bytes archived; active planner/evaluator changed explicitly"})
    if before != after:
        raise ValueError("Historical data/model/result or archived source changed")
    write_json(root / "completed.json", {"status": "completed", "cases": len(tasks), "strategy_runs": len(results),
        "failures": sum(r["status"] != "completed" for r in results), "economic_reference_matches": regression,
        "elapsed_seconds": perf_counter() - started, "new_neural_training": False, "evaluation_partition": "validation"})
    from .mode_plots import mode_figures
    mode_figures(root)
    return pd.DataFrame(results)
