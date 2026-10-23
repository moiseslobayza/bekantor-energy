"""Causal continuous outage evaluator with transparent input-keyed plan reuse."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter, time

import numpy as np
import pandas as pd

from ..config import HomeConfig
from ..contracts import EnergyAction, EnergyState, LoadForecast, PlanningInputs
from ..data.acquisition import sha256, write_json
from ..forecasting.windows import read_hourly
from .autonomy_diagnostics import oracle_windows
from .modes import ModeEnergyPlan
from .planner import PlannerConfig
from .resilience import (POLICIES, ReadinessRisk, ResilienceConfig, ResiliencePlanner,
                         realized_step, resilience_metrics, reserve_action)
from .simulator import EPS, PhysicalSimulator


def protected_snapshot(previous):
    previous = Path(previous)
    verified = json.loads((previous / "verification.json").read_text())
    preserved = json.loads((previous / "preservation.json").read_text())
    expected = {**preserved["before"], **verified["artifact_sha256"],
                str(previous / "verification.json"): sha256(previous / "verification.json")}
    for name in subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"]).decode().split("\0"):
        if name:
            expected[str(Path(name).resolve())] = sha256(Path(name))
    for name, checksum in expected.items():
        if Path(name).name in ("test.csv", "test.npz"):
            raise ValueError("Held-out data not permitted in preservation scope")
        if sha256(Path(name)) != checksum:
            raise ValueError(f"Historical input changed: {name}")
    return expected


def load_bases(prior, data_dir):
    prior, data_dir = Path(prior), Path(data_dir)
    reproducibility = json.loads((prior / "reproducibility.json").read_text())
    if sha256(data_dir / "validation.csv") != reproducibility["validation_sha256"]:
        raise ValueError("Validation checksum differs from frozen reference")
    validation = read_hourly(data_dir / "validation.csv")
    # Only past context for the first seasonal forecast, never fitting/scoring train.
    train_context = read_hourly(data_dir / "train.csv").tail(24)
    context = pd.concat([train_context, validation])
    bases = []
    for folder in sorted((prior / "cases").iterdir()):
        case = json.loads((folder / "case.json").read_text())
        hours = case["hours"]
        origins = pd.date_range(case["start"], periods=hours, freq="h")
        oracle = oracle_windows(validation, case["start"], hours)
        np.testing.assert_allclose(oracle[:, 0], case["actual_demand"], atol=1e-12, rtol=0)
        frozen = pd.read_csv(folder / "forecasts.csv.gz", parse_dates=["origin"])
        mlp = frozen.pivot(index="origin", columns="horizon", values="predicted_demand_kwh").reindex(origins).to_numpy()
        persistence = np.stack([context.reindex(pd.date_range(t - pd.Timedelta(hours=24), periods=24, freq="h")).load_kwh.to_numpy(float) for t in origins])
        if any(v.shape != (hours, 24) or not np.isfinite(v).all() or (v < 0).any() for v in (mlp, persistence, oracle)):
            raise ValueError("Incomplete causal prediction matrix")
        bases.append({"case": case, "forecasts": {"mlp": mlp, "persistence": persistence, "oracle": oracle}})
    return bases


def scenarios(hours, stage, tariff):
    if stage == "pilot" and tariff != "time_of_use":
        return []
    events = [("no_outage", None, 0)]
    onsets = (22,) if stage in ("pilot", "panel") else (6, 14, 22)
    durations = (6,) if stage in ("pilot", "panel") else (2, 6, 12)
    for onset in onsets:
        for duration in durations:
            start = 3 * 24 + onset
            if start + duration > hours:
                raise ValueError("Outage beyond evaluation interval")
            events.append((f"hour{onset:02d}_duration{duration:02d}", start, duration))
    return events


def source_signature():
    files = [Path(__file__), Path(__file__).with_name("resilience.py"), Path(__file__).with_name("planner.py"),
             Path(__file__).with_name("modes.py"), Path(__file__).with_name("search.py"), Path(__file__).with_name("simulator.py")]
    return {str(p): sha256(p) for p in files}


def restore_plan(data):
    values = dict(data)
    values["actions"] = tuple(EnergyAction(**a) for a in data["actions"])
    values["states"] = tuple(EnergyState(**s) for s in data["states"])
    values["objective_units"] = tuple(data["objective_units"])
    return ModeEnergyPlan(**values)


def validate_plan(plan, inputs, initial, settings):
    if len(plan.actions) != inputs.config.horizon_steps or len(plan.states) != len(plan.actions) + 1:
        raise ArithmeticError("Incomplete horizon returned")
    sim, state = PhysicalSimulator(), initial
    risk = ReadinessRisk(inputs.forecast.demand_kwh, inputs.solar_kwh, inputs.config, settings)
    score, cost = 0., 0.
    for h, action in enumerate(plan.actions):
        result = sim.step(state, action, inputs.forecast.demand_kwh[h], inputs.solar_kwh[h], inputs.config)
        if abs(result.state.stored_kwh - plan.states[h + 1].stored_kwh) > EPS:
            raise ArithmeticError("Predicted plan failed strict physical replay")
        state = result.state
        if state.step <= risk.starts:
            score += float(risk.phi(state.step, plan.states[h + 1].stored_kwh) / risk.starts)
        cost += result.cost
    if abs(score - plan.objective_primary) > 1e-8 or abs(cost - plan.predicted_cost) > 1e-8 or abs(cost - plan.objective_secondary) > 1e-8:
        raise ArithmeticError("Plan risk/monetary objective mismatch")


class PlanStore:
    """Key excludes outage calendar: reusable computation is a causal optimization.

    The key includes origin, forecasts, solar, tariffs, physical configuration,
    exact initial energy, policy and source signature. Every new/loaded unique
    plan is physically replayed. Historic/source cache files remain read-only.
    """
    def __init__(self, folder, source=None):
        self.folder, self.source = Path(folder), Path(source) if source else None
        self.memory = {}
        self.sources = source_signature()

    def get(self, policy, inputs, state, planner):
        payload = {"policy": policy, "origin": inputs.forecast.issued_at.isoformat(),
            "forecast": list(inputs.forecast.demand_kwh), "model_id": inputs.forecast.model_id,
            "solar": list(inputs.solar_kwh), "home": asdict(inputs.config), "initial_kwh": state.stored_kwh,
            "planner": asdict(planner.settings), "resilience": asdict(planner.resilience), "sources": self.sources}
        key = hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
        if key in self.memory:
            return self.memory[key], key, True
        destination = self.folder / f"{key}.json.gz"
        source = self.source / destination.name if self.source else None
        cached = destination if destination.exists() else source
        if cached is not None and cached.exists():
            with gzip.open(cached, "rt", encoding="utf-8") as stream:
                record = json.load(stream)
            if record["inputs"] != json.loads(json.dumps(payload)):
                raise ValueError("Cached plan input signature mismatch")
            plan = restore_plan(record["plan"])
            validate_plan(plan, inputs, state, planner.resilience)
            if cached != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(cached.read_bytes())
            hit = True
        else:
            plan = planner.plan(inputs, state)
            validate_plan(plan, inputs, state, planner.resilience)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(destination, "wt", encoding="utf-8") as stream:
                json.dump({"inputs": payload, "plan": asdict(plan)}, stream, allow_nan=False)
            hit = False
        self.memory[key] = plan
        return plan, key, hit


def replay_policy(base, event, policy, settings, resilience, store, folder, deadline=None):
    """Evaluator owns the event; planner only sees a forecast/solar/current state."""
    if policy not in POLICIES:
        raise ValueError("Unknown resilience policy")
    case = base["case"]
    home = HomeConfig(**case["home"])
    if resilience.operational_soc < home.min_soc:
        raise ValueError("Operational reserve below physical floor")
    event_id, outage_start, duration = event
    times = pd.date_range(case["start"], periods=case["hours"], freq="h")
    state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
    planner = ResiliencePlanner(settings, resilience)
    source = "persistence" if policy == "astar_persistence" else "oracle" if policy == "astar_oracle" else "mlp"
    rows, decisions, failure = [], [], None
    started = perf_counter()
    for t, now in enumerate(times):
        try:
            if deadline is not None and time() > deadline:
                raise TimeoutError("Predeclared stage wall-clock budget exhausted")
            up = not (outage_start is not None and outage_start <= t < outage_start + duration)
            demand, solar = case["actual_demand"][t], case["solar"][t]
            plan = None
            hit, key, target, achievable = False, None, None, None
            if not up:
                action = EnergyAction()  # Shared emergency replaces this; no forecast used.
            elif policy in ("reactive", "fixed_reserve"):
                target = home.min_soc * home.battery_capacity_kwh if policy == "reactive" else resilience.operational_soc * home.battery_capacity_kwh
                action = reserve_action(state.stored_kwh, target, demand, solar, home)
            else:
                forecast_values = tuple(float(v) for v in base["forecasts"][source][t])
                solar_values = tuple(case["solar"][t:t + 24])
                if policy == "adaptive_mlp":
                    risk = ReadinessRisk(forecast_values, solar_values, home, resilience)
                    target, achievable = risk.target()
                    action = reserve_action(state.stored_kwh, target, demand, solar, home)
                else:
                    targets = tuple(pd.date_range(now, periods=24, freq="h").to_pydatetime())
                    model_id = "ORACLE_DEMAND_ONLY_NO_OUTAGE_KNOWLEDGE" if source == "oracle" else "seasonal_24h" if source == "persistence" else case["model_id"]
                    forecast = LoadForecast(now.to_pydatetime(), targets, forecast_values, model_id)
                    shifted = replace(home, import_tariff_per_kwh=tuple(home.import_prices[dt.hour] for dt in targets))
                    inputs = PlanningInputs(forecast, solar_values, shifted, "generic_readiness_no_event_id")
                    plan, key, hit = store.get(policy, inputs, EnergyState(0, state.stored_kwh), planner)
                    action = plan.actions[0]
            result, row = realized_step(state, action, demand, solar, home, up, home.import_prices[now.hour], resilience)
            row.update(timestamp=now.isoformat(), step=t, policy=policy,
                tariff_sim_per_kwh=home.import_prices[now.hour], target_kwh=target, target_achievable=achievable,
                plan_id=key, cache_hit=hit, planning_seconds=plan.elapsed_seconds if plan else 0.,
                search_seconds_unique=plan.elapsed_seconds if plan and not hit else 0.,
                expanded_nodes=plan.expanded_nodes if plan else 0,
                expanded_nodes_unique=plan.expanded_nodes if plan and not hit else 0)
            if plan:
                decisions.append({"step": t, "timestamp": now.isoformat(), "plan_id": key,
                    "cache_hit": hit, "risk_critical_kwh": plan.objective_primary,
                    "predicted_cost_sim": plan.predicted_cost, "seconds": plan.elapsed_seconds, "expanded_nodes": plan.expanded_nodes})
            rows.append(row)
            state = result.state
        except Exception as error:
            failure = {"step": t, "timestamp": now.isoformat(), "type": type(error).__name__, "message": str(error)}
            break
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(folder / "ledger.csv", index=False)
    pd.DataFrame(decisions).to_csv(folder / "decisions.csv", index=False)
    write_json(folder / "failure.json", failure)
    metrics = resilience_metrics(rows, home, resilience, case["hours"])
    metrics.update({k: case[k] for k in ("case_id", "block_id", "solar_profile", "tariff_mode")},
        event_id=event_id, onset_step=outage_start, duration_hours=duration, policy=policy,
        status="failed" if failure else "completed", hours_completed=len(rows), elapsed_seconds=perf_counter() - started,
        planning_seconds=sum(r["planning_seconds"] for r in rows),
        search_seconds_unique=sum(r["search_seconds_unique"] for r in rows),
        expanded_nodes=sum(r["expanded_nodes"] for r in rows),
        expanded_nodes_unique=sum(r["expanded_nodes_unique"] for r in rows),
        cache_hits=sum(r["cache_hit"] for r in rows), output_dir=str(folder))
    write_json(folder / "metrics.json", metrics)
    return metrics


def run_base(base, root, stage, settings, resilience, cache_source, deadline):
    case = base["case"]
    root = Path(root)
    folder = root / "bases" / case["case_id"]
    folder.mkdir(parents=True)
    write_json(folder / "inputs.json", {"case": case, "forecasts": {k: v.tolist() for k, v in base["forecasts"].items()}})
    store = PlanStore(folder / "plans", Path(cache_source) / "bases" / case["case_id"] / "plans" if cache_source else None)
    results = []
    events = scenarios(case["hours"], stage, case["tariff_mode"])
    for policy in POLICIES:
        for event in events:
            result = replay_policy(base, event, policy, settings, resilience, store,
                    folder / "replays" / f"{policy}__{event[0]}", deadline)
            results.append(result)
            write_json(folder / "progress.json", {"completed_replays": len(results), "expected_replays": len(events) * len(POLICIES),
                       "failed_replays": sum(r["status"] != "completed" for r in results), "last": f"{policy}/{event[0]}"})
    return results


def run_experiment(config_path, root, stage, data_dir=Path("data/processed/uci"), cache_source=None):
    config = json.loads(Path(config_path).read_text())
    root, data_dir = Path(root), Path(data_dir)
    if root.exists():
        raise ValueError("New run directory required")
    if stage not in ("pilot", "full", "panel"):
        raise ValueError("Use pilot, full or panel stage")
    before = protected_snapshot(config["previous_run"])
    bases = load_bases(config["paired_reference"], data_dir)
    bases = [b for b in bases if scenarios(b["case"]["hours"], stage, b["case"]["tariff_mode"])]
    resilience = ResilienceConfig(**config["resilience"])
    settings = PlannerConfig(**config["planner"])
    root.mkdir(parents=True)
    expected_replays = sum(len(scenarios(b["case"]["hours"], stage, b["case"]["tariff_mode"])) * len(POLICIES) for b in bases)
    write_json(root / "design.json", {"stage": stage, "config": config, "policies": POLICIES,
        "expected_replays": expected_replays, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "source_sha256": source_signature(),
        "cache_source": str(cache_source) if cache_source else None,
        "validation_sha256": sha256(data_dir / "validation.csv"),
        "train_past_context_sha256": sha256(data_dir / "train.csv"),
        "proposal_sha256": sha256(Path("docs/phase4_5-design-proposal.md")),
        "test_access": False, "neural_training": False,
        "chronology": "same continuous UCI validation and independent chronological Cordoba PVGIS; ideal solar forecast"})
    write_json(root / "preservation.json", {"before": before})
    started, results = perf_counter(), []
    deadline = time() + config["stage_budget_seconds"][stage]
    with ProcessPoolExecutor(max_workers=config["workers"]) as pool:
        futures = {pool.submit(run_base, b, root, stage, settings, resilience, cache_source, deadline): b for b in bases}
        for future in as_completed(futures):
            base = futures[future]
            try:
                results.extend(future.result())
            except Exception as error:
                write_json(root / "bases" / base["case"]["case_id"] / "worker_failure.json", {"type": type(error).__name__, "message": str(error)})
                # Preserve successes already flushed by a worker, then explicitly mark missing tasks.
                existing = {p.parent.name: json.loads(p.read_text()) for p in (root / "bases" / base["case"]["case_id"] / "replays").glob("*/metrics.json")}
                for policy in POLICIES:
                    for event in scenarios(base["case"]["hours"], stage, base["case"]["tariff_mode"]):
                        results.append(existing.get(f"{policy}__{event[0]}", {"case_id": base["case"]["case_id"],
                            "policy": policy, "event_id": event[0], "status": "worker_failed", "hours_completed": 0}))
            pd.DataFrame(results).to_csv(root / "metrics.csv", index=False)
            write_json(root / "progress.json", {"finished_bases": sum(f.done() for f in futures), "bases": len(bases),
                "recorded_replays": len(results), "expected_replays": expected_replays,
                "failed_replays": sum(r["status"] != "completed" for r in results), "seconds": perf_counter() - started})
            print(f'{len(results)}/{expected_replays} replays; failed={sum(r["status"] != "completed" for r in results)}', flush=True)
    for name, checksum in before.items():
        if sha256(Path(name)) != checksum:
            raise ArithmeticError(f"Historical artifact changed: {name}")
    write_json(root / "preservation.json", {"before": before, "after_matches": True, "files": len(before)})
    write_json(root / "completed.json", {"replays": len(results), "expected_replays": expected_replays,
        "failed_replays": sum(r["status"] != "completed" for r in results), "seconds": perf_counter() - started,
        "historical_artifacts_preserved": True, "stage": stage})
    return pd.DataFrame(results)
