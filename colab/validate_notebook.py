"""Execute the standalone notebook in a fresh kernel/CWD and audit its outputs.

Run with a clean virtual environment containing the public runtime dependencies,
nbformat, nbclient and ipykernel. No project installation is required.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import nbformat
from nbclient import NotebookClient
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "colab/BEKANTOR_Energy_Trabajo_Final.ipynb"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_outputs(runtime):
    folder = runtime / "bekantor_academic"
    meta = json.loads((folder / "run_metadata.json").read_text())
    assert meta["cpu_only"] and not meta["historical_test_used"]
    arrays = np.load(folder / "audit_arrays.npz")
    forecasts = pd.read_csv(folder / "forecast_metrics.csv")
    for partition, key in (("Validación", "val"), ("Test académico", "test")):
        for model, suffix in (("MLP", "mlp"), ("Persistencia 24 h", "persistence")):
            actual, predicted = arrays[key + "_actual"], arrays[key + "_" + suffix]
            assert actual.shape == predicted.shape and actual.shape[1] == 24
            assert np.isfinite(actual).all() and np.isfinite(predicted).all() and (predicted >= 0).all()
            row = forecasts[(forecasts.Partición == partition) & (forecasts.Modelo == model)].iloc[0]
            error = predicted.astype(float) - actual.astype(float)
            np.testing.assert_allclose([row["MAE kWh"], row["RMSE kWh"]],
                                       [np.abs(error).mean(), np.sqrt((error**2).mean())], atol=1e-10, rtol=0)
    hourly = pd.read_csv(ROOT / "colab/assets/consumption_hourly.csv", index_col="timestamp", parse_dates=True)
    origin = pd.Timestamp(meta["episode_origin"])
    assert meta["episode_partition"] == "academic_test"
    assert pd.Timestamp(meta["model_selection_end"]) < origin
    assert origin >= hourly.index[153*24]
    train = hourly.iloc[:126*24]
    observed = train[train.is_observed_complete].load_kwh
    np.testing.assert_allclose([meta["normalizer"]["mean_kwh"], meta["normalizer"]["scale_kwh"]],
                               [observed.mean(), observed.std(ddof=0)], atol=1e-12, rtol=0)
    for t in range(24):
        now = origin + pd.Timedelta(hours=t)
        past = hourly.loc[hourly.index < now].tail(168).load_kwh.to_numpy()
        np.testing.assert_array_equal(arrays["episode_past"][t], past)
        np.testing.assert_array_equal(arrays["episode_persistence"][t], past[-24:])
    strategy_metrics = pd.read_csv(folder / "energy_metrics.csv", index_col="Estrategia")
    residual_max = 0.
    for label in ("Reactivo", "A* persistencia", "A* MLP"):
        ledger = pd.read_csv(folder / (label.replace("*", "star").replace(" ", "_") + ".csv"))
        assert len(ledger) == 24
        np.testing.assert_allclose(ledger.demand_kwh, arrays["episode_actual"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(ledger.solar_kwh, arrays["episode_solar"], atol=1e-12, rtol=0)
        c, d, e0, e1 = ledger.charge_kwh, ledger.discharge_kwh, ledger.initial_kwh, ledger.stored_kwh
        residuals = [ledger.solar_kwh + ledger.grid_import_kwh + d - ledger.served_kwh - c - ledger.curtailed_kwh,
                     e0 + .95*c - d/.95 - e1,
                     e0 + ledger.solar_kwh + ledger.grid_import_kwh - e1 - ledger.served_kwh - ledger.curtailed_kwh - ledger.conversion_loss_kwh]
        residual_max = max(residual_max, *(float(np.abs(r).max()) for r in residuals))
        assert residual_max < 1e-8
        assert e0.iloc[0] == 2.5 and e1.between(1-1e-9, 5+1e-9).all()
        assert c.between(-1e-9, 2.5+1e-9).all() and d.between(-1e-9, 2.5+1e-9).all()
        assert not ((c > 1e-9) & (d > 1e-9)).any()
        assert ledger.constraint_violations.sum() == ledger.unserved_kwh.sum() == ledger.grid_export_kwh.sum() == 0
        np.testing.assert_allclose(e0.iloc[1:], e1.iloc[:-1], atol=1e-12, rtol=0)
        prices = np.array([.08 if h < 7 else .38 if 18 <= h < 23 else .15 for h in range(24)])
        np.testing.assert_allclose(ledger.cost, ledger.grid_import_kwh*prices, atol=1e-10, rtol=0)
        row = strategy_metrics.loc[label]
        np.testing.assert_allclose([row["Costo bruto SIM"], row["Red kWh"], row["SOC inicial %"], row["SOC final %"]],
            [ledger.cost.sum(), ledger.grid_import_kwh.sum(), 50., e1.iloc[-1]/5*100], atol=1e-9, rtol=0)
        assert not row["Violaciones"]
    assert sha(folder / "uci_household.zip") == "9f84b46ade8a2d8e1286ec4b2b6c2987a45a755c59f263be3b3b3d10dfbda3ff"
    return {"status": "passed", "independent_metric_pairs": 4, "causal_origins_checked": 24,
            "realized_physical_steps_checked": 72, "max_physical_residual_kwh": residual_max,
            "cpu_only": True, "private_project_package_installed": False, "run_metadata": meta,
            "forecast_metrics": forecasts.to_dict("records"), "energy_metrics": strategy_metrics.reset_index().to_dict("records")}


def validate(run_dir):
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    runtime = run_dir / "empty-runtime"
    runtime.mkdir()
    assert not list(runtime.iterdir())
    assert importlib.util.find_spec("bekantor_energy") is None, "Use a separate environment without the MVP installed"
    kernels = run_dir / "jupyter/kernels/bekantor-academic"
    kernels.mkdir(parents=True)
    (kernels / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "BEKANTOR clean CPU", "language": "python"}), encoding="utf-8")
    os.environ["JUPYTER_PATH"] = str(run_dir / "jupyter")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(run_dir / "jupyter/runtime")
    os.environ["IPYTHONDIR"] = str(run_dir / "ipython")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    started = perf_counter()
    try:
        NotebookClient(notebook, timeout=600, kernel_name="bekantor-academic", resources={"metadata": {"path": str(runtime)}}).execute()
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert [c.execution_count for c in code_cells] == list(range(1, len(code_cells)+1))
        assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
        result = audit_outputs(runtime)
        plots = run_dir / "plots"
        plots.mkdir()
        images = []
        for index, cell in enumerate(code_cells):
            for j, output in enumerate(cell.outputs):
                if "image/png" in output.get("data", {}):
                    path = plots / f"cell{index+1:02d}-{j}.png"
                    path.write_bytes(base64.b64decode(output.data["image/png"]))
                    images.append(str(path))
        assert len(images) == 5
        result.update(seconds=perf_counter()-started, executed_code_cells=len(code_cells), plots=images,
                      data_obtained_automatically_from_uci=True, google_colab_service_executed=False,
                      validation_environment={"python": sys.version, "isolated_prefix": sys.prefix})
        notebook.metadata["bekantor"]["execution_validation"] = {k: result[k] for k in
            ("status", "seconds", "executed_code_cells", "cpu_only", "google_colab_service_executed")}
        nbformat.validate(notebook)
        nbformat.write(notebook, NOTEBOOK)
        nbformat.write(notebook, run_dir / "executed.ipynb")
        result["notebook_sha256"] = sha(NOTEBOOK)
        (run_dir / "verification.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k not in ("forecast_metrics", "energy_metrics", "plots")}, indent=2))
    except Exception as error:
        nbformat.write(notebook, run_dir / "failed.ipynb")
        (run_dir / "execution_failure.json").write_text(json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    validate(parser.parse_args().run_dir)
