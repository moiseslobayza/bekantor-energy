"""Generate the single academic notebook from frozen core source fragments.

Run from the repository root. Core files are read, never modified. Only the
hourly sample and a 48-hour PVGIS excerpt are exported as small public assets.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import nbformat as nbf

from academic_data import canonical_csv, digest, sample_from_archive, download_archive

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "colab"
SOURCES = {}


def fragment(relative, name, methods=None):
    """Copy a top-level definition, optionally retaining selected class methods."""
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    SOURCES[relative] = digest(path.read_bytes())
    node = next(n for n in ast.parse(source).body if getattr(n, "name", None) == name)
    if methods is not None:
        node.body = [n for n in node.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) or n.name in methods]
        return ast.unparse(node)
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])]) - 1
    return "\n".join(source.splitlines()[start:node.end_lineno])


def method(relative, cls, name):
    path = ROOT / relative
    SOURCES[relative] = digest(path.read_bytes())
    source = path.read_text(encoding="utf-8")
    parent = next(n for n in ast.parse(source).body if getattr(n, "name", None) == cls)
    node = next(n for n in parent.body if getattr(n, "name", None) == name)
    return textwrap.dedent("\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]))


def prepare_assets(archive=None):
    assets = HERE / "assets"
    assets.mkdir(exist_ok=True)
    archive = archive or download_archive(ROOT / "experiments/phase5/build-data")
    hourly = sample_from_archive(archive)
    # Independent equivalence check against the original train, never test.csv.
    original = pd.read_csv(ROOT / "data/processed/uci/train.csv", index_col="timestamp", parse_dates=True)
    original = original.loc[hourly.index]
    np.testing.assert_allclose(hourly.load_kwh, original.load_kwh, atol=1e-12, rtol=0, equal_nan=True)
    np.testing.assert_array_equal(hourly.is_observed_complete, original.is_observed_complete)
    (assets / "consumption_hourly.csv").write_bytes(canonical_csv(hourly))
    solar_source = ROOT / "data/processed/pvgis/cordoba-20230115-8d.csv"
    solar = pd.read_csv(solar_source, parse_dates=["timestamp"]).iloc[:48].copy()
    solar = solar.rename(columns={"timestamp": "source_utc", "pv_kwh_estimate": "solar_kwh"})
    solar["source_local"] = solar.source_utc.dt.tz_convert("America/Argentina/Cordoba")
    solar["scenario_hour"] = np.arange(48)
    solar[["scenario_hour", "source_utc", "source_local", "solar_kwh"]].to_csv(
        assets / "solar_cordoba_48h.csv", index=False, float_format="%.9f", lineterminator="\n")
    uci_receipt = json.loads((ROOT / "data/raw/uci/household_power_consumption.metadata.json").read_text())
    solar_receipt = json.loads((ROOT / "data/raw/pvgis/cordoba-20230115-8d.metadata.json").read_text())
    metadata = {
        "selection": "Calendar fixed: first six full months of 2007; not selected by model/control outcomes",
        "uci": {"authors": "Georges Hebrail and Alice Berard", "doi": "10.24432/C58K54",
            "landing_url": "https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption",
            "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "download_url": uci_receipt["url"], "archive_sha256": uci_receipt["archive_sha256"],
            "original_train_end": "2009-09-20T12:00:00", "start": hourly.index.min().isoformat(),
            "end": hourly.index.max().isoformat(), "hours": len(hourly),
            "source_train_sha256": digest((ROOT / "data/processed/uci/train.csv").read_bytes()),
            "preprocessing": "GAP kW integrated as sum(min_count=60)/60 kWh; causal ffill <=5 minutes; unknown targets rejected",
            "precision": "CSV rounded to nine decimals; <=5e-10 kWh rounding; same values used in notebook",
            "equivalence_to_core_train": True, "timezone": "UCI unspecified wall clock, kept naive"},
        "pvgis": {"provider": "European Commission, Joint Research Centre, PVGIS 6; ERA5 reanalysis",
            "original_receipt": solar_receipt, "processed_source_sha256": digest(solar_source.read_bytes()),
            "sample_hours": 48, "selection": "First 48 chronological hours of the existing January 2023 profile",
            "attribution_url": "https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en",
            "reuse_url": "https://commission.europa.eu/legal-notice_en", "not_synchronized_with_uci": True,
            "units": "AC kWh per one-hour interval; PVGIS efficiency already applied once"},
        "academic_partitions": {"train_days": 126, "validation_days": 27, "test_days": 28,
            "all_within_original_train": True, "historical_test_used": False},
        "assets": {p.name: digest(p.read_bytes()) for p in sorted(assets.glob("*.csv"))},
    }
    (assets / "provenance.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build():
    assets = HERE / "assets"
    provenance = json.loads((assets / "provenance.json").read_text(encoding="utf-8"))
    consumption_sha = provenance["assets"]["consumption_hourly.csv"]
    solar_text = (assets / "solar_cordoba_48h.csv").read_text(encoding="utf-8")
    cells = []

    def md(text): cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # BEKANTOR Energy — Trabajo Final
    **Desarrollo de Sistemas de Inteligencia Artificial · Instituto Superior Santo Domingo**

    Una vivienda puede consumir energía solar, almacenar excedentes en una batería
    o importar electricidad. Predeciremos sus próximas 24 horas de consumo con una
    MLP y usaremos esa previsión para planificar con A*. Es un ejemplo académico
    reducido del proyecto BEKANTOR, no una certificación ni una instalación real.

    **Ejecución:** abrir en Google Colab, elegir CPU (sin acelerador) y ejecutar
    todas las celdas en orden. No hay archivos que subir, claves, modelos externos
    ni rutas de una computadora. UCI se descarga automáticamente (~20 MB).
    Las dependencias faltantes se instalan en la primera celda.

    El protocolo se fija antes de obtener resultados: enero–junio de 2007,
    particiones cronológicas, una MLP 64–32 y un episodio elegido por fecha y calidad,
    nunca por ahorro. Las conclusiones al final se calculan al ejecutar.
    """)
    md("""
    ## 1. Preparar CPU y bibliotecas
    TensorFlow/Keras implementa la red; NumPy, pandas y Matplotlib permiten
    preparar datos, calcular métricas y visualizar. Fijamos una semilla. Las
    versiones quedan visibles; distintos sistemas pueden variar ligeramente.
    """)
    code("""
    import os, sys, subprocess, importlib.util
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    for module, requirement in {"numpy": "numpy>=1.26,<3", "pandas": "pandas>=2.2,<4",
                                "matplotlib": "matplotlib>=3.8,<4", "tensorflow": "tensorflow>=2.16,<2.22"}.items():
        if importlib.util.find_spec(module) is None:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", requirement])

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import tensorflow as tf
    from IPython.display import display, Markdown
    from dataclasses import dataclass, asdict
    from types import SimpleNamespace
    from itertools import count
    from time import perf_counter
    import heapq, math, io, json

    SEED = 2026
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    plt.rcParams.update({"figure.figsize": (10, 3.5), "font.size": 11, "axes.grid": True,
                         "grid.alpha": .2, "axes.spines.top": False, "axes.spines.right": False})
    print({"Python": sys.version.split()[0], "TensorFlow": tf.__version__,
           "NumPy": np.__version__, "pandas": pd.__version__, "CPU": not tf.config.get_visible_devices("GPU")})
    """)
    md("""
    ## 2. Datos reales y muestra reproducible
    [UCI Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
    Hebrail y Berard, DOI **10.24432/C58K54**, licencia **CC BY 4.0**.
    Son mediciones de una vivienda francesa; no describen la vivienda del usuario.

    Leemos sólo el prefijo de 2006–junio de 2007 del archivo original. El train
    histórico del MVP termina el 20/09/2009; toda esta muestra está dentro de él.
    El **test académico** será nuevo y distinto del test histórico, que no se abre.

    `Global_active_power` es potencia media por minuto en **kW**. La energía
    horaria es `sum(potencia_minuto)/60`, en **kWh**. No se confunden W, kW,
    kWh ni capacidad almacenada. Conservamos timestamps sin inventar una zona UCI.
    Se permite relleno con el último valor de hasta cinco minutos; nunca con
    observaciones futuras. Horas todavía desconocidas siguen como NaN y no son
    etiquetas de entrenamiento/evaluación. La copia CSV se comprueba por SHA-256.
    """)
    code((HERE / "academic_data.py").read_text(encoding="utf-8"))
    code(f"""
    archive = download_archive("bekantor_academic")
    sample = sample_from_archive(archive)
    sample_bytes = canonical_csv(sample)
    assert digest(sample_bytes) == {consumption_sha!r}, "La muestra no coincide con la extracción versionada"
    hourly = pd.read_csv(io.BytesIO(sample_bytes), index_col="timestamp", parse_dates=["timestamp"],
                         dtype={{"is_observed_complete": "boolean"}})
    train = hourly.iloc[:126*24].copy()
    validation = hourly.iloc[126*24:153*24].copy()
    academic_test = hourly.iloc[153*24:].copy()
    assert train.index.max() < validation.index.min() < academic_test.index.min()
    assert academic_test.index.max() < pd.Timestamp(ORIGINAL_TRAIN_END)
    display(pd.DataFrame([{{"partición": name, "horas": len(frame), "desde": str(frame.index.min()),
        "hasta": str(frame.index.max()), "horas desconocidas": int(frame.load_kwh.isna().sum()),
        "horas con minutos imputados": int((frame.imputed_minutes > 0).sum())}}
        for name, frame in [("train académico", train), ("validación académica", validation), ("test académico", academic_test)]]))
    """)
    md("""
    ## 3. EDA y preprocesamiento sin fuga temporal
    Variables de trabajo: `timestamp` (hora), `load_kwh` (energía),
    `is_observed_complete` (60 minutos observados), `observed_minutes` e
    `imputed_minutes` (calidad). Exploramos **sólo train**: distribución y evolución.
    La calidad de cada partición se informa sin utilizar el test para decidir parámetros.
    """)
    code("""
    display(train[["load_kwh", "observed_minutes", "imputed_minutes"]].describe().round(3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    train.load_kwh.plot(ax=axes[0], color="#2166ac", lw=.7, title="Energía horaria — train académico")
    axes[0].set(xlabel="Fecha UCI, 2007", ylabel="kWh por hora")
    axes[1].hist(train.load_kwh.dropna(), bins=35, color="#2166ac", edgecolor="white")
    axes[1].set(title="Distribución — train académico", xlabel="kWh por hora", ylabel="Cantidad de horas")
    fig.tight_layout(); plt.show()
    """)
    md("""
    ### Ventanas 168 → 24 y normalización
    Cada entrada termina antes del origen; los 24 objetivos permanecen en su
    partición y deben estar observados. Validación/test pueden usar **pasado ya
    disponible** de particiones anteriores. Nunca se unen horas separadas por huecos.

    Las siguientes funciones se extraen del núcleo del proyecto. El normalizador
    se ajusta una vez con horas observadas de train: estandariza entradas y divide
    objetivos por la desviación, sin centrarlos. Así una salida `softplus` mantiene
    consumo no negativo. Las métricas finales revierten esa transformación a kWh.
    """)
    window_code = "\n\n".join(fragment("src/bekantor_energy/forecasting/windows.py", name)
                               for name in ("validate_hourly", "WindowSet", "make_windows"))
    normalizer = fragment("src/bekantor_energy/forecasting/normalization.py", "DemandNormalizer",
                          {"__post_init__", "fit", "inputs", "targets", "inverse_targets"})
    code(window_code + "\n\n" + normalizer)
    code("""
    # Stride de train reducido; validación/test no solapan sus 24 objetivos.
    w_train = make_windows(train, stride=3)
    w_val = make_windows(validation, history=train, stride=24)
    w_test = make_windows(academic_test, history=pd.concat([train, validation]), stride=24)
    normalizer = DemandNormalizer.fit(train)
    assert normalizer.fitted_end <= str(train.index.max().isoformat())
    assert w_val.origins.min() >= validation.index.min() and w_test.origins.min() >= academic_test.index.min()
    assert all(len(w.x) > 0 for w in (w_train, w_val, w_test))
    display(pd.DataFrame([w.report for w in (w_train, w_val, w_test)], index=["train", "validación", "test académico"])
            [["candidates", "accepted", "unknown_inputs", "unknown_targets"]])
    """)
    md("""
    ## 4. MLP: aprender por optimización de pesos
    Una **red multicapa (MLP)** combina neuronas densas: cada capa calcula una
    transformación de entradas y una activación no lineal. Usamos 168 valores,
    capas ocultas de **64 y 32** neuronas ReLU y 24 salidas no negativas.

    La pérdida **MSE** es el promedio de errores al cuadrado. La retropropagación
    calcula su gradiente respecto a los pesos; **Adam** adapta los pasos usando
    estimaciones de primer y segundo momento de esos gradientes. EarlyStopping
    restaura los pesos de menor pérdida en **validación**, sin consultar test.
    Fijamos una semilla, Adam 0,001, máximo 25 épocas y paciencia 4; no hacemos
    búsqueda de arquitecturas. La versión académica no incorpora calendario ni
    MSE ponderada, para aislar los conceptos esenciales.
    """)
    code("""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(168,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(24, activation="softplus"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.RootMeanSquaredError(name="rmse")])

    def batches(windows):
        ds = tf.data.Dataset.from_tensor_slices((normalizer.inputs(windows.x), normalizer.targets(windows.y))).batch(64)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        return ds.with_options(options)

    started = perf_counter()
    history = model.fit(batches(w_train), validation_data=batches(w_val), epochs=25, verbose=0,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])
    training_seconds = perf_counter() - started
    print(f"Parámetros: {model.count_params():,}; épocas: {len(history.history['loss'])}; CPU: {training_seconds:.2f} s")
    fig, ax = plt.subplots()
    ax.plot(np.arange(1, len(history.history['loss'])+1), history.history["loss"], label="Train")
    ax.plot(np.arange(1, len(history.history['loss'])+1), history.history["val_loss"], label="Validación", ls="--")
    ax.set(xlabel="Época", ylabel="MSE normalizado", title="Aprendizaje de la MLP 64–32")
    ax.legend(); fig.tight_layout(); plt.show()
    """)
    md("""
    ### Evaluación: MLP frente a persistencia de 24 horas
    Persistencia repite el último día observado. MAE promedia errores absolutos;
    RMSE penaliza más los grandes errores. Ambos se calculan en **kWh**, sobre
    exactamente los mismos objetivos. Sólo ahora se evalúa el test académico,
    una vez congelado el modelo; no se elige ninguna configuración con él.
    """)
    code(fragment("src/bekantor_energy/forecasting/baselines.py", "seasonal_persistence") + "\n\n" + textwrap.dedent("""
    def physical_metrics(actual, predicted):
        assert actual.shape == predicted.shape and np.isfinite(predicted).all()
        error = np.asarray(predicted, float) - np.asarray(actual, float)
        return {"MAE kWh": float(np.mean(np.abs(error))), "RMSE kWh": float(np.sqrt(np.mean(error**2)))}

    metrics_rows, predictions = [], {}
    for name, windows in [("Validación", w_val), ("Test académico", w_test)]:
        predicted = normalizer.inverse_targets(model(normalizer.inputs(windows.x), training=False).numpy())
        baseline = seasonal_persistence(windows.x, lag=24)
        predictions[name] = {"MLP": predicted, "Persistencia 24 h": baseline}
        for label, values in predictions[name].items():
            metrics_rows.append({"Partición": name, "Modelo": label, **physical_metrics(windows.y, values)})
    predictive_metrics = pd.DataFrame(metrics_rows)
    display(predictive_metrics.round(4))
    # Gráfico fijado: primera ventana aceptada de test académico, nunca la mejor.
    fig, ax = plt.subplots()
    ax.plot(range(1, 25), w_test.y[0], color="black", label="Real", marker=".")
    for label, values in predictions["Test académico"].items():
        ax.plot(range(1, 25), values[0], label=label, ls="--")
    ax.set(xlabel="Horizonte (horas)", ylabel="kWh por hora",
           title=f"Primera ventana de test académico: {w_test.origins[0]:%d/%m/%Y}")
    ax.legend(); fig.tight_layout(); plt.show()
    """))
    md("""
    ## 5. Espacio de estados, física y búsqueda A*
    El estado es **(hora, energía almacenada)**. Elegir otro nivel de batería
    representa cargar, descargar o mantener. Se limita capacidad, reserva 20 %,
    potencia 2,5 kW y eficiencias 95 %. La grilla es 0,5 kWh; el simulador conserva
    energía continua y el SOC no se redondea al ejecutar. Horizonte: 24 horas.

    En el bus AC: `solar + red + descarga = consumo servido + carga + recorte`.
    En batería: `E_siguiente = E + eta_c*carga - descarga/eta_d`.
    El costo de arista es `importación_kWh * tarifa_SIM_por_kWh`. No hay
    exportación, cortes ni cargas críticas en esta versión reducida.

    El simulador y los métodos físicos siguientes son extraídos del MVP,
    no copiados como otra implementación independiente. Sólo se simplifican los
    contenedores de configuración y la construcción del grafo, conservando
    fronteras físicas y energía inicial exacta. El generador registra sus hashes.
    """)
    contracts = "\n\n".join(fragment("src/bekantor_energy/contracts.py", n)
                              for n in ("EnergyState", "EnergyAction", "StepResult"))
    code("HomeConfig = SimpleNamespace  # Contenedor académico, sin archivos de configuración.\nEPS = 1e-9\n\n" + contracts)
    code("\n\n".join(fragment("src/bekantor_energy/energy/simulator.py", n)
                          for n in ("finite_nonnegative", "PhysicalSimulator", "SelfConsumptionStrategy")))
    md("""
    ### El algoritmo de búsqueda
    A* expande el estado de menor **f = g + h**: costo acumulado más una cota
    optimista del costo restante. La heurística ignora límites y cronología,
    usa todo el solar futuro de forma optimista y compra al menor precio futuro;
    por eso no sobreestima el óptimo. Con **h=0**, A* equivale a costo uniforme.
    Validaremos ambas opciones contra enumeración en un problema pequeño.

    La cola prioritaria y la reapertura de estados son las del núcleo original.
    La garantía se refiere al grafo discretizado y al pronóstico utilizado;
    no garantiza el mejor resultado con consumo real desconocido.
    """)
    code("\n\n".join(fragment("src/bekantor_energy/energy/search.py", n)
                          for n in ("ScalarArithmetic", "SearchResult", "astar_search")))
    constructor = '''class ProblemaBateria:
    """Adaptador horario reducido; aristas, heurística y metas del núcleo."""
    def __init__(self, inputs, initial, settings):
        self.inputs, self.initial, self.settings = inputs, initial, settings
        c = inputs.config
        self.horizon = c.horizon_steps
        assert len(inputs.forecast.demand_kwh) == len(inputs.solar_kwh) == self.horizon
        assert initial.step == 0 and all(v >= 0 for v in (*inputs.forecast.demand_kwh, *inputs.solar_kwh))
        self.reserve = c.min_soc * c.battery_capacity_kwh
        self.terminal = self.reserve
        steps = math.floor((c.battery_capacity_kwh - self.reserve) / settings.storage_step_kwh)
        self.levels = tuple(sorted(set([self.reserve + i*settings.storage_step_kwh for i in range(steps+1)]
                                      + [c.battery_capacity_kwh, initial.stored_kwh, self.terminal])))
        self.simulator = PhysicalSimulator()
'''
    for name in ("physical_successors", "successors", "heuristic", "grid_lower_bound", "is_goal"):
        constructor += "\n" + textwrap.indent(method("src/bekantor_energy/energy/planner.py", "BatterySearchProblem", name), "    ") + "\n"
    code(constructor)
    md("""
    ## 6. Solar independiente y ejemplo fijado previamente
    PVGIS 6, JRC/Comisión Europea, ERA5: **Córdoba −31,4201 / −64,1888**, 3 kWp,
    inclinación 30° hacia norte, eficiencia del sistema 0,86 ya aplicada.
    Usamos las primeras 48 horas del perfil modelado 15–16/01/2023, con timestamps
    UTC y locales conservados. El perfil pequeño se incluye en este notebook;
    no depende de GitHub ni de una API cambiante al ejecutar.

    Lo alineamos por hora ordinal con un día UCI: **no son mediciones
    sincronizadas**, ni de la misma vivienda/época. El solar futuro es idealizado.
    No repetimos artificialmente un mismo día. La energía solar se mide aquí en
    kWh por intervalo; no se vuelve a aplicar el rendimiento PVGIS.

    Episodio: **primera ventana de test académico aceptada a medianoche**.
    Ocurre después de toda la validación usada por EarlyStopping: los pesos
    seleccionados ya están congelados antes del episodio. Se fija por orden y
    completitud, antes de comparar costos; no se selecciona una estrategia con
    los resultados de test. Ejecutamos 24 h,
    replanificando cada hora con 168 observaciones previas. La previsión afecta
    el déficit estimado, la importación y los costos de las aristas de A*.
    """)
    code(f"SOLAR_CSV = {solar_text!r}\nassert digest(SOLAR_CSV.encode()) == {provenance['assets']['solar_cordoba_48h.csv']!r}\n" + textwrap.dedent("""
    solar_frame = pd.read_csv(io.StringIO(SOLAR_CSV))
    solar = solar_frame.solar_kwh.to_numpy(float)
    assert len(solar) == 48 and (solar >= 0).all()
    origin = w_test.origins[0]
    assert origin.hour == 0
    assert validation.index.max() < origin  # Selección de pesos anterior a la operación.
    episode_times = pd.date_range(origin, periods=24, freq="h")
    actual_demand = hourly.reindex(episode_times).load_kwh.to_numpy(float)
    assert hourly.reindex(episode_times).is_observed_complete.all()
    past_inputs = []
    for now in episode_times:
        past = hourly.loc[hourly.index < now].tail(168)
        assert len(past) == 168 and past.index.max() < now and past.load_kwh.notna().all()
        past_inputs.append(past.load_kwh.to_numpy(float))
    past_inputs = np.asarray(past_inputs)
    # Procesar las 24 entradas juntas ahorra CPU; cada fila sólo contiene su pasado.
    causal_forecasts = {
        "A* persistencia": seasonal_persistence(past_inputs, lag=24),
        "A* MLP": normalizer.inverse_targets(model(normalizer.inputs(past_inputs), training=False).numpy()),
    }
    prices = np.array([.08 if h < 7 else .38 if 18 <= h < 23 else .15 for h in range(24)])
    print("Episodio UCI:", origin, "— 24 h; solar independiente:", solar_frame.source_local.iloc[0])
    """))
    code("""
    def home_config(prices, capacity=5., initial_soc=.5):
        return SimpleNamespace(battery_capacity_kwh=capacity, min_soc=.2, initial_soc=initial_soc,
            max_charge_kw=2.5, max_discharge_kw=2.5, charge_efficiency=.95, discharge_efficiency=.95,
            step_hours=1., horizon_steps=len(prices), import_prices=tuple(prices), export_prices=(0.,)*len(prices))

    def plan(demand, sun, prices, energy, heuristic="relaxed", capacity=5.):
        home = home_config(prices, capacity)
        settings = SimpleNamespace(storage_step_kwh=.5, heuristic=heuristic, max_expanded=10000, max_seconds=10.)
        inputs = SimpleNamespace(config=home, solar_kwh=tuple(sun), forecast=SimpleNamespace(demand_kwh=tuple(demand)))
        problem = ProblemaBateria(inputs, EnergyState(0, energy), settings)
        result = astar_search(problem, problem.initial, settings, RuntimeError)
        return result, problem

    def replay(label):
        home = home_config(prices)
        state = EnergyState(0, home.initial_soc * home.battery_capacity_kwh)
        rows = []
        for t in range(24):
            if label == "Reactivo":
                action = SelfConsumptionStrategy().action(actual_demand[t], solar[t], home)
                nodes = 0
            else:
                forecast = causal_forecasts[label][t]  # No demanda futura realizada.
                result, _ = plan(forecast, solar[t:t+24], np.roll(prices, -t), state.stored_kwh)
                action, nodes = result.actions[0], result.expanded_nodes
            # Sólo la primera acción se ejecuta con la demanda realmente observada.
            step = PhysicalSimulator().step(state, action, actual_demand[t], solar[t], home,
                                             repair=True, import_price=prices[t])
            rows.append({**asdict(step), "initial_kwh": state.stored_kwh,
                         "stored_kwh": step.state.stored_kwh, "nodes": nodes})
            state = step.state
        ledger = pd.DataFrame(rows)
        assert ledger.constraint_violations.sum() == 0
        assert ledger.balance_residual_kwh.abs().max() < 1e-8 and ledger.storage_residual_kwh.abs().max() < 1e-8
        assert ledger.stored_kwh.between(1-1e-9, 5+1e-9).all() and ledger.unserved_kwh.sum() < 1e-9
        assert np.allclose(ledger.initial_kwh.iloc[1:], ledger.stored_kwh.iloc[:-1])
        initial, final = ledger.initial_kwh.iloc[0], ledger.stored_kwh.iloc[-1]
        gross = ledger.cost.sum()
        return ledger, {"Estrategia": label, "Costo bruto SIM": gross,
            "Costo ajustado SIM": gross + .1425*(initial-final), "Red kWh": ledger.grid_import_kwh.sum(),
            "SOC inicial %": 100*initial/5, "SOC final %": 100*final/5,
            "Violaciones": int(ledger.constraint_violations.sum()), "Nodos": int(ledger.nodes.sum())}

    ledgers, energy_rows = {}, []
    for label in ("Reactivo", "A* persistencia", "A* MLP"):
        ledgers[label], row = replay(label)
        energy_rows.append(row)
    energy_metrics = pd.DataFrame(energy_rows).set_index("Estrategia")
    display(energy_metrics.round(4))
    """)
    md("""
    ### Comparación bajo condiciones idénticas
    Todas las estrategias reciben el mismo consumo realizado, solar, tarifa,
    batería y SOC inicial. El control físico común recorta comandos incompatibles
    con el consumo realizado; no crea energía ni descargas no solicitadas.
    El reactivo usa sólo el desequilibrio presente. Los A* reciben sus respectivos
    pronósticos, ejecutan una acción y vuelven a planificar al observar la hora.

    Mostramos costo bruto y saldo de batería. El costo ajustado añade
    `0,1425*(E_inicial−E_final)` SIM, valoración ilustrativa fija de energía
    almacenada; **no es una factura**. Así no se atribuye a ahorro retirar
    más energía de la batería. Esta comparación de un día no es un estudio anual.
    """)
    code("""
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(range(24), actual_demand, label="Consumo UCI", color="black")
    axes[0].plot(range(24), solar[:24], label="Solar PVGIS independiente", color="#c47f00", ls="--")
    axes[0].set(ylabel="kWh por hora", title="Episodio académico, perfiles independientes")
    axes[0].legend()
    styles = [("#555555", "-"), ("#b35806", "--"), ("#2166ac", "-.")]
    for (label, ledger), (color, style) in zip(ledgers.items(), styles):
        axes[1].plot(range(25), np.r_[2.5, ledger.stored_kwh]/5*100, label=label, color=color, ls=style)
    axes[1].axhline(20, color="black", lw=.8, ls=":", label="Piso físico")
    axes[1].set(xlabel="Hora del escenario", ylabel="SOC (%)", ylim=(0, 105))
    axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.24), ncol=2)
    fig.tight_layout(); plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    energy_metrics[["Costo bruto SIM", "Costo ajustado SIM"]].plot.bar(ax=axes[0], rot=15, color=["#80b1d3", "#2166ac"])
    axes[0].set(ylabel="SIM por episodio", xlabel="", title="Costo simulado y valoración terminal")
    axes[0].set_ylim(0, 1.35*energy_metrics[["Costo bruto SIM", "Costo ajustado SIM"]].max().max())
    axes[0].legend(loc="upper right", fontsize=9)
    energy_metrics["Red kWh"].plot.bar(ax=axes[1], rot=15, color="#555555")
    axes[1].set(ylabel="kWh por episodio", xlabel="", title="Importación de red")
    fig.tight_layout(); plt.show()
    """)
    md("""
    ## 7. Comprobaciones de métricas, búsqueda y causalidad
    Verificamos una métrica calculable a mano, optimalidad en un grafo pequeño,
    dependencia real de A* respecto al pronóstico y separación pasado/futuro.
    Las aserciones de cada replay ya comprueban restricciones y conservación.
    """)
    code("""
    from functools import lru_cache
    known = physical_metrics(np.array([0., 2.]), np.array([1., 4.]))
    assert known["MAE kWh"] == 1.5 and np.isclose(known["RMSE kWh"], np.sqrt(2.5))
    a, toy = plan([.3, .9, .4], [0., .1, 0.], [.08, .38, .15], .5, capacity=1.)
    u, _ = plan([.3, .9, .4], [0., .1, 0.], [.08, .38, .15], .5, heuristic="zero", capacity=1.)
    @lru_cache(None)
    def enumerate_cost(state):
        if toy.is_goal(state): return 0.
        return min(edge + enumerate_cost(nxt) for _, nxt, edge in toy.successors(state))
    assert np.isclose(a.value, u.value) and np.isclose(a.value, enumerate_cost(toy.initial))
    low, _ = plan([0., 0.], [0., 0.], [.08, .38], .5, capacity=1.)
    high, _ = plan([0., 1.], [0., 0.], [.08, .38], .5, capacity=1.)
    assert low.actions[0] != high.actions[0]  # Cambiar pronóstico cambia decisión.
    changed = hourly.copy()
    changed.loc[changed.index >= origin, "load_kwh"] = 999.
    np.testing.assert_array_equal(hourly.loc[hourly.index < origin].tail(168).load_kwh,
                                  changed.loc[changed.index < origin].tail(168).load_kwh)
    assert normalizer.fitted_hours == int(train.is_observed_complete.sum())
    print("Comprobaciones aprobadas: métricas, A*/UCS/enumeración, integración predictiva, causalidad y física.")
    """)
    md("""
    ## 8. Resultados y conclusión
    La siguiente síntesis se genera desde las tablas de esta ejecución. No se
    exige que MLP o A* ganen; pronosticar con menor RMSE tampoco garantiza mejores
    decisiones energéticas. No se cambian parámetros al observar test.
    """)
    code("""
    test_rows = predictive_metrics[predictive_metrics.Partición == "Test académico"].set_index("Modelo")
    neural_rmse = test_rows.loc["MLP", "RMSE kWh"]
    baseline_rmse = test_rows.loc["Persistencia 24 h", "RMSE kWh"]
    comparison = "menor" if neural_rmse < baseline_rmse else "mayor o igual"
    display(Markdown(f"**Test académico:** RMSE MLP {neural_rmse:.4f} kWh frente a "
        f"{baseline_rmse:.4f} kWh de persistencia ({comparison})."))
    display(Markdown(f"Se evaluaron **{len(w_test.x)} ventanas no solapadas de 24 h** "
        f"({len(w_test.x)*24} objetivos horarios) en test académico. Las ventanas "
        "con entradas desconocidas u objetivos no observados se excluyeron, sin rellenarlos con el futuro."))
    for label in ("A* persistencia", "A* MLP"):
        delta = energy_metrics.loc[label, "Costo ajustado SIM"] - energy_metrics.loc["Reactivo", "Costo ajustado SIM"]
        display(Markdown(f"**{label}:** diferencia de costo ajustado frente a reactivo: {delta:+.4f} SIM "
                         "en este episodio simulado; el signo positivo indica mayor costo."))
    """)
    md("""
    ### Registro pequeño de reproducibilidad
    Se guardan tablas y arreglos de auditoría en el runtime, sin guardar pesos de
    modelos. No hay pasos manuales ni archivos locales previos que preparar.
    """)
    code("""
    output = Path("bekantor_academic")
    predictive_metrics.to_csv(output / "forecast_metrics.csv", index=False)
    energy_metrics.to_csv(output / "energy_metrics.csv")
    for label, ledger in ledgers.items():
        ledger.drop(columns="state").to_csv(output / (label.replace("*", "star").replace(" ", "_") + ".csv"), index=False)
    np.savez_compressed(output / "audit_arrays.npz", val_actual=w_val.y, test_actual=w_test.y,
        val_mlp=predictions["Validación"]["MLP"], val_persistence=predictions["Validación"]["Persistencia 24 h"],
        test_mlp=predictions["Test académico"]["MLP"], test_persistence=predictions["Test académico"]["Persistencia 24 h"],
        episode_past=past_inputs, episode_actual=actual_demand, episode_solar=solar[:24],
        episode_mlp=causal_forecasts["A* MLP"], episode_persistence=causal_forecasts["A* persistencia"])
    run_metadata = {"seed": SEED, "tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "cpu_only": not tf.config.get_visible_devices("GPU"), "epochs": len(history.history["loss"]),
        "training_seconds": training_seconds, "normalizer": asdict(normalizer),
        "episode_origin": origin.isoformat(), "model_selection_end": validation.index.max().isoformat(),
        "episode_partition": "academic_test", "academic_csv_sha256": digest(sample_bytes),
        "window_counts": {"train": len(w_train.x), "validation": len(w_val.x), "test": len(w_test.x)},
        "historical_test_used": False}
    (output / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")
    print("Tablas y auditoría guardadas en bekantor_academic/; no se guardaron modelos.")
    """)
    md("""
    **Conceptos relacionados, algoritmos distintos.** Adam optimiza pesos continuos
    reduciendo MSE mediante gradientes de ejemplos históricos. A* busca una
    secuencia de estados discretos minimizando costo energético con una red ya
    congelada. La integración es efectiva: las 24 salidas previstas determinan
    importaciones y costos de transición. A* no entrena la red ni utiliza gradientes.

    **Límites:** muestra y semilla únicas, demanda sin calendario explícito,
    distribución temporal cambiante, ventanas de evaluación limitadas, una sola
    vivienda y episodio. La grilla pierde acciones posibles; solar ideal y tarifas
    SIM no representan un contrato real. No se modelan degradación, límites de red,
    cortes, resiliencia, hardware ni respuesta de usuarios. SOC alto por sí solo
    no demuestra autonomía. UCI y PVGIS no están sincronizados. Esta entrega no
    prueba ahorro real ni reutiliza el test histórico del MVP.

    **Fuentes y método:** [UCI y licencia CC BY 4.0](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption),
    [PVGIS/JRC](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en),
    [aviso legal de reutilización UE](https://commission.europa.eu/legal-notice_en),
    [TensorFlow CPU](https://www.tensorflow.org/install/pip),
    [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping).
    Los metadatos del notebook y `colab/assets/provenance.json` identifican las
    fuentes, períodos, hashes y extracción de código. No requiere publicar nuevos
    assets en GitHub para funcionar: UCI se descarga y el perfil solar está incluido.
    """)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"name": "BEKANTOR_Energy_Trabajo_Final.ipynb", "provenance": []},
        "bekantor": {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
            "source_sha256": SOURCES, "academic_data_sha256": digest((HERE / "academic_data.py").read_bytes()),
            "provenance": provenance, "historical_test_used": False,
            "simplifications": "Consumption only MLP, first 24h academic test episode after model selection, economic A*, no outages/hardware/dashboard"}})
    nbf.validate(notebook)
    nbf.write(notebook, HERE / "BEKANTOR_Energy_Trabajo_Final.ipynb")
    print(f"Notebook generated: {len(cells)} cells; core source files: {len(SOURCES)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.prepare_assets: prepare_assets(args.archive)
    build()
