"""TensorFlow/Keras MLP adapter implementing the Phase 1 DemandPredictor port."""

from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from ..contracts import LoadForecast
from ..data.acquisition import write_json
from .config import Architecture, ExperimentConfig
from .metrics import demand_metrics
from .normalization import DemandNormalizer
from .windows import WindowSet, make_windows, validate_hourly
from .calendar import CalendarInputNormalizer, FEATURE_MODES, calendar_features, with_calendar


def configure_tensorflow(seed: int, threads: int = 2):
    # Set before importing TF. No CUDA runtime or GPU is needed.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    for getter, setter, value in (
        (tf.config.threading.get_intra_op_parallelism_threads,
         tf.config.threading.set_intra_op_parallelism_threads, threads),
        (tf.config.threading.get_inter_op_parallelism_threads,
         tf.config.threading.set_inter_op_parallelism_threads, 1)):
        if getter() != value:
            setter(value)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def build_model(config: ExperimentConfig, architecture: Architecture, *, input_size: int | None = None, loss=None):
    tf = configure_tensorflow(config.seed, config.threads)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(config.seed)
    size = config.lookback if input_size is None else input_size
    if type(size) is not int or size <= 0:
        raise ValueError("input_size must be a positive integer")
    input_name = "past_hourly_kwh_scaled" if size == config.lookback else "past_kwh_and_known_calendar"
    layers = [tf.keras.layers.Input(shape=(size,), name=input_name)]
    for i, units in enumerate(architecture.hidden_units):
        layers.append(tf.keras.layers.Dense(units, activation=architecture.activation,
                      kernel_regularizer=tf.keras.regularizers.l2(architecture.l2) if architecture.l2 else None,
                      name=f"hidden_{i + 1}"))
        if architecture.dropout:
            layers.append(tf.keras.layers.Dropout(architecture.dropout, seed=config.seed + i))
    layers.append(tf.keras.layers.Dense(config.horizon, activation="softplus", name="future_hourly_kwh_scaled"))
    model = tf.keras.Sequential(layers, name=f"mlp_{architecture.name}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  loss="mse" if loss is None else loss,
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae"),
                           tf.keras.metrics.RootMeanSquaredError(name="rmse")])
    return model


def _dataset(tf, x, y, config: ExperimentConfig, training: bool):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if training:
        dataset = dataset.shuffle(len(x), seed=config.seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return dataset.batch(config.batch_size).with_options(options).prefetch(1)


def predict_windows(model, x: np.ndarray, normalizer: DemandNormalizer, batch_size: int = 256) -> np.ndarray:
    if len(x) == 0:
        raise ValueError("Cannot predict an empty window set")
    scaled = normalizer.inputs(x)
    outputs = [model(scaled[start:start + batch_size], training=False).numpy()
               for start in range(0, len(x), batch_size)]
    return normalizer.inverse_targets(np.concatenate(outputs))


def train_one(train: WindowSet, validation: WindowSet, normalizer: DemandNormalizer,
              config: ExperimentConfig, architecture: Architecture, directory: Path, *,
              input_size: int | None = None, score_training: bool = True, loss=None):
    if not len(train.x) or not len(validation.x):
        raise ValueError("Training and validation require eligible windows")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tf = configure_tensorflow(config.seed, config.threads)
    model = build_model(config, architecture, input_size=input_size, loss=loss)
    checkpoint = directory / "best.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_rmse", mode="min", patience=config.patience,
                                         min_delta=0.0, restore_best_weights=False),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_rmse", mode="min",
                                           save_best_only=True),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.CSVLogger(str(directory / "epochs.csv")),
    ]
    before = perf_counter()
    history = model.fit(
        _dataset(tf, normalizer.inputs(train.x), normalizer.targets(train.y), config, True),
        validation_data=_dataset(tf, normalizer.inputs(validation.x), normalizer.targets(validation.y), config, False),
        epochs=config.epochs, callbacks=callbacks, verbose=2, shuffle=False).history
    elapsed = perf_counter() - before
    if not all(np.isfinite(values).all() for values in history.values()):
        raise ValueError("Nonfinite training history; run aborted before selection")
    # Use exactly the persisted minimum-validation-RMSE checkpoint.
    model = tf.keras.models.load_model(checkpoint, compile=False)
    train_scores = demand_metrics(train.y, predict_windows(model, train.x, normalizer)) if score_training else None
    validation_scores = demand_metrics(validation.y, predict_windows(model, validation.x, normalizer))
    report = {"architecture": asdict(architecture), "parameters": model.count_params(),
              "epochs_run": len(history["loss"]), "best_epoch": int(np.argmin(history["val_rmse"]) + 1),
              "training_seconds": elapsed, "train": train_scores, "validation": validation_scores,
              "history_units": "loss: (kWh / scale_kwh)^2; mae/rmse: kWh / scale_kwh",
              "checkpoint_monitor": "val_rmse", "history": history,
              "loss": "mse" if loss is None else tf.keras.losses.serialize(loss),
              "training_score_computed": score_training, "input_size": model.input_shape[-1]}
    write_json(directory / "training.json", report)
    return model, report


class MLPDemandPredictor:
    def __init__(self, config: ExperimentConfig, architecture: Architecture,
                 artifact_dir: Path, model_id: str | None = None, feature_mode: str = "consumption_only"):
        if feature_mode not in FEATURE_MODES:
            raise ValueError("Unsupported feature_mode")
        self.config, self.architecture = config, architecture
        self.artifact_dir = Path(artifact_dir)
        self.model_id = model_id or f"mlp_{architecture.name}"
        self.feature_mode = feature_mode
        self.normalizer = None
        self.model = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
        self.normalizer = DemandNormalizer.fit(train)
        kwargs = {"lookback": self.config.lookback, "horizon": self.config.horizon, "stride": self.config.stride}
        training = make_windows(train, **kwargs)
        validating = make_windows(validation, history=train, **kwargs)
        if self.feature_mode == "future_calendar":
            training, validating = with_calendar(training), with_calendar(validating)
        self.model, report = train_one(training, validating, self.input_normalizer(), self.config,
                                       self.architecture, self.artifact_dir,
                                       input_size=training.x.shape[1])
        return {"train_rmse_kwh": report["train"]["rmse_kwh"],
                "validation_rmse_kwh": report["validation"]["rmse_kwh"]}

    def predict(self, history: pd.DataFrame, interval_starts: Sequence[datetime]) -> LoadForecast:
        if self.model is None or self.normalizer is None:
            raise ValueError("Load or fit a model before inference")
        validate_hourly(history)
        recent = history.tail(self.config.lookback)
        if len(recent) != self.config.lookback or not np.isfinite(recent.load_kwh).all():
            raise ValueError("Inference requires a complete known lookback")
        expected_history = pd.date_range(recent.index[0], periods=self.config.lookback, freq="h")
        if not recent.index.equals(expected_history):
            raise ValueError("Inference history must be contiguous hourly data")
        expected = pd.date_range(recent.index[-1] + pd.Timedelta(hours=1),
                                 periods=self.config.horizon, freq="h")
        requested = pd.DatetimeIndex(interval_starts)
        if not requested.equals(expected):
            raise ValueError("Forecast must cover exactly the next consecutive horizon intervals")
        inputs = recent.load_kwh.to_numpy()[None, :]
        if self.feature_mode == "future_calendar":
            calendar = calendar_features(pd.DatetimeIndex([expected[0]]), self.config.horizon).reshape(1, -1)
            inputs = np.concatenate([inputs, calendar], axis=1)
        values = predict_windows(self.model, inputs, self.input_normalizer())[0]
        return LoadForecast(issued_at=expected[0].to_pydatetime(),
                            interval_starts=tuple(expected.to_pydatetime()),
                            demand_kwh=tuple(float(value) for value in values), model_id=self.model_id)

    def save(self, path: Path) -> None:
        if self.model is None or self.normalizer is None:
            raise ValueError("No fitted model to save")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        self.normalizer.save(path / "normalization.json")
        write_json(path / "predictor.json", {"model_id": self.model_id, "feature_mode": self.feature_mode,
                                            "config": self.config.to_dict(),
                                            "architecture": asdict(self.architecture)})

    def input_normalizer(self):
        if self.normalizer is None:
            raise ValueError("Normalizer not fitted or loaded")
        if self.feature_mode == "future_calendar":
            return CalendarInputNormalizer(self.normalizer, self.config.lookback, self.config.horizon)
        return self.normalizer

    @classmethod
    def load(cls, path: Path) -> "MLPDemandPredictor":
        path = Path(path)
        metadata = json.loads((path / "predictor.json").read_text(encoding="utf-8"))
        config = ExperimentConfig(**metadata["config"])
        result = cls(config, Architecture(**metadata["architecture"]), path, metadata["model_id"],
                     feature_mode=metadata.get("feature_mode", "consumption_only"))
        tf = configure_tensorflow(config.seed, config.threads)
        result.model = tf.keras.models.load_model(path / "model.keras", compile=False)
        result.normalizer = DemandNormalizer.load(path / "normalization.json")
        return result
