"""Optional, lazily imported serializable Keras objective; weights per target."""

from functools import lru_cache
import math


@lru_cache(maxsize=1)
def weighted_loss_type():
    # The caller configures CPU/determinism before importing TensorFlow here.
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="bekantor_energy")
    class PeakWeightedMSE(tf.keras.losses.Loss):
        """mean_{batch,horizon}(w(y_true) * (y_pred - y_true)^2).

        Targets use energy/scale, without centering. threshold_scaled therefore
        equals threshold_kwh/scale_kwh, with NO subtraction of the input mean.
        The per-horizon mean is returned to Keras, which averages the batch.
        Raw weights are 1 and 2, not normalized by batch weights or their sum.
        """

        def __init__(self, threshold_scaled, normal_weight=1.0, peak_weight=2.0,
                     name="peak_weighted_mse", reduction="sum_over_batch_size", **kwargs):
            super().__init__(name=name, reduction=reduction, **kwargs)
            for value in (threshold_scaled, normal_weight, peak_weight):
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("Finite positive threshold and weights required")
            self.threshold_scaled = float(threshold_scaled)
            self.normal_weight, self.peak_weight = float(normal_weight), float(peak_weight)

        def call(self, y_true, y_pred):
            y_pred = tf.convert_to_tensor(y_pred)
            y_true = tf.cast(y_true, y_pred.dtype)
            weights = tf.where(y_true > tf.cast(self.threshold_scaled, y_true.dtype),
                               tf.cast(self.peak_weight, y_true.dtype),
                               tf.cast(self.normal_weight, y_true.dtype))
            return tf.reduce_mean(weights * tf.square(y_pred - y_true), axis=-1)

        def get_config(self):
            return {**super().get_config(), "threshold_scaled": self.threshold_scaled,
                    "normal_weight": self.normal_weight, "peak_weight": self.peak_weight}

    return PeakWeightedMSE


def make_peak_weighted_mse(threshold_kwh, scale_kwh):
    if not math.isfinite(scale_kwh) or scale_kwh <= 0:
        raise ValueError("Finite positive energy scale required")
    return weighted_loss_type()(threshold_scaled=threshold_kwh / scale_kwh)
