"""Compact spatial/temporal segmentation models for controlled EEG experiments."""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def temporal_loss(mode: str, weight: float = 0.1) -> Any:
    """Return per-timestep losses so Keras can apply the training-only sample weights."""
    if mode not in {"bce", "boundary", "tversky", "continuity"}:
        raise ValueError(f"Unsupported temporal loss: {mode}")

    def loss(y_true: Any, y_pred: Any) -> Any:
        true = tf.cast(tf.squeeze(y_true, axis=-1), tf.float32)
        pred = tf.cast(tf.squeeze(y_pred, axis=-1), tf.float32)
        bce = tf.keras.backend.binary_crossentropy(true, pred)
        if mode == "bce":
            return bce
        if mode == "boundary":
            edge_true = tf.abs(true[:, 1:] - true[:, :-1])
            edge_pred = tf.abs(pred[:, 1:] - pred[:, :-1])
            edge = tf.pad(tf.square(edge_true - edge_pred), [[0, 0], [1, 0]])
            return bce + weight * edge
        if mode == "continuity":
            stable = 1.0 - tf.abs(true[:, 1:] - true[:, :-1])
            changes = tf.square(pred[:, 1:] - pred[:, :-1]) * stable
            return bce + weight * tf.pad(changes, [[0, 0], [1, 0]])
        tp = tf.reduce_sum(true * pred, axis=1, keepdims=True)
        fp = tf.reduce_sum((1.0 - true) * pred, axis=1, keepdims=True)
        fn = tf.reduce_sum(true * (1.0 - pred), axis=1, keepdims=True)
        positive = tf.cast(tf.reduce_sum(true, axis=1, keepdims=True) > 0, tf.float32)
        tversky = (tp + 1.0) / (tp + 0.3 * fp + 0.7 * fn + 1.0)
        return bce + weight * positive * (1.0 - tversky)

    return loss


def _residual_tcn(x: Any, dilation: int, width: int) -> Any:
    residual = x
    x = layers.Conv1D(
        width, 3, padding="same", dilation_rate=dilation,
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SpatialDropout1D(0.15)(x)
    x = layers.Conv1D(width, 1, padding="same")(x)
    return layers.Activation("relu")(layers.Add()([residual, x]))


def create_multiscale_model(
    *,
    input_steps: int,
    output_steps: int,
    n_channels: int,
    features_per_channel: int,
    variant: str,
    loss_mode: str = "bce",
) -> Any:
    """Project each channel before temporal fusion; preserve one output per target step."""
    if variant not in {"tcn", "multiscale", "attention"}:
        raise ValueError(f"Unsupported architecture variant: {variant}")
    if input_steps < output_steps or (input_steps - output_steps) % 2:
        raise ValueError("Input context must contain a centered target of even offset.")
    inputs = layers.Input(shape=(input_steps, n_channels * features_per_channel))
    x = layers.Reshape((input_steps, n_channels, features_per_channel))(inputs)
    presence = layers.Lambda(
        lambda values: tf.cast(tf.reduce_any(tf.abs(values) > 1e-6, axis=(1, 3)), tf.float32),
        name="observed_channels",
    )(x)
    presence_mask = layers.Reshape((1, n_channels, 1))(presence)
    x = layers.Dense(8, activation="relu", name="shared_channel_projection")(x)
    x = layers.LayerNormalization()(x)
    x = layers.Multiply(name="mask_missing_channels")([x, presence_mask])
    if variant == "attention":
        channel_summary = layers.Lambda(
            lambda values: tf.reduce_mean(values, axis=(1, 3)),
            name="channel_summary",
        )(x)
        channel_gate = layers.Dense(n_channels, activation="sigmoid", name="channel_gate")(
            channel_summary
        )
        channel_gate = layers.Multiply()([channel_gate, presence])
        channel_gate = layers.Reshape((1, n_channels, 1))(channel_gate)
        x = layers.Multiply()([x, channel_gate])
    x = layers.Reshape((input_steps, n_channels * 8))(x)
    x = layers.Conv1D(32, 1, activation="relu", name="spatial_fusion")(x)
    if variant in {"multiscale", "attention"}:
        branches = [
            layers.Conv1D(
                16, 3, padding="same", dilation_rate=dilation, activation="relu",
                kernel_regularizer=regularizers.l2(1e-4),
            )(x)
            for dilation in (1, 2, 4)
        ]
        x = layers.Conv1D(32, 1, activation="relu")(
            layers.Concatenate(axis=-1)(branches)
        )
    else:
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    for dilation in (1, 2, 4):
        x = _residual_tcn(x, dilation, 32)
    if input_steps > output_steps:
        margin = (input_steps - output_steps) // 2
        x = layers.Cropping1D((margin, margin), name="target_steps")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="ictal_probability")(x)
    model = tf.keras.Model(inputs, outputs, name=f"eeg_{variant}_segmentation")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=temporal_loss(loss_mode),
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def create_morphology_fusion_model(
    *,
    sequence_length: int,
    n_channels: int,
    spectral_features: int,
    morphology_features: int,
) -> Any:
    """CNN-BiLSTM with a zero-initialized residual morphology branch."""
    total_per_channel = spectral_features + morphology_features
    inputs = layers.Input(shape=(sequence_length, n_channels * total_per_channel))
    grouped = layers.Reshape((sequence_length, n_channels, total_per_channel))(inputs)
    spectral = layers.Lambda(lambda value: value[..., :spectral_features])(grouped)
    morphology = layers.Lambda(lambda value: value[..., spectral_features:])(grouped)
    spectral = layers.Reshape((sequence_length, n_channels * spectral_features))(spectral)
    morphology = layers.Reshape((sequence_length, n_channels * morphology_features))(morphology)
    spectral = layers.Conv1D(64, 3, activation="relu", padding="same", name="spectral_conv")(spectral)
    morphology = layers.SpatialDropout1D(0.25)(morphology)
    morphology = layers.Conv1D(
        64,
        3,
        padding="same",
        kernel_initializer="zeros",
        bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(1e-4),
        name="morphology_residual",
    )(morphology)
    x = layers.Add(name="spectral_plus_morphology")([spectral, morphology])
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
    model = tf.keras.Model(inputs, outputs, name="cnn_bilstm_morphology_residual")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def create_raw_waveform_fusion_model(
    *,
    sequence_length: int,
    n_channels: int,
    spectral_features: int,
    raw_points: int,
) -> Any:
    """Fuse the spectral baseline with a compact per-channel raw waveform branch."""
    spectral_input = layers.Input(
        shape=(sequence_length, n_channels * spectral_features), name="spectral_input",
    )
    raw_input = layers.Input(
        shape=(sequence_length, n_channels, raw_points), name="raw_waveform_input",
    )
    spectral = layers.Conv1D(
        64, 3, activation="relu", padding="same", name="spectral_conv",
    )(spectral_input)

    raw = layers.Reshape((sequence_length, n_channels, raw_points, 1))(raw_input)
    raw = layers.TimeDistributed(layers.SeparableConv2D(
        8, (1, 9), strides=(1, 2), padding="same", activation="relu",
        depthwise_regularizer=regularizers.l2(1e-4),
        pointwise_regularizer=regularizers.l2(1e-4),
    ))(raw)
    raw = layers.TimeDistributed(layers.SeparableConv2D(
        8, (1, 7), strides=(1, 2), padding="same", activation="relu",
        depthwise_regularizer=regularizers.l2(1e-4),
        pointwise_regularizer=regularizers.l2(1e-4),
    ))(raw)
    raw = layers.Lambda(lambda value: tf.reduce_mean(value, axis=3), name="raw_temporal_pool")(raw)
    raw = layers.Reshape((sequence_length, n_channels * 8))(raw)
    raw = layers.SpatialDropout1D(0.25)(raw)
    raw = layers.Conv1D(
        64, 1, padding="same", kernel_initializer="zeros", bias_initializer="zeros",
        kernel_regularizer=regularizers.l2(1e-4), name="raw_residual_projection",
    )(raw)

    x = layers.Add(name="spectral_plus_raw")([spectral, raw])
    x = layers.Dropout(0.2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.TimeDistributed(layers.Dense(32, activation="relu"))(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"))(x)
    model = tf.keras.Model(
        [spectral_input, raw_input], outputs, name="cnn_bilstm_raw_waveform_fusion",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model
