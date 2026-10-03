"""Temporal input transforms used by controlled EEG experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import resample_poly

from app.ai_engine.feature_extractor import normalizar_matriz_features_robusta


def append_temporal_deltas(
    sequences: np.ndarray,
    *,
    features_per_channel: int,
) -> np.ndarray:
    """Append first differences per channel without crossing sequence boundaries.

    The flattened input is expected to be ordered as contiguous feature groups
    for each EEG channel. The output keeps that grouping, storing the original
    features followed by their first differences for every channel.
    """
    values = np.asarray(sequences, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("sequences must have shape (n_sequences, n_steps, n_features).")
    if features_per_channel <= 0:
        raise ValueError("features_per_channel must be greater than zero.")
    if values.shape[-1] % features_per_channel:
        raise ValueError("The feature dimension must be divisible by features_per_channel.")

    n_sequences, n_steps, n_features = values.shape
    n_channels = n_features // features_per_channel
    by_channel = values.reshape(n_sequences, n_steps, n_channels, features_per_channel)
    deltas = np.diff(by_channel, axis=1, prepend=by_channel[:, :1])
    augmented = np.concatenate((by_channel, deltas), axis=-1)
    return augmented.reshape(n_sequences, n_steps, n_channels * features_per_channel * 2)


def append_per_channel_features(
    base: np.ndarray,
    extra: np.ndarray,
    *,
    base_features_per_channel: int,
    extra_features_per_channel: int,
) -> np.ndarray:
    """Append feature groups while preserving each channel's contiguous layout."""
    base_values = np.asarray(base, dtype=np.float32)
    extra_values = np.asarray(extra, dtype=np.float32)
    if base_values.shape[:2] != extra_values.shape[:2]:
        raise ValueError("Base and extra sequences must have matching sequence and step axes.")
    if base_values.shape[-1] % base_features_per_channel:
        raise ValueError("Invalid base channel layout.")
    n_channels = base_values.shape[-1] // base_features_per_channel
    if extra_values.shape[-1] != n_channels * extra_features_per_channel:
        raise ValueError("Extra features do not match the base channel count.")
    base_grouped = base_values.reshape(*base_values.shape[:2], n_channels, base_features_per_channel)
    extra_grouped = extra_values.reshape(*extra_values.shape[:2], n_channels, extra_features_per_channel)
    combined = np.concatenate((base_grouped, extra_grouped), axis=-1)
    return combined.reshape(*base_values.shape[:2], n_channels * combined.shape[-1])


def morphology_windows(signals: np.ndarray, *, window_samples: int, step_samples: int) -> np.ndarray:
    """Vectorized amplitude-invariant morphology for all windows and channels."""
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or window_samples < 4 or step_samples < 1:
        raise ValueError("Invalid signals or window configuration.")
    windows = np.lib.stride_tricks.sliding_window_view(values, window_samples, axis=1)
    windows = windows[:, ::step_samples, :]
    output = np.empty((windows.shape[1], windows.shape[0], 8), dtype=np.float32)
    for channel_index, channel_windows in enumerate(windows):
        centered = channel_windows - np.mean(channel_windows, axis=1, keepdims=True)
        scale = np.maximum(np.std(centered, axis=1, keepdims=True), 1e-12)
        normalized = centered / scale
        first = np.diff(normalized, axis=1)
        second = np.diff(first, axis=1)
        var_first = np.maximum(np.var(first, axis=1), 1e-12)
        mobility = np.sqrt(var_first)
        mobility_first = np.sqrt(np.maximum(np.var(second, axis=1), 0.0) / var_first)
        bits = normalized >= 0.0
        codes = (
            bits[:, :-2].astype(np.uint8) * 4
            + bits[:, 1:-1].astype(np.uint8) * 2
            + bits[:, 2:].astype(np.uint8)
        )
        counts = np.stack([(codes == code).sum(axis=1) for code in range(8)], axis=1)
        probabilities = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        nonzero = probabilities > 0
        safe_probabilities = np.where(nonzero, probabilities, 1.0)
        entropy = -np.sum(
            np.where(nonzero, probabilities * np.log(safe_probabilities), 0.0), axis=1
        )
        symbols = np.sum(nonzero, axis=1)
        entropy = np.divide(
            entropy,
            np.log(np.maximum(symbols, 2)),
            out=np.zeros_like(entropy),
            where=symbols > 1,
        )
        output[:, channel_index, :] = np.stack(
            (
                np.mean(np.abs(first), axis=1),
                mobility,
                mobility_first / np.maximum(mobility, 1e-12),
                np.mean(normalized[:, 1:] * normalized[:, :-1] < 0, axis=1),
                np.max(np.abs(normalized), axis=1),
                np.mean(normalized**3, axis=1),
                np.mean(normalized**4, axis=1) - 3.0,
                np.clip(entropy, 0.0, 1.0),
            ),
            axis=1,
        )
    flattened = output.reshape(output.shape[0], -1)
    return normalizar_matriz_features_robusta(flattened)


def sequence_windows_from_timeline(
    timeline: np.ndarray,
    metas: list[dict[str, object]],
    *,
    sequence_length: int,
    step_seconds: float,
    dtype: np.dtype[Any] | type[np.generic] = np.float32,
) -> np.ndarray:
    """Select contiguous timeline windows using sequence metadata."""
    starts = np.rint(
        np.asarray([float(meta["context_start_seconds"]) for meta in metas]) / step_seconds
    ).astype(int)
    indices = starts[:, None] + np.arange(sequence_length)[None, :]
    if indices.size and (indices.min() < 0 or indices.max() >= len(timeline)):
        raise ValueError("Sequence metadata falls outside the morphology timeline.")
    return np.asarray(timeline[indices], dtype=dtype)


def raw_waveform_windows(
    signals: np.ndarray,
    *,
    sfreq: float,
    target_sfreq: int = 64,
    context_seconds: float = 2.0,
    window_seconds: float = 4.0,
    step_seconds: float = 2.0,
) -> np.ndarray:
    """Return centered, scale-normalized raw snippets for each analysis window."""
    values = np.asarray(signals, dtype=np.float64)
    source_rate = int(round(sfreq))
    if values.ndim != 2 or source_rate <= 0 or target_sfreq <= 0:
        raise ValueError("Invalid raw EEG matrix or sampling rate.")
    if context_seconds > window_seconds or context_seconds <= 0:
        raise ValueError("Raw context must be positive and fit inside the analysis window.")
    divisor = int(np.gcd(source_rate, target_sfreq))
    downsampled = resample_poly(
        values, target_sfreq // divisor, source_rate // divisor, axis=1,
    )
    context_points = int(round(context_seconds * target_sfreq))
    step_points = int(round(step_seconds * target_sfreq))
    margin_points = int(round((window_seconds - context_seconds) * target_sfreq / 2.0))
    duration_seconds = values.shape[1] / float(sfreq)
    n_windows = max(0, int(np.floor((duration_seconds - window_seconds) / step_seconds)) + 1)
    starts = np.arange(n_windows) * step_points + margin_points
    snippets = np.lib.stride_tricks.sliding_window_view(
        downsampled, context_points, axis=1,
    )[:, starts, :]
    centered = snippets - np.mean(snippets, axis=-1, keepdims=True)
    scale = np.std(centered, axis=-1, keepdims=True)
    normalized = np.divide(
        centered,
        scale,
        out=np.zeros_like(centered),
        where=scale > 1e-12,
    )
    return np.clip(normalized.transpose(1, 0, 2), -6.0, 6.0).astype(np.float16)
