import numpy as np
import pytest

from app.ai_engine.temporal_representation import (
    append_per_channel_features,
    append_temporal_deltas,
    morphology_windows,
    raw_waveform_windows,
    sequence_windows_from_timeline,
)


def test_append_temporal_deltas_preserves_channel_groups() -> None:
    sequences = np.asarray(
        [[[1, 10, 100, 1000], [3, 15, 130, 900], [8, 19, 160, 850]]],
        dtype=np.float32,
    )

    transformed = append_temporal_deltas(sequences, features_per_channel=2)

    assert transformed.shape == (1, 3, 8)
    np.testing.assert_array_equal(transformed[0, 0], [1, 10, 0, 0, 100, 1000, 0, 0])
    np.testing.assert_array_equal(transformed[0, 1], [3, 15, 2, 5, 130, 900, 30, -100])
    np.testing.assert_array_equal(transformed[0, 2], [8, 19, 5, 4, 160, 850, 30, -50])


def test_append_temporal_deltas_does_not_cross_sequence_boundaries() -> None:
    sequences = np.asarray([[[1], [2]], [[100], [105]]], dtype=np.float32)

    transformed = append_temporal_deltas(sequences, features_per_channel=1)

    np.testing.assert_array_equal(transformed[:, 0, 1], [0, 0])
    np.testing.assert_array_equal(transformed[:, 1, 1], [1, 5])


def test_append_temporal_deltas_rejects_invalid_channel_layout() -> None:
    with pytest.raises(ValueError, match="divisible"):
        append_temporal_deltas(np.zeros((2, 3, 5), dtype=np.float32), features_per_channel=2)


def test_append_per_channel_features_keeps_channel_identity() -> None:
    base = np.asarray([[[1, 2, 10, 20]]], dtype=np.float32)
    extra = np.asarray([[[3, 30]]], dtype=np.float32)

    combined = append_per_channel_features(
        base, extra, base_features_per_channel=2, extra_features_per_channel=1,
    )

    np.testing.assert_array_equal(combined, [[[1, 2, 3, 10, 20, 30]]])


def test_morphology_windows_are_scale_invariant_and_sequence_aligned() -> None:
    t = np.linspace(0.0, 4.0, 40, endpoint=False)
    signals = np.stack((np.sin(2 * np.pi * t + 0.13), np.cos(2 * np.pi * t + 0.27)))
    timeline = morphology_windows(signals, window_samples=20, step_samples=10)
    scaled = morphology_windows(signals * 25.0, window_samples=20, step_samples=10)
    metas = [{"context_start_seconds": 0.0}, {"context_start_seconds": 1.0}]

    np.testing.assert_allclose(timeline, scaled, rtol=1e-5, atol=1e-5)
    selected = sequence_windows_from_timeline(
        timeline, metas, sequence_length=2, step_seconds=1.0,
    )
    np.testing.assert_array_equal(selected[0], timeline[:2])
    np.testing.assert_array_equal(selected[1], timeline[1:3])


def test_raw_waveform_windows_preserve_channels_and_are_scale_invariant() -> None:
    sfreq = 256.0
    t = np.arange(0.0, 8.0, 1.0 / sfreq)
    signals = np.stack((np.sin(2 * np.pi * 8 * t), np.sin(2 * np.pi * 13 * t + 0.2)))

    raw = raw_waveform_windows(signals, sfreq=sfreq)
    scaled = raw_waveform_windows(signals * 50.0, sfreq=sfreq)

    assert raw.shape == (3, 2, 128)
    assert raw.dtype == np.float16
    np.testing.assert_allclose(raw, scaled, rtol=2e-3, atol=2e-3)
