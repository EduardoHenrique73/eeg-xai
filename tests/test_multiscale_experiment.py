import numpy as np
import pytest

from app.ai_engine.multiscale_data import centered_context, event_aware_indices
from app.ai_engine.training import SeizureInterval


def test_centered_context_keeps_short_targets_and_pads_edf_edges():
    windows = np.arange(16, dtype=np.float32).reshape(16, 1)
    sequences = np.stack([windows[start : start + 8] for start in range(0, 9, 2)])
    metas = [{"context_start_seconds": float(start * 2)} for start in range(0, 9, 2)]

    context = centered_context(sequences, metas, input_steps=24, target_steps=8)

    assert context.shape == (5, 24, 1)
    np.testing.assert_array_equal(context[:, 8:16], sequences)
    assert context[0, 0, 0] == 0
    assert context[-1, -1, 0] == 15


def test_event_sampler_covers_onset_middle_offset_without_duplication():
    labels = np.asarray([0] * 4 + [1] * 20 + [0] * 5 + [1] * 8 + [0] * 4)
    metas = [{"start_seconds": float(i * 2)} for i in range(len(labels))]
    intervals = [SeizureInterval(8.0, 48.0), SeizureInterval(58.0, 74.0)]

    indices = event_aware_indices(labels, metas, intervals, max_positive=12)

    assert len(indices) == 12
    assert len(set(indices.tolist())) == 12
    assert {4, 23, 29, 36}.issubset(set(indices.tolist()))
    assert np.all(labels[indices] == 1)


@pytest.mark.parametrize("mode", ["bce", "boundary", "tversky", "continuity"])
def test_multiscale_model_outputs_one_probability_per_target_step(mode):
    from app.ai_engine.multiscale_segmentation import create_multiscale_model

    model = create_multiscale_model(
        input_steps=24, output_steps=8, n_channels=3,
        features_per_channel=15, variant="attention", loss_mode=mode,
    )
    x = np.zeros((2, 24, 45), dtype=np.float32)
    y = np.zeros((2, 8, 1), dtype=np.float32)
    weights = np.ones((2, 8), dtype=np.float32)

    assert model(x).shape == (2, 8, 1)
    assert np.isfinite(model.train_on_batch(x, y, sample_weight=weights)[0])


def test_missing_channel_stays_zero_after_shared_projection():
    import tensorflow as tf

    from app.ai_engine.multiscale_segmentation import create_multiscale_model

    model = create_multiscale_model(
        input_steps=8, output_steps=8, n_channels=2,
        features_per_channel=15, variant="tcn",
    )
    projected = tf.keras.Model(model.input, model.get_layer("mask_missing_channels").output)
    values = np.zeros((1, 8, 30), dtype=np.float32)
    values[:, :, :15] = 1.0

    assert np.allclose(projected(values).numpy()[:, :, 1, :], 0.0)


def test_morphology_fusion_preserves_temporal_output():
    from app.ai_engine.multiscale_segmentation import create_morphology_fusion_model

    model = create_morphology_fusion_model(
        sequence_length=8,
        n_channels=2,
        spectral_features=3,
        morphology_features=2,
    )

    prediction = model(np.zeros((2, 8, 10), dtype=np.float32), training=False)

    assert tuple(prediction.shape) == (2, 8, 1)


def test_raw_waveform_fusion_preserves_temporal_output():
    from app.ai_engine.multiscale_segmentation import create_raw_waveform_fusion_model

    model = create_raw_waveform_fusion_model(
        sequence_length=8, n_channels=2, spectral_features=3, raw_points=128,
    )
    prediction = model(
        [
            np.zeros((2, 8, 6), dtype=np.float32),
            np.zeros((2, 8, 2, 128), dtype=np.float32),
        ],
        training=False,
    )

    assert tuple(prediction.shape) == (2, 8, 1)
