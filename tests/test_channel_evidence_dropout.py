"""Check temporal coherence, inference identity and portable serialization."""

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from app.ai_engine.multiscale_segmentation import create_late_channel_fusion_bilstm


def test_channel_dropout_shared_over_time_and_features():
    layer = tf.keras.layers.Dropout(0.1, noise_shape=(None, 1, 23, 1))
    values = tf.ones((64, 8, 23, 16))
    result = layer(values, training=True).numpy()
    np.testing.assert_allclose(result, np.broadcast_to(result[:, :1, :, :1], result.shape))
    assert np.any(result == 0)
    assert np.any(result > 0)
    np.testing.assert_array_equal(layer(values, training=False).numpy(), values.numpy())


def test_candidate_inference_identical_with_same_weights_and_reload(tmp_path):
    kwargs = dict(sequence_length=8, n_channels=3, features_per_channel=2)
    baseline = create_late_channel_fusion_bilstm(**kwargs)
    candidate = create_late_channel_fusion_bilstm(**kwargs, channel_dropout=0.1)
    candidate.set_weights(baseline.get_weights())
    values = np.random.default_rng(42).normal(size=(2, 8, 6)).astype(np.float32)
    values[:, :, :2] = 0
    expected = baseline(values, training=False).numpy()
    np.testing.assert_allclose(candidate(values, training=False).numpy(), expected, atol=1e-7)
    assert candidate.count_params() == baseline.count_params()
    path = tmp_path / "candidate.keras"
    candidate.save(path)
    restored = tf.keras.models.load_model(path)
    np.testing.assert_allclose(restored(values, training=False).numpy(), expected, atol=1e-7)


@pytest.mark.parametrize("rate", [-0.1, 1.0, float("nan")])
def test_invalid_channel_dropout(rate):
    with pytest.raises(ValueError):
        create_late_channel_fusion_bilstm(
            sequence_length=8, n_channels=3, features_per_channel=2,
            channel_dropout=rate,
        )
