import numpy as np
import pytest

from scripts.experiment_channel_mask_v32 import append_channel_presence, saturation_audit


def test_append_presence_preserves_features_and_marks_each_file():
    x = np.arange(2 * 8 * 30, dtype=np.float32).reshape(2, 8, 30)
    metas = [{"arquivo": "a.edf"}, {"arquivo": "b.edf"}]
    masks = {
        "a.edf": np.asarray([1, 0], dtype=np.float32),
        "b.edf": np.asarray([0, 1], dtype=np.float32),
    }

    result = append_channel_presence(x, metas, masks)

    assert result.shape == (2, 8, 32)
    np.testing.assert_array_equal(result[:, :, :30], x)
    np.testing.assert_array_equal(result[0, :, 30:], np.tile([1, 0], (8, 1)))
    np.testing.assert_array_equal(result[1, :, 30:], np.tile([0, 1], (8, 1)))


def test_append_presence_rejects_misaligned_metadata():
    with pytest.raises(ValueError, match="misaligned"):
        append_channel_presence(np.zeros((2, 8, 15)), [{"arquivo": "a.edf"}], {})


def test_saturation_audit_separates_ictal_and_normal():
    x = np.zeros((2, 8, 15), dtype=np.float32)
    x[1, 4, :5] = 10.0
    result = saturation_audit(x, np.asarray([0, 1]))

    assert result["normal"]["absolute_band_power_fraction"] == 0.0
    assert result["ictal_center"]["absolute_band_power_fraction"] == 1.0
    assert result["ictal_center"]["relative_band_power_fraction"] == 0.0
