import numpy as np

from scripts.audit_invisible_events_v45 import (
    cosine_similarity,
    describe_signature,
    event_signature,
    local_normal_separation,
)


def test_event_signature_uses_center_windows_with_sufficient_overlap():
    sequences = np.zeros((3, 8, 4), dtype=np.float32)
    sequences[1, 4] = 2.0
    metas = [
        {"start_seconds": 0.0, "end_seconds": 4.0},
        {"start_seconds": 4.0, "end_seconds": 8.0},
        {"start_seconds": 8.0, "end_seconds": 12.0},
    ]

    signature, indices = event_signature(
        sequences, metas, start_seconds=5.0, end_seconds=7.0,
    )

    np.testing.assert_array_equal(indices, [1])
    np.testing.assert_array_equal(signature, np.full(4, 2.0))


def test_similarity_and_description_are_bounded_and_rank_channels():
    signature = np.concatenate((
        np.full(15, 3.0, dtype=np.float32),
        np.full(15, 0.1, dtype=np.float32),
    ))

    assert np.isclose(cosine_similarity(signature, signature), 1.0)
    description = describe_signature(signature, channels=["A", "B"])
    assert description["top_channels"][0]["channel"] == "A"
    assert 0.0 <= description["channel_concentration_top3"] <= 1.0


def test_local_normal_separation_uses_non_ictal_nearby_windows():
    sequences = np.zeros((3, 8, 2), dtype=np.float32)
    sequences[1, 4] = 2.0
    metas = [
        {"start_seconds": 0.0, "end_seconds": 4.0, "label": 0},
        {"start_seconds": 4.0, "end_seconds": 8.0, "label": 1},
        {"start_seconds": 8.0, "end_seconds": 12.0, "label": 0},
    ]

    result = local_normal_separation(
        sequences, metas, np.full(2, 2.0), start_seconds=4.0, end_seconds=8.0,
    )

    assert result["local_normal_windows"] == 2
    assert np.isclose(result["local_normal_rms_distance"], 2.0)
