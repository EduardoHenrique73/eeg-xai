from types import SimpleNamespace

import numpy as np

from scripts.experiment_multiresolution_v47 import (
    load_paired_training,
    pair_resolutions,
    summarize_event_scores,
)


class _FakeCache:
    def __init__(self, payload):
        self.payload = payload
        self._cache = {}

    def carregar_arquivo(self, file, **limits):
        return self.payload[(file, limits["max_normal_windows"])]


def test_pair_resolutions_preserves_coarse_order_and_skips_missing_contexts() -> None:
    coarse = [
        {"arquivo": "a.edf", "context_start_seconds": 4.0},
        {"arquivo": "a.edf", "context_start_seconds": 8.0},
        {"arquivo": "b.edf", "context_start_seconds": 2.0},
    ]
    fine = [
        {"arquivo": "b.edf", "context_start_seconds": 2.0},
        {"arquivo": "a.edf", "context_start_seconds": 4.0},
    ]

    coarse_indices, fine_indices = pair_resolutions(coarse, fine)

    assert coarse_indices.tolist() == [0, 2]
    assert fine_indices.tolist() == [1, 0]


def test_event_summary_reports_contrast_against_normal_tail() -> None:
    scores = np.asarray([0.1, 0.2, 0.8, 0.6], dtype=np.float32)
    metas = [
        {"start_seconds": 0, "end_seconds": 4, "label": 0},
        {"start_seconds": 4, "end_seconds": 8, "label": 0},
        {"start_seconds": 8, "end_seconds": 12, "label": 1},
        {"start_seconds": 12, "end_seconds": 16, "label": 1},
    ]
    intervals = [SimpleNamespace(start_seconds=9, end_seconds=15)]

    summary = summarize_event_scores(scores, metas, intervals)

    assert summary["events"][0]["max_score"] == float(scores[2])
    assert summary["events"][0]["contrast_vs_normal_p99"] > 0.5


def test_paired_training_uses_full_fine_context_for_coarse_sample() -> None:
    coarse_meta = [
        {"arquivo": "a.edf", "context_start_seconds": 4.0},
        {"arquivo": "a.edf", "context_start_seconds": 8.0},
    ]
    fine_meta = [
        {"arquivo": "a.edf", "context_start_seconds": 0.0},
        {"arquivo": "a.edf", "context_start_seconds": 4.0},
        {"arquivo": "a.edf", "context_start_seconds": 8.0},
    ]
    coarse = np.asarray([[[1.0]], [[2.0]]])
    fine = np.asarray([[[0.0]], [[10.0]], [[20.0]]])
    coarse_cache = _FakeCache({("a.edf", 48): (coarse, np.zeros(2), coarse_meta)})
    fine_cache = _FakeCache({("a.edf", None): (fine, np.zeros(3), fine_meta)})

    paired_coarse, paired_fine, metas = load_paired_training(
        coarse_cache, fine_cache, ["a.edf"],
    )

    assert paired_coarse[:, 0, 0].tolist() == [1.0, 2.0]
    assert paired_fine[:, 0, 0].tolist() == [10.0, 20.0]
    assert metas == coarse_meta
