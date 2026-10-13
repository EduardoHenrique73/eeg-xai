import numpy as np
import pytest

from scripts.experiment_expanded_train_v37 import (
    build_dual_path_scores,
    build_event_diagnostics,
    checkpoint_rank,
    create_event_checkpoint_callback,
    evaluate_dual_path,
    restore_missing_channel_zeros,
    selecionar_hard_negatives,
)


def test_event_checkpoint_rejects_invalid_evaluation_interval():
    with pytest.raises(ValueError, match="at least 1"):
        create_event_checkpoint_callback(
            calibration_x=np.zeros((1, 1, 1), dtype=np.float32),
            calibration_y=np.zeros((1, 1, 1), dtype=np.float32),
            calibration_temporal_meta=[[{
                "arquivo": "a.edf", "start_seconds": 0.0, "end_seconds": 1.0,
            }]],
            max_false_alarms_per_hour=0.75,
            patience=1,
            min_epochs=1,
            evaluate_every_epochs=0,
        )


def _metrics(*, recall: float, fa_h: float, localized_f1: float = 0.7):
    return {
        "event_recall": recall,
        "localized_f1": localized_f1,
        "event_precision": 0.8,
        "false_alarms_per_hour": fa_h,
        "edf_f1": 0.9,
    }


def test_feasible_checkpoint_beats_higher_recall_above_fa_limit():
    feasible = checkpoint_rank(
        _metrics(recall=0.6, fa_h=0.4), constraint_satisfied=True,
    )
    excessive_false_alarms = checkpoint_rank(
        _metrics(recall=0.9, fa_h=0.8), constraint_satisfied=False,
    )

    assert feasible > excessive_false_alarms


def test_checkpoint_prefers_event_recall_inside_fa_limit():
    lower_recall = checkpoint_rank(
        _metrics(recall=0.6, fa_h=0.3), constraint_satisfied=True,
    )
    higher_recall = checkpoint_rank(
        _metrics(recall=0.7, fa_h=0.5), constraint_satisfied=True,
    )

    assert higher_recall > lower_recall


def test_checkpoint_uses_localization_to_break_recall_tie():
    weaker_location = checkpoint_rank(
        _metrics(recall=0.7, fa_h=0.3, localized_f1=0.6),
        constraint_satisfied=True,
    )
    stronger_location = checkpoint_rank(
        _metrics(recall=0.7, fa_h=0.5, localized_f1=0.8),
        constraint_satisfied=True,
    )

    assert stronger_location > weaker_location


def test_checkpoint_edf_mode_prefers_edf_before_event_precision():
    lower_edf = _metrics(recall=0.7, fa_h=0.4, localized_f1=0.8)
    lower_edf.update(edf_f1=0.85, event_precision=0.9)
    higher_edf = _metrics(recall=0.7, fa_h=0.4, localized_f1=0.8)
    higher_edf.update(edf_f1=0.95, event_precision=0.7)

    assert checkpoint_rank(
        higher_edf, constraint_satisfied=True, mode="event_localization_edf",
    ) > checkpoint_rank(
        lower_edf, constraint_satisfied=True, mode="event_localization_edf",
    )


def test_event_diagnostics_includes_detected_and_missed_events():
    starts = [float(value) for value in range(0, 28, 2)]
    meta = [
        {"arquivo": "sample.edf", "start_seconds": start, "end_seconds": start + 4.0}
        for start in starts
    ]
    y_true = np.asarray([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    scores = np.asarray([0.1, 0.8, 0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    operating_point = {
        "threshold": 0.5,
        "min_duration_seconds": 4.0,
        "max_gap_seconds": 0.0,
    }

    diagnostics = build_event_diagnostics(y_true, scores, meta, operating_point)

    assert len(diagnostics) == 2
    assert diagnostics[0]["detected"] is True
    assert diagnostics[0]["max_ictal_score"] == 0.9
    assert diagnostics[1]["detected"] is False
    assert diagnostics[1]["max_ictal_score"] == 0.2


def test_dual_path_recovers_short_high_confidence_event_only():
    starts = [float(value) for value in range(0, 36, 2)]
    meta = [
        {"arquivo": "sample.edf", "start_seconds": start, "end_seconds": start + 4.0}
        for start in starts
    ]
    y_true = np.asarray(
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    scores = np.asarray(
        [0.1, 0.1, 0.92, 0.94, 0.91, 0.1, 0.1, 0.1, 0.1, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    sustained = {
        "threshold": 0.5,
        "min_duration_seconds": 18.0,
        "max_gap_seconds": 0.0,
    }

    result = evaluate_dual_path(
        y_true,
        scores,
        meta,
        sustained=sustained,
        transient_threshold=0.9,
        transient_min_duration_seconds=6.0,
    )

    assert result["detected_seizure_events"] == 1
    assert result["false_alarm_events"] == 0
    assert result["dual_path"]["transient_runs"] == 1

    dual_scores, _ = build_dual_path_scores(
        scores,
        meta,
        sustained=sustained,
        transient_threshold=0.9,
        transient_min_duration_seconds=6.0,
    )
    diagnostics = build_event_diagnostics(y_true, dual_scores, meta, result)
    assert diagnostics[0]["detected"] is True


def test_restore_missing_channel_zeros_after_scaling():
    original = np.asarray([[[1.0, 2.0, 0.0, 0.0], [2.0, 3.0, 0.0, 0.0]]])
    scaled = np.full_like(original, 4.0)

    restored = restore_missing_channel_zeros(
        original, scaled, n_channels=2, features_per_channel=2,
    )

    np.testing.assert_array_equal(restored[..., :2], 4.0)
    np.testing.assert_array_equal(restored[..., 2:], 0.0)


def test_hard_negative_selection_excludes_ictal_and_repeated_alarm():
    scores = np.asarray([
        [[0.95], [0.92]],
        [[0.94], [0.91]],
        [[0.99], [0.98]],
        [[0.90], [0.89]],
    ])
    targets = np.asarray([
        [[0.0], [0.0]],
        [[0.0], [0.0]],
        [[1.0], [1.0]],
        [[0.0], [0.0]],
    ])
    metas = [
        {"context_start_seconds": 0.0, "context_end_seconds": 16.0},
        {"context_start_seconds": 4.0, "context_end_seconds": 20.0},
        {"context_start_seconds": 100.0, "context_end_seconds": 116.0},
        {"context_start_seconds": 200.0, "context_end_seconds": 216.0},
    ]

    selected = selecionar_hard_negatives(
        scores,
        targets,
        metas,
        threshold=0.8,
        limit=3,
        min_separation_seconds=30.0,
    )

    assert selected == [0, 3]
