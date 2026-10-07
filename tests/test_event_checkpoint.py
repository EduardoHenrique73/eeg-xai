import numpy as np

from scripts.experiment_expanded_train_v37 import (
    build_dual_path_scores,
    build_event_diagnostics,
    checkpoint_rank,
    evaluate_dual_path,
    restore_missing_channel_zeros,
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
