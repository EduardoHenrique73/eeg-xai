from copy import deepcopy

import pytest

from scripts.summarize_channel_dropout import summarize


def reports():
    point = dict(cap=0.75, detected_events=15, localized_f1=0.91,
                 edf_f1=0.96, false_alarms_per_hour=0.55)
    baseline = dict(
        test_accessed=False, patient_weight_audit={"exponent": 0.0},
        channel_dropout=0.0,
        runs=[dict(seed=42, checkpoint_best_epoch=14, calibration_froc=[point],
                   event_diagnostics_by_fa_cap={"0.75": []})],
    )
    candidate = deepcopy(baseline)
    candidate["channel_dropout"] = 0.1
    return baseline, candidate


def test_more_events_required_and_localization_preserved():
    baseline, candidate = reports()
    assert not summarize(baseline, candidate)["rows"][0]["continue_gate"]
    candidate["runs"][0]["calibration_froc"][0]["detected_events"] = 16
    assert summarize(baseline, candidate)["rows"][0]["continue_gate"]
    candidate["runs"][0]["calibration_froc"][0]["localized_f1"] = 0.85
    assert not summarize(baseline, candidate)["rows"][0]["continue_gate"]


def test_rejects_split_drift_or_historical_access():
    baseline, candidate = reports()
    candidate["train_files"] = ["chb15_10.edf"]
    with pytest.raises(ValueError, match="Uncontrolled"):
        summarize(baseline, candidate)
    candidate.pop("train_files")
    candidate["test_accessed"] = True
    with pytest.raises(ValueError, match="Calibration-only"):
        summarize(baseline, candidate)
