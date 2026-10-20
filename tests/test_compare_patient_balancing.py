from copy import deepcopy

import pytest

from scripts.compare_patient_balancing import compare, normalize_recorded_baseline, point_at_cap


def fixture_report(exponent):
    point = {"cap": 0.75, "detected_events": 15, "total_events": 20,
             "localized_f1": 0.90, "edf_f1": 0.95, "event_f1": 0.71,
             "false_alarms_per_hour": 0.55, "event_precision": 0.68}
    payload = {"test_accessed": False, "train_files": ["chb01_03.edf"],
               "calibration_files": ["chb16_17.edf"], "architecture": "late_channel_fusion",
               "patient_weight_audit": {"exponent": exponent}}
    run = {"calibration_froc": [point], "event_diagnostics_by_fa_cap": {"0.75": []}}
    return {42: (payload, run)}


def test_uses_exact_cap_rather_than_higher_cap_best():
    run = fixture_report(0)[42][1]
    higher = {**run["calibration_froc"][0], "cap": 1.0, "detected_events": 18}
    run["calibration_froc"].append(higher)
    run["calibration_best"] = higher
    assert point_at_cap(run, 0.75)["detected_events"] == 15


def test_rejects_unpaired_seeds_and_different_splits():
    baseline, candidate = fixture_report(0), fixture_report(0.5)
    with pytest.raises(ValueError, match="seed sets"):
        compare(baseline, {43: candidate[42]}, 0.75)
    candidate[42][0]["train_files"] = ["chb15_01.edf"]
    with pytest.raises(ValueError, match="Uncontrolled"):
        compare(baseline, candidate, 0.75)


def test_one_seed_cannot_promote_and_identifies_tradeoff():
    baseline, candidate = fixture_report(0), fixture_report(0.5)
    candidate[42][1]["calibration_froc"][0]["detected_events"] = 16
    result = compare(baseline, candidate, 0.75)
    assert result["pareto_improvements"] == 1
    assert result["promotion"] is False
    tradeoff = deepcopy(candidate)
    tradeoff[42][1]["calibration_froc"][0]["localized_f1"] = 0.80
    assert compare(baseline, tradeoff, 0.75)["pareto_improvements"] == 0


def test_recorded_defaults_apply_only_to_known_v43_and_preserve_explicit_values():
    legacy = {"experiment": "v43_late_channel_fusion", "learning_rate": 0.0001}
    normalized = normalize_recorded_baseline(legacy)
    assert normalized["learning_rate"] == 0.0001
    assert normalized["checkpoint_rank_mode"] == "event_localization"
    assert "checkpoint_rank_mode" not in legacy
    assert "learning_rate" not in normalized["recorded_v43_defaults_applied"]
    assert "patient_weight_audit" not in normalize_recorded_baseline({"experiment": "unknown"})


def test_confirmation_requires_multiple_seeds_and_preserved_means():
    baseline = {s: deepcopy(fixture_report(0)[42]) for s in (42, 43, 44)}
    candidate = {s: deepcopy(fixture_report(0.5)[42]) for s in (42, 43, 44)}
    for s in (42, 43):
        candidate[s][1]["calibration_froc"][0]["detected_events"] = 16
    assert compare(baseline, candidate, 0.75)["calibration_confirmation"] is True
    candidate[44][1]["calibration_froc"][0]["detected_events"] = 10
    result = compare(baseline, candidate, 0.75)
    assert result["calibration_confirmation"] is False
    assert result["status"] == "REJEITAR"
