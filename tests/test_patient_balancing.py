import numpy as np
import pytest

from scripts.patient_balancing import balance_patient_class_weights, subject_for_file


def test_balancing_reduces_dominance_and_preserves_each_class_mass():
    labels = np.array([[[1], [0]], [[1], [0]], [[1], [0]], [[1], [0]]])
    weights = np.array([[3, 1], [3, 1], [3, 1], [3, 1]], dtype=np.float32)
    meta = [{"arquivo": name} for name in ["chb01_01.edf"] * 3 + ["chb02_01.edf"]]
    adjusted, audit = balance_patient_class_weights(weights, labels, meta, exponent=0.5)
    for label in (0, 1):
        mask = labels[..., 0] == label
        np.testing.assert_allclose(adjusted[mask].sum(), weights[mask].sum(), rtol=1e-6)
        patients = audit["classes"][str(label)]["patients"]
        assert 0.5 < patients["chb01"]["share_after"] < patients["chb01"]["share_before"]
    np.testing.assert_array_equal(weights, [[3, 1]] * 4)


def test_same_subject_and_zero_weight_are_preserved():
    labels = np.array([[1, 0], [1, 0], [1, 0]])
    weights = np.array([[2, 0], [2, 1], [2, 1]], dtype=np.float32)
    meta = [{"arquivo": name} for name in ["chb01_03.edf", "chb21_19.edf", "chb17a_04.edf"]]
    adjusted, audit = balance_patient_class_weights(weights, labels, meta, exponent=1.0)
    assert audit["subject_count"] == 2
    assert audit["sequences_by_subject"]["chb01"] == 2
    assert adjusted[0, 1] == 0
    assert subject_for_file("chb17a_04.edf") == "chb17"
    assert subject_for_file("chb21_19.edf") == "chb01"
    for label in (0, 1):
        masses = [item["after"] for item in audit["classes"][str(label)]["patients"].values()]
        np.testing.assert_allclose(masses[0], masses[1])


def test_disabled_balancing_is_bit_identical_even_with_missing_class():
    weights = np.array([[0.123456, 0.987654]], dtype=np.float32)
    adjusted, audit = balance_patient_class_weights(
        weights, np.zeros((1, 2, 1)), [{"arquivo": "chb03_01.edf"}], exponent=0.0,
    )
    np.testing.assert_array_equal(adjusted, weights)
    assert adjusted.dtype == weights.dtype
    assert audit["classes"]["1"]["patients"] == {}


@pytest.mark.parametrize("exponent", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_exponent(exponent):
    with pytest.raises(ValueError, match="exponent"):
        balance_patient_class_weights(np.ones((1, 1)), np.zeros((1, 1)),
                                      [{"arquivo": "chb03_01.edf"}], exponent=exponent)


def test_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="align"):
        balance_patient_class_weights(np.ones((1, 2)), np.zeros((1, 1)),
                                      [{"arquivo": "chb03_01.edf"}], exponent=0.5)
