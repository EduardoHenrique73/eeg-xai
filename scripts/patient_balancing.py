"""Patient-balanced temporal loss weights for controlled training experiments."""

from __future__ import annotations

import re
from typing import Any

import numpy as np


def subject_for_file(filename: str) -> str:
    match = re.match(r"(chb\d+)", filename.lower())
    if match is None:
        raise ValueError(f"Unknown CHB-MIT case: {filename}")
    case = match.group(1)
    # CHB01 and CHB21 are recordings of the same person.
    return "chb01" if case == "chb21" else case


def balance_patient_class_weights(
    weights: np.ndarray,
    targets: np.ndarray,
    metadata: list[dict[str, Any]],
    *,
    exponent: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Temper patient dominance while preserving each class's total loss weight."""
    original = np.asarray(weights)
    labels = np.asarray(targets)
    if labels.ndim == 3 and labels.shape[-1] == 1:
        labels = labels[..., 0]
    if original.ndim != 2 or labels.shape != original.shape or len(metadata) != len(original):
        raise ValueError("Temporal weights, targets and metadata must align.")
    if not np.isfinite(exponent) or not 0.0 <= exponent <= 1.0:
        raise ValueError("exponent must be between zero and one.")
    if not np.all(np.isfinite(original)) or np.any(original < 0):
        raise ValueError("Weights must be finite and nonnegative.")
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Temporal targets must be binary.")

    subjects = np.asarray([subject_for_file(str(item["arquivo"])) for item in metadata])
    cases = sorted(set(subjects))
    adjusted = original.astype(np.float64, copy=True)
    class_audit: dict[str, Any] = {}
    for label in (0, 1):
        selected = labels == label
        masses = {
            case: float(np.sum(original[selected & (subjects == case)[:, None]], dtype=np.float64))
            for case in cases
        }
        total = sum(masses.values())
        active = [case for case in cases if masses[case] > 0.0]
        if not active:
            class_audit[str(label)] = {"total_before": total, "total_after": total, "patients": {}}
            continue
        mean_mass = total / len(active)
        factors = {case: (mean_mass / masses[case]) ** exponent for case in active}
        normalizer = total / sum(masses[case] * factors[case] for case in active)
        for case in active:
            adjusted[selected & (subjects == case)[:, None]] *= factors[case] * normalizer
        after = {
            case: float(np.sum(adjusted[selected & (subjects == case)[:, None]]))
            for case in active
        }
        class_audit[str(label)] = {
            "total_before": total,
            "total_after": sum(after.values()),
            "patients": {
                case: {
                    "before": masses[case], "after": after[case],
                    "share_before": masses[case] / total,
                    "share_after": after[case] / total,
                    "multiplier": factors[case] * normalizer,
                }
                for case in active
            },
        }
    # Preserve the old training path exactly when balancing is disabled.
    result = np.array(original, copy=True) if exponent == 0.0 else adjusted.astype(np.float32)
    return result, {
        "exponent": exponent,
        "same_subject_cases": ["chb01", "chb21"],
        "subject_count": len(cases),
        "sequences_by_subject": {case: int(np.sum(subjects == case)) for case in cases},
        "classes": class_audit,
    }
