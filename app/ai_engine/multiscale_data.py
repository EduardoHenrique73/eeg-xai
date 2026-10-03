"""Event-aware sampling and longer input context for short segmentation targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai_engine.training import SeizureInterval


def centered_context(
    sequences: np.ndarray,
    metas: list[dict[str, Any]],
    *,
    input_steps: int = 24,
    target_steps: int = 8,
    step_seconds: float = 2.0,
    target_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Expand a short target using adjacent windows, without dropping edge targets."""
    if input_steps < target_steps or (input_steps - target_steps) % 2:
        raise ValueError("A centered context requires a nonnegative even margin.")
    if len(sequences) != len(metas) or sequences.shape[1] != target_steps:
        raise ValueError("Sequences and metadata must describe the same target length.")
    if not len(sequences):
        return np.empty((0, input_steps, sequences.shape[-1]), dtype=sequences.dtype)
    starts = np.asarray([float(meta["context_start_seconds"]) for meta in metas])
    origin = starts.min()
    n_windows = int(round((starts.max() - origin) / step_seconds)) + target_steps
    windows = np.zeros((n_windows, sequences.shape[-1]), dtype=sequences.dtype)
    observed = np.zeros(n_windows, dtype=bool)
    for row, start in enumerate(starts):
        offset = int(round((start - origin) / step_seconds))
        windows[offset : offset + target_steps] = sequences[row]
        observed[offset : offset + target_steps] = True
    if not np.all(observed):
        raise ValueError("A full, contiguous EDF sequence set is required for context expansion.")
    targets = np.arange(len(sequences)) if target_indices is None else np.asarray(target_indices)
    margin = (input_steps - target_steps) // 2
    offsets = np.rint((starts[targets] - origin) / step_seconds).astype(int)
    positions = offsets[:, None] + np.arange(-margin, target_steps + margin)[None, :]
    return windows[np.clip(positions, 0, n_windows - 1)]


def event_aware_indices(
    labels: np.ndarray,
    metas: list[dict[str, Any]],
    intervals: list[SeizureInterval],
    *,
    max_positive: int = 32,
) -> np.ndarray:
    """Keep onset, middle and offset for each event before uniform filling."""
    labels = np.asarray(labels)
    groups: list[np.ndarray] = []
    assigned: set[int] = set()
    for interval in intervals:
        candidates = np.asarray([
            idx for idx, (label, meta) in enumerate(zip(labels, metas))
            if label == 1 and idx not in assigned
            and interval.start_seconds <= float(meta["start_seconds"]) < interval.end_seconds
        ], dtype=int)
        if candidates.size:
            groups.append(candidates)
            assigned.update(candidates.tolist())
    positives = np.flatnonzero(labels == 1)
    if not groups and positives.size:
        groups = [positives]
    if positives.size <= max_positive:
        return positives
    selected: set[int] = set()
    for group in groups:
        for idx in np.linspace(0, len(group) - 1, min(3, len(group)), dtype=int):
            selected.add(int(group[idx]))
    if len(selected) > max_positive:
        raise ValueError("Positive limit is too small to cover all event phases.")
    remaining = np.asarray([idx for idx in positives if idx not in selected], dtype=int)
    slots = max_positive - len(selected)
    if slots and remaining.size:
        selected.update(remaining[np.linspace(0, len(remaining) - 1, min(slots, len(remaining)), dtype=int)].tolist())
    return np.asarray(sorted(selected), dtype=int)
