"""Audit calibration seizures that remain invisible to the v43 first stage."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai_engine.feature_extractor import (  # noqa: E402
    TIME_FREQUENCY_FEATURE_NAMES,
    extrair_metadados_edf,
)
from app.config import get_settings  # noqa: E402
from scripts.validate_sequence_by_patient import (  # noqa: E402
    SequenceDatasetCache,
    agrupar_por_paciente,
    carregar_arquivos,
    separar_treino_calibracao,
    validar_isolamento_fold,
)


def overlap_ratio(start: float, end: float, event_start: float, event_end: float) -> float:
    overlap = max(0.0, min(end, event_end) - max(start, event_start))
    return overlap / max(end - start, 1e-9)


def event_signature(
    sequences: np.ndarray,
    metas: list[dict[str, Any]],
    *,
    start_seconds: float,
    end_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median center-window signature and selected sequence indices."""
    indices = np.asarray([
        index for index, meta in enumerate(metas)
        if overlap_ratio(
            float(meta["start_seconds"]), float(meta["end_seconds"]),
            start_seconds, end_seconds,
        ) >= 0.5
    ], dtype=int)
    if not indices.size:
        return np.asarray([], dtype=np.float32), indices
    center = sequences.shape[1] // 2
    return np.median(sequences[indices, center, :], axis=0), indices


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def describe_signature(
    signature: np.ndarray,
    *,
    channels: list[str],
) -> dict[str, Any]:
    n_features = len(TIME_FREQUENCY_FEATURE_NAMES)
    grouped = signature.reshape(len(channels), n_features)
    channel_strength = np.mean(np.abs(grouped), axis=1)
    feature_strength = np.mean(np.abs(grouped), axis=0)
    top_channels = np.argsort(channel_strength)[::-1][:5]
    top_features = np.argsort(feature_strength)[::-1][:5]
    return {
        "top_channels": [
            {"channel": channels[index], "strength": float(channel_strength[index])}
            for index in top_channels
        ],
        "top_features": [
            {
                "feature": TIME_FREQUENCY_FEATURE_NAMES[index],
                "strength": float(feature_strength[index]),
            }
            for index in top_features
        ],
        "channel_concentration_top3": float(
            np.sum(np.sort(channel_strength)[-3:]) / max(np.sum(channel_strength), 1e-9)
        ),
    }


def local_normal_separation(
    sequences: np.ndarray,
    metas: list[dict[str, Any]],
    event_signature_value: np.ndarray,
    *,
    start_seconds: float,
    end_seconds: float,
    radius_seconds: float = 120.0,
) -> dict[str, Any]:
    center = sequences.shape[1] // 2
    local_indices = np.asarray([
        index for index, meta in enumerate(metas)
        if int(meta.get("label", 0)) == 0
        and float(meta["end_seconds"]) >= start_seconds - radius_seconds
        and float(meta["start_seconds"]) <= end_seconds + radius_seconds
    ], dtype=int)
    if not local_indices.size:
        return {"local_normal_windows": 0, "local_normal_rms_distance": None}
    local_signature = np.median(sequences[local_indices, center, :], axis=0)
    return {
        "local_normal_windows": int(local_indices.size),
        "local_normal_rms_distance": float(np.sqrt(np.mean(
            np.square(event_signature_value - local_signature)
        ))),
        "local_normal_cosine_similarity": cosine_similarity(
            event_signature_value, local_signature,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "dataset_amostra/manifests/v38_expanded_train.txt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "modelos/v45_invisible_event_audit.json",
    )
    parser.add_argument("--target-file", default="chb16_17.edf")
    parser.add_argument("--window-seconds", type=int, default=4)
    parser.add_argument("--step-seconds", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--training-files", nargs="*")
    args = parser.parse_args()

    files = carregar_arquivos(args.dataset_dir, args.manifest)
    by_patient = agrupar_por_paciente(files)
    reserved = {"chb09", "chb15", "chb18"}
    fixed_calibration = ["chb16", "chb19", "chb20", "chb24"]
    train_files, calibration_files, calibration_patients = separar_treino_calibracao(
        by_patient, "chb09", n_calibration_patients=4,
        reserved_test_patients=reserved,
        fixed_calibration_patients=fixed_calibration,
    )
    test_files = [file for patient in sorted(reserved) for file in by_patient[patient]]
    validar_isolamento_fold(
        train_files=train_files, calibration_files=calibration_files,
        test_files=test_files, reserved_test_patients=reserved,
    )
    if args.target_file not in calibration_files:
        raise RuntimeError("Target file must belong exclusively to calibration.")
    selected_training_files = args.training_files or train_files
    unknown_training_files = sorted(set(selected_training_files) - set(train_files))
    if unknown_training_files:
        raise RuntimeError(
            f"Files outside the isolated training split: {unknown_training_files}"
        )

    channels = list(
        extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"]
    )
    settings = get_settings()
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir,
        window_seconds=args.window_seconds, step_seconds=args.step_seconds,
        max_normal_windows=48, max_seizure_windows=32,
        sequence_length=args.sequence_length,
        sequence_stride=2, sequence_target_mode="center", sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5,
        feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel", canais_referencia=channels,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
        event_balanced_sampling=True,
    )

    target_x, _, target_meta = cache.carregar_arquivo(
        args.target_file, max_normal_windows=None, max_seizure_windows=None,
    )
    target_intervals = cache.intervalos_por_arquivo[args.target_file]
    training_events: list[dict[str, Any]] = []
    for file in selected_training_files:
        x, _, metas = cache.carregar_arquivo(
            file, max_normal_windows=48, max_seizure_windows=32,
        )
        for event_index, interval in enumerate(cache.intervalos_por_arquivo[file], start=1):
            signature, indices = event_signature(
                x, metas, start_seconds=interval.start_seconds,
                end_seconds=interval.end_seconds,
            )
            if signature.size:
                training_events.append({
                    "file": file,
                    "event": event_index,
                    "start_seconds": float(interval.start_seconds),
                    "end_seconds": float(interval.end_seconds),
                    "duration_seconds": float(interval.end_seconds - interval.start_seconds),
                    "windows": int(indices.size),
                    "signature": signature,
                })
        cache._cache.pop((file, 48, 32), None)

    target_events = []
    for event_index, interval in enumerate(target_intervals, start=1):
        signature, indices = event_signature(
            target_x, target_meta, start_seconds=interval.start_seconds,
            end_seconds=interval.end_seconds,
        )
        similarities = sorted(
            (
                {
                    key: value for key, value in event.items() if key != "signature"
                } | {"cosine_similarity": cosine_similarity(signature, event["signature"])}
                for event in training_events
            ),
            key=lambda item: item["cosine_similarity"], reverse=True,
        )
        target_events.append({
            "event": event_index,
            "start_seconds": float(interval.start_seconds),
            "end_seconds": float(interval.end_seconds),
            "duration_seconds": float(interval.end_seconds - interval.start_seconds),
            "windows": int(indices.size),
            **describe_signature(signature, channels=channels),
            **local_normal_separation(
                target_x, target_meta, signature,
                start_seconds=interval.start_seconds,
                end_seconds=interval.end_seconds,
            ),
            "nearest_training_events": similarities[:10],
            "nearest_short_training_events": [
                item for item in similarities if item["duration_seconds"] <= 20.0
            ][:10],
            "similar_training_events_ge_0_8": int(sum(
                item["cosine_similarity"] >= 0.8 for item in similarities
            )),
        })

    payload = {
        "experiment": "v45_invisible_event_representation_audit",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "target_file": args.target_file,
        "target_patient": "chb16",
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sequence_length": args.sequence_length,
        "context_span_seconds": (
            args.window_seconds + (args.sequence_length - 1) * args.step_seconds
        ),
        "calibration_patients": calibration_patients,
        "training_files": len(selected_training_files),
        "training_file_names": selected_training_files,
        "training_events_with_windows": len(training_events),
        "training_event_duration_counts": {
            "le_10_seconds": int(sum(
                event["duration_seconds"] <= 10.0 for event in training_events
            )),
            "le_20_seconds": int(sum(
                event["duration_seconds"] <= 20.0 for event in training_events
            )),
            "gt_20_seconds": int(sum(
                event["duration_seconds"] > 20.0 for event in training_events
            )),
        },
        "reference_channels": channels,
        "feature_names": list(TIME_FREQUENCY_FEATURE_NAMES),
        "target_events": target_events,
        "reserved_test_files_not_loaded": test_files,
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
