"""Compare v43 against a dual 4 s + 2 s representation on hard calibration events.

This is an exploratory representation experiment. It never loads the reserved
test cohort and does not tune an operating point on the target EDF.
"""

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

from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402
from app.ai_engine.multiscale_segmentation import (  # noqa: E402
    create_dual_resolution_channel_fusion_bilstm,
    create_late_channel_fusion_bilstm,
    create_residual_dual_resolution_bilstm,
)
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.config import get_settings  # noqa: E402
from scripts.experiment_channel_mask_v32 import load_sequences  # noqa: E402
from scripts.experiment_expanded_train_v37 import restore_missing_channel_zeros  # noqa: E402
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    ajustar_scaler,
    aplicar_scaler,
    resolver_class_weight,
)
from scripts.validate_sequence_by_patient import (  # noqa: E402
    SequenceDatasetCache,
    agrupar_por_paciente,
    carregar_arquivos,
    separar_treino_calibracao,
    validar_isolamento_fold,
)


DEFAULT_TRAINING_FILES = [
    "chb07_12.edf", "chb07_13.edf", "chb07_19.edf",
    "chb05_06.edf", "chb03_01.edf", "chb02_19.edf",
    "chb21_22.edf", "chb12_08.edf", "chb06_10.edf",
    "chb06_13.edf", "chb06_18.edf",
]


def sequence_key(meta: dict[str, Any]) -> tuple[str, float]:
    return str(meta["arquivo"]), round(float(meta["context_start_seconds"]), 6)


def pair_resolutions(
    coarse_meta: list[dict[str, Any]], fine_meta: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pair contexts by EDF and start time while preserving coarse ordering."""
    fine_by_key = {sequence_key(meta): index for index, meta in enumerate(fine_meta)}
    coarse_indices: list[int] = []
    fine_indices: list[int] = []
    for index, meta in enumerate(coarse_meta):
        fine_index = fine_by_key.get(sequence_key(meta))
        if fine_index is not None:
            coarse_indices.append(index)
            fine_indices.append(fine_index)
    return np.asarray(coarse_indices, dtype=int), np.asarray(fine_indices, dtype=int)


def load_paired_training(
    coarse_cache: SequenceDatasetCache,
    fine_cache: SequenceDatasetCache,
    files: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Sample once on the coarse branch, then recover every matching fine context."""
    coarse_parts: list[np.ndarray] = []
    fine_parts: list[np.ndarray] = []
    paired_metas: list[dict[str, Any]] = []
    for file in files:
        coarse, _, coarse_meta = coarse_cache.carregar_arquivo(
            file, max_normal_windows=48, max_seizure_windows=32,
        )
        fine, _, fine_meta = fine_cache.carregar_arquivo(
            file, max_normal_windows=None, max_seizure_windows=None,
        )
        coarse_indices, fine_indices = pair_resolutions(coarse_meta, fine_meta)
        if len(coarse_indices) != len(coarse_meta):
            raise RuntimeError(
                f"Fine resolution is missing {len(coarse_meta) - len(coarse_indices)} "
                f"sampled contexts from {file}."
            )
        coarse_parts.append(coarse[coarse_indices])
        fine_parts.append(fine[fine_indices])
        paired_metas.extend(coarse_meta[index] for index in coarse_indices)
        coarse_cache._cache.pop((file, 48, 32), None)
        fine_cache._cache.pop((file, None, None), None)
    return np.concatenate(coarse_parts), np.concatenate(fine_parts), paired_metas


def summarize_event_scores(
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    intervals: list[Any],
) -> dict[str, Any]:
    normal_scores = np.asarray([
        score for score, meta in zip(scores, metas) if int(meta["label"]) == 0
    ])
    normal_p99 = float(np.quantile(normal_scores, 0.99)) if normal_scores.size else None
    events: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals, start=1):
        selected = np.asarray([
            score for score, meta in zip(scores, metas)
            if max(
                0.0,
                min(float(meta["end_seconds"]), interval.end_seconds)
                - max(float(meta["start_seconds"]), interval.start_seconds),
            ) > 0
        ])
        maximum = float(np.max(selected)) if selected.size else None
        events.append({
            "event": index,
            "start_seconds": float(interval.start_seconds),
            "end_seconds": float(interval.end_seconds),
            "duration_seconds": float(interval.end_seconds - interval.start_seconds),
            "max_score": maximum,
            "mean_score": float(np.mean(selected)) if selected.size else None,
            "contrast_vs_normal_p99": (
                maximum - normal_p99
                if maximum is not None and normal_p99 is not None else None
            ),
        })
    return {
        "normal_p95": float(np.quantile(normal_scores, 0.95)) if normal_scores.size else None,
        "normal_p99": normal_p99,
        "events": events,
    }


def make_cache(
    dataset_dir: Path, channels: list[str], *, window: int, step: int, length: int,
) -> SequenceDatasetCache:
    settings = get_settings()
    return SequenceDatasetCache(
        dataset_dir=dataset_dir, window_seconds=window, step_seconds=step,
        max_normal_windows=48, max_seizure_windows=32,
        sequence_length=length, sequence_stride=2,
        sequence_target_mode="center", sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5,
        feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel", canais_referencia=channels,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
        event_balanced_sampling=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "dataset_amostra/manifests/v45_short_seizures_train.txt",
    )
    parser.add_argument("--target-file", default="chb16_17.edf")
    parser.add_argument("--training-files", nargs="*", default=DEFAULT_TRAINING_FILES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument(
        "--variants", nargs="+",
        choices=[
            "v43_single_resolution", "v47_dual_resolution",
            "v48_residual_dual_resolution",
        ],
        default=[
            "v43_single_resolution", "v47_dual_resolution",
            "v48_residual_dual_resolution",
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "modelos/v47_multiresolution_prototype.json",
    )
    args = parser.parse_args()

    import tensorflow as tf

    files = carregar_arquivos(args.dataset_dir, args.manifest)
    by_patient = agrupar_por_paciente(files)
    reserved = {"chb09", "chb15", "chb18"}
    train_files, calibration_files, calibration_patients = separar_treino_calibracao(
        by_patient, "chb09", n_calibration_patients=4,
        reserved_test_patients=reserved,
        fixed_calibration_patients=["chb16", "chb19", "chb20", "chb24"],
    )
    test_files = [name for patient in sorted(reserved) for name in by_patient[patient]]
    validar_isolamento_fold(
        train_files=train_files, calibration_files=calibration_files,
        test_files=test_files, reserved_test_patients=reserved,
    )
    unknown = sorted(set(args.training_files) - set(train_files))
    if unknown:
        raise RuntimeError(f"Training files outside isolated split: {unknown}")
    if args.target_file not in calibration_files:
        raise RuntimeError("Target EDF must belong exclusively to calibration.")

    channels = list(extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"])
    coarse_cache = make_cache(args.dataset_dir, channels, window=4, step=2, length=8)
    fine_cache = make_cache(args.dataset_dir, channels, window=2, step=1, length=16)

    coarse_train, fine_train, coarse_train_meta = load_paired_training(
        coarse_cache, fine_cache, args.training_files,
    )
    coarse_target, _, coarse_target_meta = load_sequences(
        coarse_cache, [args.target_file], sampled=False,
    )
    fine_target, _, fine_target_meta = load_sequences(
        fine_cache, [args.target_file], sampled=False,
    )

    target_coarse_idx, target_fine_idx = pair_resolutions(coarse_target_meta, fine_target_meta)
    if not len(coarse_train) or not len(target_coarse_idx):
        raise RuntimeError("No aligned multiresolution sequences were found.")
    coarse_target = coarse_target[target_coarse_idx]
    fine_target = fine_target[target_fine_idx]
    coarse_target_meta = [coarse_target_meta[index] for index in target_coarse_idx]

    y_train, temporal_train = construir_alvos_temporais(
        coarse_train_meta, coarse_cache.intervalos_por_arquivo,
        sequence_length=8, window_seconds=4, step_seconds=2,
        min_ictal_overlap_ratio=0.5,
    )
    y_target, temporal_target = construir_alvos_temporais(
        coarse_target_meta, coarse_cache.intervalos_por_arquivo,
        sequence_length=8, window_seconds=4, step_seconds=2,
        min_ictal_overlap_ratio=0.5,
    )
    unique_labels, _, _ = agregar_predicoes_temporais(
        np.zeros_like(y_train), y_train, temporal_train,
    )
    weights = pesos_temporais(
        y_train, temporal_train, boundary_weight=1.5,
        class_weight=resolver_class_weight(
            unique_labels, mode="balanced", positive_weight=3.0,
        ),
    )

    n_channels = len(channels)
    coarse_features = coarse_train.shape[-1] // n_channels
    fine_features = fine_train.shape[-1] // n_channels
    if coarse_features != fine_features:
        raise RuntimeError("Feature schemas differ between resolutions.")
    coarse_scaler = ajustar_scaler(coarse_train)
    fine_scaler = ajustar_scaler(fine_train)

    def scale(values: np.ndarray, scaler: Any) -> np.ndarray:
        scaled = aplicar_scaler(values, scaler)
        return restore_missing_channel_zeros(
            values, scaled, n_channels=n_channels,
            features_per_channel=coarse_features,
        )

    coarse_train_scaled = scale(coarse_train, coarse_scaler)
    coarse_target_scaled = scale(coarse_target, coarse_scaler)
    fine_train_scaled = scale(fine_train, fine_scaler)
    fine_target_scaled = scale(fine_target, fine_scaler)

    payload: dict[str, Any] = {
        "experiment": "v47_dual_resolution_exploratory_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "training_files": args.training_files,
        "target_file": args.target_file,
        "calibration_patients": calibration_patients,
        "reserved_test_files_not_loaded": test_files,
        "test_accessed": False,
        "exploratory_target_used": True,
        "promotion_allowed": False,
        "paired_train_sequences": int(len(coarse_train)),
        "paired_target_sequences": int(len(target_coarse_idx)),
        "seeds": args.seeds,
        "variants": args.variants,
        "runs": [],
    }
    intervals = coarse_cache.intervalos_por_arquivo[args.target_file]
    for seed in args.seeds:
        seed_result: dict[str, Any] = {"seed": seed}
        for variant in args.variants:
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            if variant == "v43_single_resolution":
                model = create_late_channel_fusion_bilstm(
                    sequence_length=8, n_channels=n_channels,
                    features_per_channel=coarse_features,
                )
                train_input: Any = coarse_train_scaled
                target_input: Any = coarse_target_scaled
            elif variant == "v47_dual_resolution":
                model = create_dual_resolution_channel_fusion_bilstm(
                    coarse_steps=8, fine_steps=16, n_channels=n_channels,
                    features_per_channel=coarse_features,
                )
                train_input = [coarse_train_scaled, fine_train_scaled]
                target_input = [coarse_target_scaled, fine_target_scaled]
            else:
                model = create_residual_dual_resolution_bilstm(
                    coarse_steps=8, fine_steps=16, n_channels=n_channels,
                    features_per_channel=coarse_features,
                )
                train_input = [coarse_train_scaled, fine_train_scaled]
                target_input = [coarse_target_scaled, fine_target_scaled]
            history = model.fit(
                train_input, y_train, sample_weight=weights,
                epochs=args.epochs, batch_size=32, verbose=0, shuffle=True,
            )
            raw_scores = model.predict(target_input, verbose=0)
            _, scores, metas = agregar_predicoes_temporais(
                raw_scores, y_target, temporal_target,
            )
            seed_result[variant] = {
                "epochs": len(history.history["loss"]),
                "final_train_loss": float(history.history["loss"][-1]),
                "raw_target_scores": summarize_event_scores(scores, metas, intervals),
            }
        payload["runs"].append(seed_result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
