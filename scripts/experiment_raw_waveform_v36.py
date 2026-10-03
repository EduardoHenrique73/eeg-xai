"""Evaluate a short per-channel raw waveform branch on calibration only."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai_engine.feature_extractor import (  # noqa: E402
    TIME_FREQUENCY_FEATURE_NAMES,
    extrair_metadados_edf,
    selecionar_canais_eeg_validos,
)
from app.ai_engine.multiscale_segmentation import create_raw_waveform_fusion_model  # noqa: E402
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.ai_engine.temporal_representation import (  # noqa: E402
    raw_waveform_windows,
    sequence_windows_from_timeline,
)
from app.config import get_settings  # noqa: E402
from scripts.experiment_multiscale_v31 import _compact, _froc  # noqa: E402
from scripts.experiment_temporal_delta_v33 import _summarize  # noqa: E402
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

RAW_POINTS = 128


def _raw_timeline(
    dataset_dir: Path,
    file: str,
    reference: list[str],
    cache_dir: Path,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{Path(file).stem}__raw64hz_2s_v1.npy"
    if cache_file.exists():
        return np.load(cache_file, mmap_mode="r")
    raw = mne.io.read_raw_edf(dataset_dir / file, preload=True, verbose=False)
    available = selecionar_canais_eeg_validos(raw)
    signals = np.zeros((len(reference), raw.n_times), dtype=np.float64)
    for index, channel in enumerate(reference):
        if channel in available:
            signals[index] = np.asarray(raw.get_data(picks=[channel])[0], dtype=np.float64)
    timeline = raw_waveform_windows(signals, sfreq=float(raw.info["sfreq"]))
    temporary = cache_file.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, timeline, allow_pickle=False)
    temporary.replace(cache_file)
    return np.load(cache_file, mmap_mode="r")


def _load_files(
    cache: SequenceDatasetCache,
    files: list[str],
    reference: list[str],
    raw_cache_dir: Path,
    *,
    sampled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    spectral_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    normal_limit = 48 if sampled else None
    seizure_limit = 32 if sampled else None
    for file in files:
        spectral, y, meta = cache.carregar_arquivo(
            file, max_normal_windows=normal_limit, max_seizure_windows=seizure_limit,
        )
        timeline = _raw_timeline(cache.dataset_dir, file, reference, raw_cache_dir)
        raw_sequences = sequence_windows_from_timeline(
            timeline, meta, sequence_length=8, step_seconds=2.0, dtype=np.float16,
        )
        spectral_parts.append(spectral)
        raw_parts.append(raw_sequences)
        y_parts.append(y)
        metas.extend(meta)
        cache._cache.pop((file, normal_limit, seizure_limit), None)
        print(f"{file}: raw branch attached to {len(y)} sequences", flush=True)
    return (
        np.concatenate(spectral_parts),
        np.concatenate(raw_parts),
        np.concatenate(y_parts),
        metas,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "dataset_amostra/manifests/v23_expanded_reserved_test.txt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "modelos/v36_raw_waveform_calibration.json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    import tensorflow as tf

    files = carregar_arquivos(args.dataset_dir, args.manifest)
    by_patient = agrupar_por_paciente(files)
    reserved = {"chb09", "chb15", "chb18"}
    fixed_calibration = ["chb16", "chb19", "chb20", "chb24"]
    train_files, calibration_files, calibration_patients = separar_treino_calibracao(
        by_patient, "chb09", n_calibration_patients=4,
        reserved_test_patients=reserved, fixed_calibration_patients=fixed_calibration,
    )
    test_files = [file for patient in sorted(reserved) for file in by_patient[patient]]
    validar_isolamento_fold(
        train_files=train_files, calibration_files=calibration_files,
        test_files=test_files, reserved_test_patients=reserved,
    )
    reference = list(extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"])
    settings = get_settings()
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir, window_seconds=4, step_seconds=2,
        max_normal_windows=48, max_seizure_windows=32, sequence_length=8,
        sequence_stride=2, sequence_target_mode="center", sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5, feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel", canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    raw_cache_dir = args.dataset_dir / "raw_waveform_cache_v1"
    print("Loading training and sampled calibration raw waveforms", flush=True)
    x_train, raw_train, y_train_sequence, train_meta = _load_files(
        cache, train_files, reference, raw_cache_dir, sampled=True,
    )
    x_cal_fit, raw_cal_fit, _y_cal_sequence, cal_fit_meta = _load_files(
        cache, calibration_files, reference, raw_cache_dir, sampled=True,
    )
    y_train, train_temporal_meta = construir_alvos_temporais(
        train_meta, cache.intervalos_por_arquivo, sequence_length=8,
        window_seconds=4, step_seconds=2, min_ictal_overlap_ratio=0.5,
    )
    y_cal_fit, _cal_fit_temporal_meta = construir_alvos_temporais(
        cal_fit_meta, cache.intervalos_por_arquivo, sequence_length=8,
        window_seconds=4, step_seconds=2, min_ictal_overlap_ratio=0.5,
    )
    unique_y, _, _ = agregar_predicoes_temporais(
        np.zeros_like(y_train), y_train, train_temporal_meta,
    )
    weights = pesos_temporais(
        y_train, train_temporal_meta, boundary_weight=1.5,
        class_weight=resolver_class_weight(unique_y, mode="balanced", positive_weight=3.0),
    )
    scaler = ajustar_scaler(x_train)
    train_scaled = aplicar_scaler(x_train, scaler)
    cal_fit_scaled = aplicar_scaler(x_cal_fit, scaler)
    payload: dict[str, Any] = {
        "experiment": "v36_short_raw_waveform_branch_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "train_files": train_files,
        "calibration_files": calibration_files,
        "calibration_patients": calibration_patients,
        "reserved_test_files_not_loaded": test_files,
        "test_accessed": False,
        "raw_context_seconds": 2.0,
        "raw_sampling_rate_hz": 64,
        "raw_points": RAW_POINTS,
        "seeds": args.seeds,
        "epochs_limit": args.epochs,
        "train_sequences": len(x_train),
        "sampled_calibration_sequences_for_early_stopping": len(x_cal_fit),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = create_raw_waveform_fusion_model(
            sequence_length=8, n_channels=len(reference),
            spectral_features=len(TIME_FREQUENCY_FEATURE_NAMES), raw_points=RAW_POINTS,
        )
        history = model.fit(
            [train_scaled, raw_train], y_train, epochs=args.epochs, batch_size=16,
            verbose=0, sample_weight=weights,
            validation_data=([cal_fit_scaled, raw_cal_fit], y_cal_fit),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=4,
                min_delta=0.001, restore_best_weights=True,
            )],
        )
        all_y: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_meta: list[dict[str, Any]] = []
        for file in calibration_files:
            spectral, raw_values, _labels, metas = _load_files(
                cache, [file], reference, raw_cache_dir, sampled=False,
            )
            targets, temporal_meta = construir_alvos_temporais(
                metas, cache.intervalos_por_arquivo, sequence_length=8,
                window_seconds=4, step_seconds=2, min_ictal_overlap_ratio=0.5,
            )
            scores_raw = np.asarray(model.predict(
                [aplicar_scaler(spectral, scaler), raw_values], batch_size=32, verbose=0,
            ))
            yy, scores, evaluated_meta = agregar_predicoes_temporais(
                scores_raw, targets, temporal_meta,
            )
            all_y.append(yy)
            all_scores.append(scores)
            all_meta.extend(evaluated_meta)
        best, frontier, feasible = _froc(
            np.concatenate(all_y), np.concatenate(all_scores), all_meta,
        )
        result = {
            "seed": seed,
            "variant": "raw_waveform_branch",
            "parameters": int(model.count_params()),
            "epochs_trained": len(history.history["loss"]),
            "best_validation_pr_auc": float(max(history.history["val_pr_auc"])),
            "calibration_constraint_satisfied": feasible,
            "calibration_best": _compact(best),
            "calibration_froc": frontier,
        }
        payload["runs"].append(result)
        payload["summary"] = _summarize(payload["runs"], "raw_waveform_branch")
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{seed} raw_waveform_branch: {result['calibration_best']}", flush=True)
        del model
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
