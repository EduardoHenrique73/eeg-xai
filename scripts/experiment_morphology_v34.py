"""Evaluate scale-invariant per-channel morphology on the frozen calibration split."""

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
    MORPHOLOGY_FEATURE_NAMES,
    TIME_FREQUENCY_FEATURE_NAMES,
    TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES,
    extrair_metadados_edf,
    selecionar_canais_eeg_validos,
)
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.ai_engine.multiscale_segmentation import create_morphology_fusion_model  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.ai_engine.temporal_representation import (  # noqa: E402
    append_per_channel_features,
    morphology_windows,
    sequence_windows_from_timeline,
)
from scripts.experiment_multiscale_v31 import _compact, _froc  # noqa: E402
from scripts.experiment_temporal_delta_v33 import _summarize  # noqa: E402
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    ajustar_scaler,
    aplicar_scaler,
    criar_modelo_segmentacao,
    resolver_class_weight,
)
from scripts.validate_sequence_by_patient import (  # noqa: E402
    SequenceDatasetCache,
    agrupar_por_paciente,
    carregar_arquivos,
    separar_treino_calibracao,
    validar_isolamento_fold,
)


def _morphology_for_file(
    dataset_dir: Path,
    file: str,
    metas: list[dict[str, Any]],
    reference: list[str],
) -> np.ndarray:
    raw = mne.io.read_raw_edf(dataset_dir / file, preload=True, verbose=False)
    available = selecionar_canais_eeg_validos(raw)
    signal_by_channel = {
        channel: np.asarray(raw.get_data(picks=[channel])[0], dtype=np.float64)
        for channel in available
    }
    n_samples = raw.n_times
    signals = np.zeros((len(reference), n_samples), dtype=np.float64)
    for index, channel in enumerate(reference):
        if channel in signal_by_channel:
            signals[index] = signal_by_channel[channel]
    sfreq = float(raw.info["sfreq"])
    timeline = morphology_windows(
        signals,
        window_samples=int(round(4.0 * sfreq)),
        step_samples=int(round(2.0 * sfreq)),
    )
    return sequence_windows_from_timeline(
        timeline, metas, sequence_length=8, step_seconds=2.0,
    )


def _load_hybrid_sequences(
    cache: SequenceDatasetCache,
    files: list[str],
    reference: list[str],
    *,
    sampled: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    normal_limit = 48 if sampled else None
    seizure_limit = 32 if sampled else None
    for file in files:
        spectral, y, meta = cache.carregar_arquivo(
            file, max_normal_windows=normal_limit, max_seizure_windows=seizure_limit,
        )
        morphology = _morphology_for_file(cache.dataset_dir, file, meta, reference)
        x_parts.append(append_per_channel_features(
            spectral,
            morphology,
            base_features_per_channel=len(TIME_FREQUENCY_FEATURE_NAMES),
            extra_features_per_channel=len(MORPHOLOGY_FEATURE_NAMES),
        ))
        y_parts.append(y)
        metas.extend(meta)
        cache._cache.pop((file, normal_limit, seizure_limit), None)
        print(f"{file}: morphology attached to {len(y)} sequences", flush=True)
    return np.concatenate(x_parts), np.concatenate(y_parts), metas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "dataset_amostra/manifests/v23_expanded_reserved_test.txt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "modelos/v34_morphology_calibration.json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--model", choices=["early_fusion", "residual_branch"], default="early_fusion")
    args = parser.parse_args()

    import tensorflow as tf

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
    reference = list(extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"])
    settings = get_settings()
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir, window_seconds=4, step_seconds=2,
        max_normal_windows=48, max_seizure_windows=32, sequence_length=8,
        sequence_stride=2, sequence_target_mode="center", sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5, feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel",
        canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    print("Loading morphology training/calibration data", flush=True)
    x_train, y_train_sequence, train_meta = _load_hybrid_sequences(
        cache, train_files, reference, sampled=True,
    )
    x_calibration, y_calibration_sequence, calibration_meta = _load_hybrid_sequences(
        cache, calibration_files, reference, sampled=False,
    )
    y_train, train_temporal_meta = construir_alvos_temporais(
        train_meta, cache.intervalos_por_arquivo, sequence_length=8,
        window_seconds=4, step_seconds=2, min_ictal_overlap_ratio=0.5,
    )
    y_calibration, calibration_temporal_meta = construir_alvos_temporais(
        calibration_meta, cache.intervalos_por_arquivo, sequence_length=8,
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
    calibration_scaled = aplicar_scaler(x_calibration, scaler)
    payload: dict[str, Any] = {
        "experiment": f"v34_time_frequency_plus_morphology_{args.model}_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "feature_mode": "time_frequency_per_channel_plus_morphology_v1",
        "features_per_channel": TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES,
        "train_files": train_files,
        "calibration_files": calibration_files,
        "calibration_patients": calibration_patients,
        "reserved_test_files_not_loaded": test_files,
        "test_accessed": False,
        "seeds": args.seeds,
        "epochs_limit": args.epochs,
        "train_sequences": len(x_train),
        "train_positive_sequences": int(y_train_sequence.sum()),
        "calibration_sequences": len(x_calibration),
        "calibration_positive_sequences": int(y_calibration_sequence.sum()),
        "input_features": int(x_train.shape[-1]),
        "baseline_reference": "modelos/v32_channel_mask_calibration.json::baseline",
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        if args.model == "residual_branch":
            model = create_morphology_fusion_model(
                sequence_length=8,
                n_channels=len(reference),
                spectral_features=len(TIME_FREQUENCY_FEATURE_NAMES),
                morphology_features=len(MORPHOLOGY_FEATURE_NAMES),
            )
        else:
            model = criar_modelo_segmentacao(8, train_scaled.shape[-1])
        history = model.fit(
            train_scaled, y_train, epochs=args.epochs, batch_size=32, verbose=0,
            sample_weight=weights, validation_data=(calibration_scaled, y_calibration),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=4,
                min_delta=0.001, restore_best_weights=True,
            )],
        )
        raw_scores = np.asarray(model.predict(calibration_scaled, batch_size=128, verbose=0))
        y_eval, scores, meta_eval = agregar_predicoes_temporais(
            raw_scores, y_calibration, calibration_temporal_meta,
        )
        best, frontier, feasible = _froc(y_eval, scores, meta_eval)
        result = {
            "seed": seed,
            "variant": args.model,
            "parameters": int(model.count_params()),
            "epochs_trained": len(history.history["loss"]),
            "best_validation_pr_auc": float(max(history.history["val_pr_auc"])),
            "calibration_constraint_satisfied": feasible,
            "calibration_best": _compact(best),
            "calibration_froc": frontier,
        }
        payload["runs"].append(result)
        payload["summary"] = _summarize(payload["runs"], args.model)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{seed} {args.model}: {result['calibration_best']}", flush=True)
        del model
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
