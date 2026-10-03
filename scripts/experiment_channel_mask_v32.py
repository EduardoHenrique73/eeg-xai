"""Audit the existing features and compare a channel-presence input on calibration only."""

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
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.config import get_settings  # noqa: E402
from scripts.experiment_multiscale_v31 import _compact, _froc  # noqa: E402
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


def channel_presence(
    dataset_dir: Path, files: list[str], reference: list[str],
) -> dict[str, np.ndarray]:
    result = {}
    for file in files:
        available = set(extrair_metadados_edf(dataset_dir / file)["canais_eeg"])
        result[file] = np.asarray([channel in available for channel in reference], dtype=np.float32)
    return result


def append_channel_presence(
    x_scaled: np.ndarray, metas: list[dict[str, Any]], masks: dict[str, np.ndarray],
) -> np.ndarray:
    if len(x_scaled) != len(metas):
        raise ValueError("Features and sequence metadata are misaligned.")
    presence = np.asarray([masks[str(meta["arquivo"])] for meta in metas], dtype=np.float32)
    return np.concatenate(
        (x_scaled, np.broadcast_to(presence[:, None, :], (len(metas), x_scaled.shape[1], presence.shape[1]))),
        axis=-1,
    )


def saturation_audit(x: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if len(x) != len(labels) or x.shape[-1] % len(TIME_FREQUENCY_FEATURE_NAMES):
        raise ValueError("Expected labeled per-channel time-frequency sequences.")
    reshaped = x[:, x.shape[1] // 2, :].reshape(len(x), -1, len(TIME_FREQUENCY_FEATURE_NAMES))
    output = {}
    for name, selector in (("normal", labels == 0), ("ictal_center", labels == 1)):
        group = np.abs(reshaped[selector]) >= 9.999
        output[name] = {
            "sequences": int(np.sum(selector)),
            "all_features_fraction": float(np.mean(group)) if group.size else None,
            "absolute_band_power_fraction": float(np.mean(group[:, :, :5])) if group.size else None,
            "relative_band_power_fraction": float(np.mean(group[:, :, 5:10])) if group.size else None,
            "other_features_fraction": float(np.mean(group[:, :, 10:])) if group.size else None,
        }
    return output


def load_sequences(
    cache: SequenceDatasetCache, files: list[str], *, sampled: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    limits = {"max_normal_windows": 48, "max_seizure_windows": 32} if sampled else {
        "max_normal_windows": None, "max_seizure_windows": None,
    }
    for file in files:
        x, y, meta = cache.carregar_arquivo(file, **limits)
        x_parts.append(x)
        y_parts.append(y)
        metas.extend(meta)
        cache._cache.pop((file, limits["max_normal_windows"], limits["max_seizure_windows"]), None)
    return np.concatenate(x_parts), np.concatenate(y_parts), metas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset_amostra/manifests/v23_expanded_reserved_test.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "modelos/v32_channel_mask_calibration.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()

    import tensorflow as tf

    files = carregar_arquivos(args.dataset_dir, args.manifest)
    by_patient = agrupar_por_paciente(files)
    reserved = {"chb09", "chb15", "chb18"}
    fixed_cal = ["chb16", "chb19", "chb20", "chb24"]
    train_files, cal_files, cal_patients = separar_treino_calibracao(
        by_patient, "chb09", n_calibration_patients=4,
        reserved_test_patients=reserved, fixed_calibration_patients=fixed_cal,
    )
    test_files = [file for patient in sorted(reserved) for file in by_patient[patient]]
    validar_isolamento_fold(
        train_files=train_files, calibration_files=cal_files,
        test_files=test_files, reserved_test_patients=reserved,
    )
    if not all(any(file.startswith(prefix) for file in train_files) for prefix in ("chb01_", "chb21_")):
        raise RuntimeError("chb01 and chb21, the same subject, must remain in training together.")

    reference = list(extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"])
    masks = channel_presence(args.dataset_dir, train_files + cal_files, reference)
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
    x_train, y_train_seq, meta_train = load_sequences(cache, train_files, sampled=True)
    x_cal, y_cal_seq, meta_cal = load_sequences(cache, cal_files, sampled=False)
    y_train, temporal_train = construir_alvos_temporais(
        meta_train, cache.intervalos_por_arquivo, sequence_length=8,
        window_seconds=4, step_seconds=2, min_ictal_overlap_ratio=0.5,
    )
    y_cal, temporal_cal = construir_alvos_temporais(
        meta_cal, cache.intervalos_por_arquivo, sequence_length=8,
        window_seconds=4, step_seconds=2, min_ictal_overlap_ratio=0.5,
    )
    y_train_unique, _, _ = agregar_predicoes_temporais(np.zeros_like(y_train), y_train, temporal_train)
    weights = pesos_temporais(
        y_train, temporal_train, boundary_weight=1.5,
        class_weight=resolver_class_weight(y_train_unique, mode="balanced", positive_weight=3.0),
    )
    scaler = ajustar_scaler(x_train)
    train_scaled = aplicar_scaler(x_train, scaler)
    cal_scaled = aplicar_scaler(x_cal, scaler)
    train_masked = append_channel_presence(train_scaled, meta_train, masks)
    cal_masked = append_channel_presence(cal_scaled, meta_cal, masks)
    missing_scaled = {
        channel: float(abs(scaler.mean_[index * 15] / scaler.scale_[index * 15]))
        for index, channel in enumerate(reference)
        if any(mask[index] == 0 for mask in masks.values())
    }
    payload: dict[str, Any] = {
        "experiment": "v32_explicit_channel_presence_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "train_files": train_files, "calibration_files": cal_files,
        "reserved_test_files_not_loaded": test_files,
        "calibration_patients": cal_patients, "seeds": args.seeds,
        "epochs_limit": args.epochs, "same_subject_cases": ["chb01", "chb21"],
        "reference_channels": reference,
        "missing_channels_by_file": {
            file: [channel for channel, observed in zip(reference, masks[file]) if not observed]
            for file in train_files + cal_files if not np.all(masks[file])
        },
        "scaled_missing_zero_magnitude_by_channel_example_feature": missing_scaled,
        "train_saturation": saturation_audit(x_train, y_train_seq),
        "calibration_saturation": saturation_audit(x_cal, y_cal_seq),
        "train_sequences": len(x_train), "train_positive_sequences": int(y_train_seq.sum()),
        "calibration_sequences": len(x_cal), "calibration_positive_sequences": int(y_cal_seq.sum()),
        "runs": [], "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for variant, train_input, cal_input in (
            ("baseline", train_scaled, cal_scaled),
            ("presence_mask", train_masked, cal_masked),
        ):
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = criar_modelo_segmentacao(8, train_input.shape[-1], loss_mode="binary_crossentropy")
            history = model.fit(
                train_input, y_train, epochs=args.epochs, batch_size=32,
                verbose=0, sample_weight=weights, validation_data=(cal_input, y_cal),
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc", mode="max", patience=4,
                    min_delta=0.001, restore_best_weights=True,
                )],
            )
            raw_scores = np.asarray(model.predict(cal_input, batch_size=128, verbose=0))
            yy, scores, meta_eval = agregar_predicoes_temporais(raw_scores, y_cal, temporal_cal)
            best, frontier, feasible = _froc(yy, scores, meta_eval)
            result = {
                "seed": seed, "variant": variant, "parameters": model.count_params(),
                "epochs_trained": len(history.history["loss"]),
                "calibration_constraint_satisfied": feasible,
                "calibration_best": _compact(best), "calibration_froc": frontier,
            }
            payload["runs"].append(result)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"{seed} {variant}: {result['calibration_best']}", flush=True)
            del model


if __name__ == "__main__":
    main()
