"""Compare the current representation with per-channel temporal deltas on calibration."""

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
from app.ai_engine.temporal_representation import append_temporal_deltas  # noqa: E402
from app.config import get_settings  # noqa: E402
from scripts.calibrate_sequence_cnn_lstm import avaliar_combo  # noqa: E402
from scripts.experiment_multiscale_v31 import (  # noqa: E402
    _compact,
    _event_diagnostics,
    _froc,
)
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


def _load_sequences(
    cache: SequenceDatasetCache,
    files: list[str],
    *,
    sampled: bool,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []
    normal_limit = 48 if sampled else None
    seizure_limit = 32 if sampled else None
    for file in files:
        x, y, meta = cache.carregar_arquivo(
            file,
            max_normal_windows=normal_limit,
            max_seizure_windows=seizure_limit,
        )
        x_parts.append(x)
        y_parts.append(y)
        metas.extend(meta)
        cache._cache.pop((file, normal_limit, seizure_limit), None)
    return np.concatenate(x_parts), np.concatenate(y_parts), metas


def _summarize(runs: list[dict[str, Any]], variant: str) -> dict[str, float]:
    selected = [run["calibration_best"] for run in runs if run["variant"] == variant]
    fields = (
        "edf_precision",
        "edf_recall",
        "edf_f1",
        "localized_f1",
        "event_recall",
        "event_precision",
        "event_f1",
        "false_alarms_per_hour",
        "detected_events",
        "false_alarms",
    )
    return {field: float(np.mean([item[field] for item in selected])) for field in fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "dataset_amostra/manifests/v23_expanded_reserved_test.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "modelos/v33_temporal_delta_calibration.json",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--resume-calibration", action="store_true")
    parser.add_argument("--evaluate-reserved", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf

    files = carregar_arquivos(args.dataset_dir, args.manifest)
    by_patient = agrupar_por_paciente(files)
    reserved = {"chb09", "chb15", "chb18"}
    fixed_calibration = ["chb16", "chb19", "chb20", "chb24"]
    train_files, calibration_files, calibration_patients = separar_treino_calibracao(
        by_patient,
        "chb09",
        n_calibration_patients=4,
        reserved_test_patients=reserved,
        fixed_calibration_patients=fixed_calibration,
    )
    test_files = [file for patient in sorted(reserved) for file in by_patient[patient]]
    validar_isolamento_fold(
        train_files=train_files,
        calibration_files=calibration_files,
        test_files=test_files,
        reserved_test_patients=reserved,
    )
    if not all(any(file.startswith(prefix) for file in train_files) for prefix in ("chb01_", "chb21_")):
        raise RuntimeError("chb01 and chb21 must remain together in training.")

    reference = list(extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"])
    settings = get_settings()
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir,
        window_seconds=4,
        step_seconds=2,
        max_normal_windows=48,
        max_seizure_windows=32,
        sequence_length=8,
        sequence_stride=2,
        sequence_target_mode="center",
        sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5,
        feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel",
        canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    print("Loading frozen training/calibration split", flush=True)
    x_train, y_train_sequence, train_meta = _load_sequences(cache, train_files, sampled=True)
    x_calibration, y_calibration_sequence, calibration_meta = _load_sequences(
        cache, calibration_files, sampled=False,
    )
    y_train, train_temporal_meta = construir_alvos_temporais(
        train_meta,
        cache.intervalos_por_arquivo,
        sequence_length=8,
        window_seconds=4,
        step_seconds=2,
        min_ictal_overlap_ratio=0.5,
    )
    y_calibration, calibration_temporal_meta = construir_alvos_temporais(
        calibration_meta,
        cache.intervalos_por_arquivo,
        sequence_length=8,
        window_seconds=4,
        step_seconds=2,
        min_ictal_overlap_ratio=0.5,
    )
    unique_train_y, _, _ = agregar_predicoes_temporais(
        np.zeros_like(y_train), y_train, train_temporal_meta,
    )
    weights = pesos_temporais(
        y_train,
        train_temporal_meta,
        boundary_weight=1.5,
        class_weight=resolver_class_weight(
            unique_train_y, mode="balanced", positive_weight=3.0,
        ),
    )
    scaler = ajustar_scaler(x_train)
    train_baseline = aplicar_scaler(x_train, scaler)
    calibration_baseline = aplicar_scaler(x_calibration, scaler)
    features_per_channel = len(TIME_FREQUENCY_FEATURE_NAMES)
    train_delta = append_temporal_deltas(
        train_baseline, features_per_channel=features_per_channel,
    )
    calibration_delta = append_temporal_deltas(
        calibration_baseline, features_per_channel=features_per_channel,
    )

    payload: dict[str, Any] = {
        "experiment": "v33_per_channel_temporal_delta_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "train_patients": sorted(set(by_patient) - reserved - set(fixed_calibration)),
        "calibration_patients": calibration_patients,
        "reserved_test_patients": sorted(reserved),
        "train_files": train_files,
        "calibration_files": calibration_files,
        "reserved_test_files_not_loaded": test_files,
        "test_accessed": False,
        "seeds": args.seeds,
        "epochs_limit": args.epochs,
        "sampling_level": "sequence",
        "feature_mode": "time_frequency_per_channel",
        "feature_normalization": "per_edf_robust",
        "sequence_length": 8,
        "sequence_stride": 2,
        "train_sequences": len(x_train),
        "train_positive_sequences": int(y_train_sequence.sum()),
        "calibration_sequences": len(x_calibration),
        "calibration_positive_sequences": int(y_calibration_sequence.sum()),
        "input_features": {
            "baseline": int(train_baseline.shape[-1]),
            "temporal_delta": int(train_delta.shape[-1]),
        },
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_calibration:
        if not args.output.exists():
            raise FileNotFoundError(f"Calibration artifact is missing: {args.output}")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload.get("experiment") != "v33_per_channel_temporal_delta_calibration_only"
            or payload.get("manifest") != str(args.manifest.resolve())
            or payload.get("train_files") != train_files
            or payload.get("calibration_files") != calibration_files
            or payload.get("seeds") != args.seeds
            or payload.get("epochs_limit") != args.epochs
        ):
            raise RuntimeError("Frozen calibration artifact does not match this configuration.")
    else:
        for seed in args.seeds:
            for variant, train_input, calibration_input in (
                ("baseline", train_baseline, calibration_baseline),
                ("temporal_delta", train_delta, calibration_delta),
            ):
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = criar_modelo_segmentacao(
                    8, train_input.shape[-1], loss_mode="binary_crossentropy",
                )
                history = model.fit(
                    train_input,
                    y_train,
                    epochs=args.epochs,
                    batch_size=32,
                    verbose=0,
                    sample_weight=weights,
                    validation_data=(calibration_input, y_calibration),
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
                            monitor="val_pr_auc",
                            mode="max",
                            patience=4,
                            min_delta=0.001,
                            restore_best_weights=True,
                        )
                    ],
                )
                raw_scores = np.asarray(
                    model.predict(calibration_input, batch_size=128, verbose=0)
                )
                y_eval, scores, meta_eval = agregar_predicoes_temporais(
                    raw_scores,
                    y_calibration,
                    calibration_temporal_meta,
                )
                best, frontier, feasible = _froc(y_eval, scores, meta_eval)
                result = {
                    "seed": seed,
                    "variant": variant,
                    "parameters": int(model.count_params()),
                    "epochs_trained": len(history.history["loss"]),
                    "best_validation_pr_auc": float(max(history.history["val_pr_auc"])),
                    "calibration_constraint_satisfied": feasible,
                    "calibration_best": _compact(best),
                    "calibration_froc": frontier,
                }
                payload["runs"].append(result)
                payload["summary"] = {
                    name: _summarize(payload["runs"], name)
                    for name in ("baseline", "temporal_delta")
                    if any(run["variant"] == name for run in payload["runs"])
                }
                args.output.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
                )
                print(f"{seed} {variant}: {result['calibration_best']}", flush=True)
                del model

    if args.evaluate_reserved:
        reserved_results: list[dict[str, Any]] = []
        for fold_index, patient in enumerate(sorted(reserved), start=1):
            seed = 42 + fold_index
            print(f"Frozen temporal-delta evaluation: {patient} (seed {seed})", flush=True)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            model = criar_modelo_segmentacao(
                8, train_delta.shape[-1], loss_mode="binary_crossentropy",
            )
            history = model.fit(
                train_delta,
                y_train,
                epochs=args.epochs,
                batch_size=32,
                verbose=0,
                sample_weight=weights,
                validation_data=(calibration_delta, y_calibration),
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_pr_auc",
                        mode="max",
                        patience=4,
                        min_delta=0.001,
                        restore_best_weights=True,
                    )
                ],
            )
            calibration_raw = np.asarray(
                model.predict(calibration_delta, batch_size=128, verbose=0)
            )
            calibration_y, calibration_scores, calibration_eval_meta = agregar_predicoes_temporais(
                calibration_raw,
                y_calibration,
                calibration_temporal_meta,
            )
            operating_point, _, _ = _froc(
                calibration_y, calibration_scores, calibration_eval_meta,
            )

            patient_data: list[dict[str, Any]] = []
            test_x_parts: list[np.ndarray] = []
            test_meta: list[dict[str, Any]] = []
            for file in by_patient[patient]:
                file_x, _file_y, file_meta = cache.carregar_arquivo(
                    file, max_normal_windows=None, max_seizure_windows=None,
                )
                patient_data.append({"file": file, "x8": file_x, "meta": file_meta})
                test_x_parts.append(file_x)
                test_meta.extend(file_meta)
            test_x = append_temporal_deltas(
                aplicar_scaler(np.concatenate(test_x_parts), scaler),
                features_per_channel=features_per_channel,
            )
            test_y_temporal, test_temporal_meta = construir_alvos_temporais(
                test_meta,
                cache.intervalos_por_arquivo,
                sequence_length=8,
                window_seconds=4,
                step_seconds=2,
                min_ictal_overlap_ratio=0.5,
            )
            test_raw = np.asarray(model.predict(test_x, batch_size=128, verbose=0))
            test_y, test_scores, test_eval_meta = agregar_predicoes_temporais(
                test_raw, test_y_temporal, test_temporal_meta,
            )
            evaluation = avaliar_combo(
                test_y,
                test_scores,
                test_eval_meta,
                threshold=float(operating_point["threshold"]),
                min_duration_seconds=float(operating_point["min_duration_seconds"]),
                max_gap_seconds=float(operating_point["max_gap_seconds"]),
                hysteresis_ratio=1.0,
                min_overlap_ratio=0.25,
            )
            diagnostics = _event_diagnostics(
                test_y,
                test_scores,
                test_eval_meta,
                evaluation,
                patient_data,
                reference,
            )
            reserved_results.append(
                {
                    "patient": patient,
                    "seed": seed,
                    "epochs_trained": len(history.history["loss"]),
                    "calibration_point": _compact(operating_point),
                    "test_metrics": _compact(evaluation),
                    "normal_hours": float(evaluation["normal_hours"]),
                    "event_diagnostics": diagnostics,
                    "files": evaluation["files"],
                }
            )
            payload["reserved_development_test"] = reserved_results
            payload["test_accessed"] = True
            payload["test_used_for_selection"] = False
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            del model

        metrics = [item["test_metrics"] for item in reserved_results]
        total_false_alarms = sum(int(item["false_alarms"]) for item in metrics)
        total_normal_hours = sum(float(item["normal_hours"]) for item in reserved_results)
        payload["reserved_development_summary"] = {
            "edf_precision_macro": float(np.mean([item["edf_precision"] for item in metrics])),
            "edf_recall_macro": float(np.mean([item["edf_recall"] for item in metrics])),
            "edf_f1_macro": float(np.mean([item["edf_f1"] for item in metrics])),
            "localized_f1_macro": float(np.mean([item["localized_f1"] for item in metrics])),
            "event_recall_macro": float(np.mean([item["event_recall"] for item in metrics])),
            "event_precision_macro": float(np.mean([item["event_precision"] for item in metrics])),
            "event_f1_macro": float(np.mean([item["event_f1"] for item in metrics])),
            "detected_events": int(sum(item["detected_events"] for item in metrics)),
            "total_events": int(sum(item["total_events"] for item in metrics)),
            "false_alarms": total_false_alarms,
            "normal_hours": total_normal_hours,
            "false_alarms_per_hour": (
                float(total_false_alarms / total_normal_hours) if total_normal_hours else 0.0
            ),
            "classification": "development_only_not_external_validation",
        }
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
