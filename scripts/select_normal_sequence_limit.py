"""Seleciona o limite de sequencias normais usando somente calibracao."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.config import get_settings  # noqa: E402
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    ajustar_scaler,
    aplicar_scaler,
    criar_modelo_segmentacao,
    resolver_class_weight,
)
from scripts.validate_sequence_by_patient import (  # noqa: E402
    SequenceDatasetCache,
    agrupar_por_paciente,
    calibrar_no_treino,
    carregar_arquivos,
    ranking_calibracao_eventos,
    separar_treino_calibracao,
    validar_isolamento_fold,
)


def parse_int_list(value: str) -> list[int]:
    values = list(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Informe limites inteiros positivos separados por virgula.")
    return values


def serializar_ranking(value: tuple[float, ...]) -> list[float]:
    return [float(item) for item in value]


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reserved-test-patients", nargs="+", required=True)
    parser.add_argument("--fixed-calibration-patients", nargs="+", required=True)
    parser.add_argument("--normal-sequence-limits", type=parse_int_list, default=parse_int_list("32,48,64,96"))
    parser.add_argument("--max-seizure-sequences-per-file", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import tensorflow as tf

    arquivos = carregar_arquivos(args.dataset_dir, args.manifest)
    por_paciente = agrupar_por_paciente(arquivos)
    reserved = set(args.reserved_test_patients)
    representative = args.reserved_test_patients[0]
    train_files, calibration_files, calibration_patients = separar_treino_calibracao(
        por_paciente,
        representative,
        n_calibration_patients=len(args.fixed_calibration_patients),
        reserved_test_patients=reserved,
        fixed_calibration_patients=args.fixed_calibration_patients,
    )
    reserved_files = [arquivo for paciente in sorted(reserved) for arquivo in por_paciente.get(paciente, [])]
    validar_isolamento_fold(
        train_files=train_files,
        calibration_files=calibration_files,
        test_files=reserved_files,
        reserved_test_patients=reserved,
    )

    reference_path = args.dataset_dir / (settings.ai_sequence_channel_reference_edf or "chb01_01.edf")
    canais_referencia = list(extrair_metadados_edf(reference_path)["canais_eeg"])
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir,
        window_seconds=settings.ai_sequence_window_seconds,
        step_seconds=settings.ai_sequence_step_seconds,
        max_normal_windows=max(args.normal_sequence_limits),
        max_seizure_windows=args.max_seizure_sequences_per_file,
        sequence_length=settings.ai_sequence_length,
        sequence_stride=settings.ai_sequence_stride,
        sequence_target_mode="center",
        sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5,
        feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel",
        canais_referencia=canais_referencia,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    x_cal, _y_cal, meta_cal = cache.carregar_lista(
        arquivos=calibration_files,
        max_normal_windows=None,
        max_seizure_windows=None,
    )
    y_cal_temporal, meta_cal_temporal = construir_alvos_temporais(
        meta_cal,
        cache.intervalos_por_arquivo,
        sequence_length=settings.ai_sequence_length,
        window_seconds=settings.ai_sequence_window_seconds,
        step_seconds=settings.ai_sequence_step_seconds,
        min_ictal_overlap_ratio=0.5,
    )

    resultados: list[dict[str, Any]] = []
    for normal_limit in args.normal_sequence_limits:
        print(f"\n=== Calibracao: {normal_limit} sequencias normais/EDF ===", flush=True)
        x_train, _y_train, meta_train = cache.carregar_lista(
            arquivos=train_files,
            max_normal_windows=normal_limit,
            max_seizure_windows=args.max_seizure_sequences_per_file,
        )
        y_train_temporal, meta_train_temporal = construir_alvos_temporais(
            meta_train,
            cache.intervalos_por_arquivo,
            sequence_length=settings.ai_sequence_length,
            window_seconds=settings.ai_sequence_window_seconds,
            step_seconds=settings.ai_sequence_step_seconds,
            min_ictal_overlap_ratio=0.5,
        )
        y_train_eval, _dummy, _meta_eval = agregar_predicoes_temporais(
            np.zeros_like(y_train_temporal), y_train_temporal, meta_train_temporal
        )
        scaler = ajustar_scaler(x_train)
        x_train_scaled = aplicar_scaler(x_train, scaler)
        x_cal_scaled = aplicar_scaler(x_cal, scaler)
        class_weight = resolver_class_weight(y_train_eval, mode="balanced", positive_weight=3.0)
        sample_weight = pesos_temporais(
            y_train_temporal,
            meta_train_temporal,
            boundary_weight=1.5,
            class_weight=class_weight,
            balance_events=False,
        )

        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(args.random_state)
        model = criar_modelo_segmentacao(
            settings.ai_sequence_length,
            x_train.shape[2],
            loss_mode="binary_crossentropy",
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc",
                mode="max",
                patience=args.early_stopping_patience,
                min_delta=0.001,
                restore_best_weights=True,
            )
        ]
        history = model.fit(
            x_train_scaled,
            y_train_temporal,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
            sample_weight=sample_weight,
            validation_data=(x_cal_scaled, y_cal_temporal),
            callbacks=callbacks,
        )
        scores_cal_raw = np.asarray(model.predict(x_cal_scaled, verbose=0))
        y_cal_eval, scores_cal, meta_cal_eval = agregar_predicoes_temporais(
            scores_cal_raw, y_cal_temporal, meta_cal_temporal
        )
        calibracao = calibrar_no_treino(
            y_cal_eval,
            scores_cal,
            meta_cal_eval,
            thresholds=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6],
            durations=[6.0, 10.0, 18.0, 26.0, 30.0, 40.0, 60.0],
            max_gap_values=[0.0, 4.0, 6.0, 8.0, 10.0],
            hysteresis_ratios=[1.0],
            min_overlap_ratio=0.25,
            selection_mode="event_balanced",
        )
        best = calibracao["best"]
        ranking = ranking_calibracao_eventos(best)
        resultado = {
            "max_normal_sequences_per_file": normal_limit,
            "n_train_sequences": int(len(x_train)),
            "n_train_temporal_positive": int(np.sum(y_train_temporal)),
            "n_train_temporal_negative": int(y_train_temporal.size - np.sum(y_train_temporal)),
            "epochs_trained": int(len(history.history.get("loss", []))),
            "ranking": serializar_ranking(ranking),
            "calibration_best": best,
            "calibration_top_10": calibracao["top_10"],
        }
        resultados.append(resultado)
        print(
            f"limite={normal_limit}: event_f1={best['event_f1']:.3f}, "
            f"sens={best['event_sensitivity']:.3f}, FA/h={best['false_alarms_per_hour']:.3f}, "
            f"localizado_f1={best['localized_metrics']['f1']:.3f}",
            flush=True,
        )

    resultados.sort(key=lambda item: tuple(item["ranking"]), reverse=True)
    payload = {
        "experiment": "select_normal_sequence_limit_calibration_only",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "reserved_test_patients": sorted(reserved),
        "train_patients": sorted({item.split("_")[0] for item in train_files}),
        "calibration_patients": calibration_patients,
        "test_data_accessed": False,
        "random_state": args.random_state,
        "normal_sequence_candidates": args.normal_sequence_limits,
        "selected_max_normal_sequences_per_file": resultados[0]["max_normal_sequences_per_file"],
        "selection_rule": "event_f1, event_sensitivity, localized_f1, -FA/h, EDF F1",
        "results": resultados,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    print(f"Selecao salva em: {args.output}", flush=True)


if __name__ == "__main__":
    main()
