"""Avalia um ponto operacional congelado no reservado historico."""

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
from app.ai_engine.sequence_inference import carregar_recursos_sequenciais  # noqa: E402
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
)
from app.config import get_settings  # noqa: E402
from scripts.calibrate_sequence_cnn_lstm import (  # noqa: E402
    avaliar_combo,
    intervalos_crise_reais,
    listar_runs,
    melhor_overlap_com_crise,
)
from scripts.experiment_channel_mask_v32 import load_sequences  # noqa: E402
from scripts.experiment_expanded_train_v37 import restore_missing_channel_zeros  # noqa: E402
from scripts.train_sequence_cnn_lstm import aplicar_scaler  # noqa: E402
from scripts.validate_sequence_by_patient import (  # noqa: E402
    SequenceDatasetCache,
    agrupar_por_paciente,
    carregar_arquivos,
)


def diagnosticos_eventos(
    y_true: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    *,
    threshold: float,
    min_duration_seconds: float,
    max_gap_seconds: float,
) -> list[dict[str, Any]]:
    """Compara cada crise real com o run previsto de maior sobreposicao."""
    resultado: list[dict[str, Any]] = []
    for arquivo in sorted({str(meta["arquivo"]) for meta in metas}):
        indices = np.asarray(
            [idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo],
            dtype=int,
        )
        eventos = intervalos_crise_reais(y_true, metas, indices)
        runs = [
            run
            for run in listar_runs(
                scores,
                metas,
                indices,
                threshold,
                min_duration_seconds=min_duration_seconds,
                max_gap_seconds=max_gap_seconds,
            )
            if float(run["duration_seconds"]) >= min_duration_seconds
            and float(run["score_mean"]) >= threshold
        ]
        for numero, evento in enumerate(eventos, start=1):
            melhor = max(
                runs,
                key=lambda run: melhor_overlap_com_crise(run, [evento]),
                default=None,
            )
            overlap, overlap_ratio = (
                melhor_overlap_com_crise(melhor, [evento])
                if melhor is not None
                else (0.0, 0.0)
            )
            real_start = float(evento["start_seconds"])
            real_end = float(evento["end_seconds"])
            union = 0.0
            if melhor is not None:
                union = max(real_end, float(melhor["end_seconds"])) - min(
                    real_start, float(melhor["start_seconds"])
                )
            resultado.append(
                {
                    "arquivo": arquivo,
                    "evento": numero,
                    "real_start_seconds": real_start,
                    "real_end_seconds": real_end,
                    "real_duration_seconds": real_end - real_start,
                    "detectado": bool(overlap >= 1.0 and overlap_ratio >= 0.25),
                    "overlap_seconds": float(overlap),
                    "overlap_ratio": float(overlap_ratio),
                    "temporal_iou": float(overlap / union) if union > 0 else 0.0,
                    "predicted_start_seconds": (
                        float(melhor["start_seconds"]) if melhor is not None else None
                    ),
                    "predicted_end_seconds": (
                        float(melhor["end_seconds"]) if melhor is not None else None
                    ),
                    "predicted_duration_seconds": (
                        float(melhor["duration_seconds"]) if melhor is not None else None
                    ),
                    "onset_error_seconds": (
                        float(melhor["start_seconds"]) - real_start
                        if melhor is not None and overlap > 0
                        else None
                    ),
                    "offset_error_seconds": (
                        float(melhor["end_seconds"]) - real_end
                        if melhor is not None and overlap > 0
                        else None
                    ),
                    "latency_seconds": (
                        max(0.0, float(melhor["start_seconds"]) - real_start)
                        if melhor is not None and overlap > 0
                        else None
                    ),
                    "score_max": float(melhor["score_max"]) if melhor is not None else None,
                    "score_mean": float(melhor["score_mean"]) if melhor is not None else None,
                }
            )
    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "dataset_amostra/manifests/v38_expanded_train.txt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "modelos/v43_final_seed42_reserved_historical_eval.json",
    )
    parser.add_argument(
        "--patients", nargs="+", default=["chb09", "chb15", "chb18"],
    )
    parser.add_argument(
        "--model-path", type=Path,
        help="Modelo experimental; deriva scaler/metadata/calibration e exige --output separado.",
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.model_path is not None:
        if args.output.resolve() == (ROOT / "modelos/v43_final_seed42_reserved_historical_eval.json").resolve():
            parser.error("An experimental model requires a separate --output report.")
        model_path = args.model_path.resolve()
        experimental_paths = {
            "ai_sequence_model_path": model_path,
            "ai_sequence_scaler_path": model_path.with_name(f"{model_path.stem}_scaler.pkl"),
            "ai_sequence_metadata_path": model_path.with_name(f"{model_path.stem}_metadata.json"),
            "ai_sequence_calibration_path": model_path.with_name(f"{model_path.stem}_calibration.json"),
        }
        if not all(path.is_file() for path in experimental_paths.values()):
            parser.error("The experimental model, scaler, metadata and calibration must all exist.")
        settings = settings.model_copy(update=experimental_paths)
    recursos = carregar_recursos_sequenciais(settings)
    if recursos is None:
        raise SystemExit("Artefatos sequenciais ativos nao puderam ser carregados.")
    calibration = recursos.calibration or {}
    operating_point = calibration.get("best_aligned") or calibration.get("best")
    if not isinstance(operating_point, dict):
        raise SystemExit("Ponto operacional congelado ausente.")
    threshold = float(operating_point["threshold"])
    min_duration = float(operating_point["min_duration_seconds"])
    max_gap = float(operating_point.get("max_gap_seconds", 0.0))

    dataset_dir = ROOT / "dataset_amostra"
    by_patient = agrupar_por_paciente(carregar_arquivos(dataset_dir, args.manifest))
    reserved_patients = list(args.patients)
    test_files = [name for patient in reserved_patients for name in by_patient[patient]]
    metadata = recursos.metadata
    reference = list(
        metadata.get("canais_referencia")
        or extrair_metadados_edf(dataset_dir / "chb01_01.edf")["canais_eeg"]
    )
    cache = SequenceDatasetCache(
        dataset_dir=dataset_dir,
        window_seconds=float(metadata.get("window_seconds", 4)),
        step_seconds=float(metadata.get("step_seconds", 2)),
        max_normal_windows=48,
        max_seizure_windows=32,
        sequence_length=int(metadata.get("sequence_length", 8)),
        sequence_stride=int(metadata.get("sequence_stride", 2)),
        sequence_target_mode=str(metadata.get("sequence_target_mode", "center")),
        sampling_level="sequence",
        min_window_ictal_overlap_ratio=float(
            metadata.get("min_window_ictal_overlap_ratio", 0.5)
        ),
        feature_normalization=str(metadata.get("feature_normalization", "per_edf_robust")),
        feature_mode=str(metadata.get("feature_mode", "time_frequency_per_channel")),
        canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    x_test, _, meta_test = load_sequences(cache, test_files, sampled=False)
    y_test, temporal_test = construir_alvos_temporais(
        meta_test,
        cache.intervalos_por_arquivo,
        sequence_length=int(metadata.get("sequence_length", 8)),
        window_seconds=float(metadata.get("window_seconds", 4)),
        step_seconds=float(metadata.get("step_seconds", 2)),
        min_ictal_overlap_ratio=float(metadata.get("min_window_ictal_overlap_ratio", 0.5)),
    )
    scaled = aplicar_scaler(x_test, recursos.scaler)
    scaled = restore_missing_channel_zeros(
        x_test,
        scaled,
        n_channels=int(metadata["n_channels"]),
        features_per_channel=int(metadata["features_per_channel"]),
    )
    raw_scores = np.asarray(recursos.model.predict(scaled, batch_size=128, verbose=0))
    yy, scores, metas = agregar_predicoes_temporais(raw_scores, y_test, temporal_test)
    fixed = avaliar_combo(
        yy,
        scores,
        metas,
        threshold=threshold,
        min_duration_seconds=min_duration,
        max_gap_seconds=max_gap,
        max_suspicious_coverage=float(settings.ai_sequence_max_suspicious_coverage),
        min_overlap_seconds=1.0,
        min_overlap_ratio=0.25,
    )
    eventos = diagnosticos_eventos(
        yy,
        scores,
        metas,
        threshold=threshold,
        min_duration_seconds=min_duration,
        max_gap_seconds=max_gap,
    )
    detectados = [item for item in eventos if item["detectado"]]
    fixed["event_temporal_summary"] = {
        "mean_iou_detected": float(np.mean([item["temporal_iou"] for item in detectados]))
        if detectados else 0.0,
        "mean_absolute_onset_error_seconds": float(
            np.mean([abs(float(item["onset_error_seconds"])) for item in detectados])
        ) if detectados else None,
        "mean_absolute_offset_error_seconds": float(
            np.mean([abs(float(item["offset_error_seconds"])) for item in detectados])
        ) if detectados else None,
        "mean_latency_seconds": float(
            np.mean([float(item["latency_seconds"]) for item in detectados])
        ) if detectados else None,
    }
    payload = {
        "experiment": (
            f"{args.model_path.stem}_reserved_historical_frozen"
            if args.model_path is not None else "v43_final_seed42_reserved_historical_frozen"
        ),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "methodological_status": (
            "historical_reserved_development_test_already_consulted"
            if set(reserved_patients) == {"chb09", "chb15", "chb18"}
            else "artifact_verification_only"
        ),
        "parameters_were_tuned_on_this_evaluation": False,
        "model_path": str(Path(recursos.model_path).resolve()),
        "manifest": str(args.manifest.resolve()),
        "patients": reserved_patients,
        "files": test_files,
        "operating_point": {
            "threshold": threshold,
            "min_duration_seconds": min_duration,
            "max_gap_seconds": max_gap,
        },
        "fixed_evaluation": fixed,
        "events": eventos,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    resumo = {
        "edf": fixed["metrics"],
        "localized": fixed["localized_metrics"],
        "events": {
            "detected": fixed["detected_seizure_events"],
            "total": fixed["total_seizure_events"],
            "precision": fixed["event_precision"],
            "recall": fixed["event_sensitivity"],
            "f1": fixed["event_f1"],
            "false_alarms": fixed["false_alarm_events"],
            "false_alarms_per_hour": fixed["false_alarms_per_hour"],
        },
        "temporal": fixed["event_temporal_summary"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
