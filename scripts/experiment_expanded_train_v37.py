"""Compare the fixed CNN-BiLSTM after adding independent training events."""

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
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.config import get_settings  # noqa: E402
from scripts.calibrate_sequence_cnn_lstm import (  # noqa: E402
    intervalos_crise_reais,
    listar_runs,
    melhor_overlap_com_crise,
)
from scripts.experiment_channel_mask_v32 import load_sequences  # noqa: E402
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


def summarize(runs: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "edf_precision", "edf_recall", "edf_f1", "localized_f1",
        "event_recall", "event_precision", "event_f1",
        "false_alarms_per_hour", "detected_events", "false_alarms",
    )
    return {
        key: float(np.mean([run["calibration_best"][key] for run in runs]))
        for key in keys
    }


def checkpoint_rank(
    metrics: dict[str, Any], *, constraint_satisfied: bool,
) -> tuple[float, ...]:
    """Rank epochs by event utility without trading away the FA/h constraint."""
    if constraint_satisfied:
        return (
            1.0,
            float(metrics["event_recall"]),
            float(metrics["localized_f1"]),
            float(metrics["event_precision"]),
            -float(metrics["false_alarms_per_hour"]),
            float(metrics["edf_f1"]),
        )
    return (
        0.0,
        -float(metrics["false_alarms_per_hour"]),
        float(metrics["event_recall"]),
        float(metrics["localized_f1"]),
        float(metrics["event_precision"]),
        float(metrics["edf_f1"]),
    )


def select_checkpoint_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    meta: list[dict[str, Any]],
    *,
    max_false_alarms_per_hour: float,
) -> tuple[dict[str, Any], bool]:
    """Select the operating point used to compare one training epoch."""
    best, frontier, _ = _froc(y_true, scores, meta)
    point = next(
        (
            item for item in frontier
            if abs(float(item["cap"]) - max_false_alarms_per_hour) < 1e-9
        ),
        None,
    )
    if point is not None and point.get("constraint_satisfied", True):
        return {
            key: value for key, value in point.items()
            if key not in {"cap", "constraint_satisfied"}
        }, True
    return _compact(best), False


def build_event_diagnostics(
    y_true: np.ndarray,
    scores: np.ndarray,
    meta: list[dict[str, Any]],
    operating_point: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe every calibration seizure at one fixed operating point."""
    diagnostics: list[dict[str, Any]] = []
    threshold = float(operating_point["threshold"])
    min_duration = float(operating_point["min_duration_seconds"])
    max_gap = float(operating_point["max_gap_seconds"])
    for arquivo in sorted({str(item["arquivo"]) for item in meta}):
        indices = np.asarray(
            [idx for idx, item in enumerate(meta) if item["arquivo"] == arquivo],
            dtype=int,
        )
        events = intervalos_crise_reais(y_true, meta, indices)
        if not events:
            continue
        runs = [
            run for run in listar_runs(
                scores,
                meta,
                indices,
                threshold,
                min_duration_seconds=min_duration,
                max_gap_seconds=max_gap,
            )
            if float(run["duration_seconds"]) >= min_duration
            and float(run["score_mean"]) >= threshold
        ]
        for event_index, event in enumerate(events, start=1):
            event_start = float(event["start_seconds"])
            event_end = float(event["end_seconds"])
            event_indices = np.asarray([
                idx for idx in indices
                if int(y_true[idx]) == 1
                and float(meta[int(idx)]["start_seconds"]) >= event_start - 1e-9
                and float(meta[int(idx)]["end_seconds"]) <= event_end + 1e-9
            ], dtype=int)
            best_run = max(
                runs,
                key=lambda run: melhor_overlap_com_crise(run, [event]),
                default=None,
            )
            overlap, overlap_ratio = (
                melhor_overlap_com_crise(best_run, [event])
                if best_run is not None else (0.0, 0.0)
            )
            diagnostics.append({
                "file": arquivo,
                "event": event_index,
                "real_start_seconds": event_start,
                "real_end_seconds": event_end,
                "real_duration_seconds": event_end - event_start,
                "max_ictal_score": (
                    float(np.max(scores[event_indices])) if event_indices.size else None
                ),
                "mean_ictal_score": (
                    float(np.mean(scores[event_indices])) if event_indices.size else None
                ),
                "detected": bool(overlap >= 1.0 and overlap_ratio >= 0.25),
                "overlap_seconds": float(overlap),
                "overlap_ratio": float(overlap_ratio),
                "onset_error_seconds": (
                    float(best_run["start_seconds"] - event_start)
                    if best_run is not None and overlap > 0 else None
                ),
                "predicted_duration_seconds": (
                    float(best_run["duration_seconds"])
                    if best_run is not None and overlap > 0 else None
                ),
            })
    return diagnostics


def create_event_checkpoint_callback(
    *,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    calibration_temporal_meta: list[dict[str, Any]],
    max_false_alarms_per_hour: float,
    patience: int,
    min_epochs: int,
) -> Any:
    """Create a callback that restores the best event-level calibration epoch."""
    import tensorflow as tf

    class EventCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best_rank: tuple[float, ...] | None = None
            self.best_weights: list[np.ndarray] | None = None
            self.best_epoch: int | None = None
            self.best_metrics: dict[str, Any] | None = None
            self.wait = 0
            self.trajectory: list[dict[str, Any]] = []

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            raw_scores = np.asarray(
                self.model.predict(calibration_x, batch_size=128, verbose=0)
            )
            yy, scores, meta_eval = agregar_predicoes_temporais(
                raw_scores, calibration_y, calibration_temporal_meta,
            )
            metrics, feasible = select_checkpoint_metrics(
                yy,
                scores,
                meta_eval,
                max_false_alarms_per_hour=max_false_alarms_per_hour,
            )
            rank = checkpoint_rank(metrics, constraint_satisfied=feasible)
            record = {
                "epoch": int(epoch + 1),
                "constraint_satisfied": bool(feasible),
                **metrics,
            }
            self.trajectory.append(record)
            if self.best_rank is None or rank > self.best_rank:
                self.best_rank = rank
                self.best_weights = self.model.get_weights()
                self.best_epoch = int(epoch + 1)
                self.best_metrics = record
                self.wait = 0
            else:
                self.wait += 1
            if epoch + 1 >= min_epochs and self.wait > patience:
                self.model.stop_training = True

        def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
            if self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    return EventCheckpoint()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "dataset_amostra",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "dataset_amostra/manifests/v37_expanded_train.txt",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "modelos/v37_expanded_train_calibration.json",
    )
    parser.add_argument(
        "--experiment", default="v37_expanded_training_events_calibration_only",
    )
    parser.add_argument(
        "--added-files", nargs="+",
        default=["chb02_19.edf", "chb17a_04.edf"],
    )
    parser.add_argument("--added-events", type=int, default=2)
    parser.add_argument("--event-balanced-sampling", action="store_true")
    parser.add_argument(
        "--checkpoint-monitor", choices=("val_pr_auc", "event_recall"),
        default="val_pr_auc",
    )
    parser.add_argument("--checkpoint-fa-cap", type=float, default=0.5)
    parser.add_argument("--checkpoint-patience", type=int, default=5)
    parser.add_argument("--checkpoint-min-epochs", type=int, default=6)
    parser.add_argument("--short-event-max-duration-seconds", type=float)
    parser.add_argument("--short-event-weight", type=float, default=1.0)
    parser.add_argument(
        "--event-audit-caps", type=float, nargs="*", default=[0.5, 0.75],
    )
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
    added = set(args.added_files)
    if not added.issubset(train_files):
        raise RuntimeError("The expanded EDFs must be isolated in the training split.")

    reference = list(
        extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"]
    )
    settings = get_settings()
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir, window_seconds=4, step_seconds=2,
        max_normal_windows=48, max_seizure_windows=32, sequence_length=8,
        sequence_stride=2, sequence_target_mode="center", sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5,
        feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel",
        canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
        event_balanced_sampling=args.event_balanced_sampling,
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
    unique_y, _, _ = agregar_predicoes_temporais(
        np.zeros_like(y_train), y_train, temporal_train,
    )
    weights = pesos_temporais(
        y_train, temporal_train, boundary_weight=1.5,
        class_weight=resolver_class_weight(
            unique_y, mode="balanced", positive_weight=3.0,
        ),
        short_event_max_duration_seconds=args.short_event_max_duration_seconds,
        short_event_weight=args.short_event_weight,
    )
    scaler = ajustar_scaler(x_train)
    train_scaled = aplicar_scaler(x_train, scaler)
    cal_scaled = aplicar_scaler(x_cal, scaler)
    payload: dict[str, Any] = {
        "experiment": args.experiment,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "added_training_files": sorted(added),
        "added_training_events": args.added_events,
        "train_files": train_files,
        "calibration_files": cal_files,
        "calibration_patients": cal_patients,
        "reserved_test_files_not_loaded": test_files,
        "test_accessed": False,
        "event_balanced_sampling": args.event_balanced_sampling,
        "checkpoint_monitor": args.checkpoint_monitor,
        "checkpoint_false_alarms_per_hour_cap": args.checkpoint_fa_cap,
        "checkpoint_patience": args.checkpoint_patience,
        "checkpoint_min_epochs": args.checkpoint_min_epochs,
        "short_event_max_duration_seconds": args.short_event_max_duration_seconds,
        "short_event_weight": args.short_event_weight,
        "event_audit_caps": args.event_audit_caps,
        "seeds": args.seeds,
        "epochs_limit": args.epochs,
        "train_sequences": int(len(x_train)),
        "train_positive_sequences": int(y_train_seq.sum()),
        "calibration_sequences": int(len(x_cal)),
        "calibration_positive_sequences": int(y_cal_seq.sum()),
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        model = criar_modelo_segmentacao(
            8, train_scaled.shape[-1], loss_mode="binary_crossentropy",
        )
        event_checkpoint = None
        if args.checkpoint_monitor == "event_recall":
            event_checkpoint = create_event_checkpoint_callback(
                calibration_x=cal_scaled,
                calibration_y=y_cal,
                calibration_temporal_meta=temporal_cal,
                max_false_alarms_per_hour=args.checkpoint_fa_cap,
                patience=args.checkpoint_patience,
                min_epochs=args.checkpoint_min_epochs,
            )
            callbacks = [event_checkpoint]
        else:
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max", patience=4,
                min_delta=0.001, restore_best_weights=True,
            )]
        history = model.fit(
            train_scaled, y_train, epochs=args.epochs, batch_size=32,
            verbose=0, sample_weight=weights,
            validation_data=(cal_scaled, y_cal),
            callbacks=callbacks,
        )
        raw_scores = np.asarray(model.predict(cal_scaled, batch_size=128, verbose=0))
        yy, scores, meta_eval = agregar_predicoes_temporais(
            raw_scores, y_cal, temporal_cal,
        )
        best, frontier, feasible = _froc(yy, scores, meta_eval)
        event_diagnostics = {}
        for cap in args.event_audit_caps:
            point = next(
                (
                    item for item in frontier
                    if abs(float(item["cap"]) - cap) < 1e-9
                    and item.get("constraint_satisfied", True)
                ),
                None,
            )
            if point is not None:
                event_diagnostics[str(cap)] = build_event_diagnostics(
                    yy, scores, meta_eval, point,
                )
        run = {
            "seed": seed,
            "parameters": int(model.count_params()),
            "epochs_trained": len(history.history["loss"]),
            "best_validation_pr_auc": float(max(history.history["val_pr_auc"])),
            "checkpoint_monitor": args.checkpoint_monitor,
            "checkpoint_best_epoch": (
                event_checkpoint.best_epoch if event_checkpoint is not None else None
            ),
            "checkpoint_best_metrics": (
                event_checkpoint.best_metrics if event_checkpoint is not None else None
            ),
            "checkpoint_trajectory": (
                event_checkpoint.trajectory if event_checkpoint is not None else []
            ),
            "calibration_constraint_satisfied": feasible,
            "calibration_best": _compact(best),
            "calibration_froc": frontier,
            "event_diagnostics_by_fa_cap": event_diagnostics,
        }
        payload["runs"].append(run)
        payload["summary"] = summarize(payload["runs"])
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"{seed} expanded_train: {run['calibration_best']}", flush=True)
        del model

    print(f"Saved: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
