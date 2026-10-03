"""Calibrated architecture ablation with a fixed interpatient development split."""

from __future__ import annotations

import argparse
import gc
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
from app.ai_engine.multiscale_data import centered_context, event_aware_indices  # noqa: E402
from app.ai_engine.multiscale_segmentation import create_multiscale_model  # noqa: E402
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.config import get_settings  # noqa: E402
from scripts.calibrate_sequence_cnn_lstm import (  # noqa: E402
    avaliar_combo,
    intervalos_crise_reais,
    listar_runs,
    melhor_overlap_com_crise,
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

THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)
DURATIONS = (6.0, 10.0, 18.0, 26.0, 30.0, 40.0, 60.0)
GAPS = (0.0, 4.0, 6.0, 8.0, 10.0)
VARIANTS = (
    ("A_baseline", "baseline", "bce", False),
    ("B_tcn", "tcn", "bce", False),
    ("C_multiscale", "multiscale", "bce", False),
    ("D_channel_attention", "attention", "bce", False),
    ("E_boundary", "attention", "boundary", False),
    ("E_tversky", "attention", "tversky", False),
    ("E_continuity", "attention", "continuity", False),
    ("F_event_sampler", "attention", "boundary", True),
)


def _meta_key(meta: dict[str, Any]) -> float:
    return float(meta["context_start_seconds"])


def _targets(metas: list[dict[str, Any]], intervals: dict[str, Any]) -> tuple[np.ndarray, list[Any]]:
    return construir_alvos_temporais(
        metas, intervals, sequence_length=8, window_seconds=4.0,
        step_seconds=2.0, min_ictal_overlap_ratio=0.5,
    )


def _load_training(cache: SequenceDatasetCache, files: list[str]) -> dict[str, Any]:
    parts: dict[str, list[Any]] = {key: [] for key in ("x8", "x24", "meta", "event_x8", "event_x24", "event_meta")}
    audit: list[dict[str, Any]] = []
    for file in files:
        x8_all, y_all, meta_all = cache.carregar_arquivo(
            file, max_normal_windows=None, max_seizure_windows=None,
        )
        x8_sample, y_sample, meta_sample = cache.carregar_arquivo(
            file, max_normal_windows=48, max_seizure_windows=32,
        )
        lookup = {_meta_key(meta): idx for idx, meta in enumerate(meta_all)}
        sampled_indices = np.asarray([lookup[_meta_key(meta)] for meta in meta_sample], dtype=int)
        event_positive = event_aware_indices(
            y_all, meta_all, cache.intervalos_por_arquivo.get(file, []), max_positive=32,
        )
        event_indices = np.asarray(sorted(set(sampled_indices[y_sample == 0]) | set(event_positive)), dtype=int)
        context = centered_context(
            x8_all, meta_all, target_indices=np.unique(np.concatenate([sampled_indices, event_indices])),
        )
        all_selected = np.unique(np.concatenate([sampled_indices, event_indices]))
        context_lookup = {int(idx): row for row, idx in enumerate(all_selected)}
        parts["x8"].append(x8_sample)
        parts["x24"].append(context[[context_lookup[int(idx)] for idx in sampled_indices]])
        parts["meta"].extend(meta_sample)
        parts["event_x8"].append(x8_all[event_indices])
        parts["event_x24"].append(context[[context_lookup[int(idx)] for idx in event_indices]])
        parts["event_meta"].extend([meta_all[int(idx)] for idx in event_indices])
        audit.append({
            "file": file, "all_positive_sequences": int(np.sum(y_all == 1)),
            "baseline_positive_sequences": int(np.sum(y_sample == 1)),
            "event_sampler_positive_sequences": int(np.sum(y_all[event_indices] == 1)),
        })
        cache._cache.pop((file, None, None), None)
        cache._cache.pop((file, 48, 32), None)
        del x8_all, x8_sample, context
        gc.collect()
    return {
        key: np.concatenate(parts[key]) if key in {"x8", "x24", "event_x8", "event_x24"} else parts[key]
        for key in parts
    } | {"sampling_audit": audit}


def _load_evaluation(cache: SequenceDatasetCache, files: list[str]) -> list[dict[str, Any]]:
    result = []
    for file in files:
        x8, _y, metas = cache.carregar_arquivo(
            file, max_normal_windows=None, max_seizure_windows=None,
        )
        result.append({"file": file, "x8": x8, "x24": centered_context(x8, metas), "meta": metas})
    return result


def _predict_eval(model: Any, scaler: Any, files: list[dict[str, Any]], *, long_context: bool,
                  intervals: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    all_y: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_meta: list[dict[str, Any]] = []
    for item in files:
        x = aplicar_scaler(item["x24"] if long_context else item["x8"], scaler)
        y, temporal_meta = _targets(item["meta"], intervals)
        scores = np.asarray(model.predict(x, batch_size=128, verbose=0))
        yy, ss, mm = agregar_predicoes_temporais(scores, y, temporal_meta)
        all_y.append(yy)
        all_scores.append(ss)
        all_meta.extend(mm)
    return np.concatenate(all_y), np.concatenate(all_scores), all_meta


def _froc(y: np.ndarray, scores: np.ndarray, meta: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    results = [
        avaliar_combo(
            y, scores, meta, threshold=threshold, min_duration_seconds=duration,
            max_gap_seconds=gap, hysteresis_ratio=1.0, min_overlap_ratio=0.25,
        )
        for threshold in THRESHOLDS for duration in DURATIONS for gap in GAPS
    ]
    eligible = [item for item in results if item["false_alarms_per_hour"] <= 1.0]
    pool = eligible or results
    best = max(pool, key=lambda item: (
        item["event_sensitivity"], item["localized_metrics"]["f1"],
        item["event_precision"], -item["false_alarms_per_hour"],
    ) if eligible else (
        -item["false_alarms_per_hour"], item["event_sensitivity"],
        item["localized_metrics"]["f1"], item["event_precision"],
    ))
    frontier = []
    for cap in (0.5, 0.75, 1.0, 1.25, 1.5):
        feasible = [item for item in results if item["false_alarms_per_hour"] <= cap]
        if feasible:
            point = max(feasible, key=lambda item: (
                item["event_sensitivity"], item["localized_metrics"]["f1"],
                item["event_precision"], -item["false_alarms_per_hour"],
            ))
            frontier.append({"cap": cap, **_compact(point)})
        else:
            frontier.append({"cap": cap, "constraint_satisfied": False})
    return best, frontier, bool(eligible)


def _compact(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "threshold": float(item["threshold"]),
        "min_duration_seconds": float(item["min_duration_seconds"]),
        "max_gap_seconds": float(item["max_gap_seconds"]),
        "edf_precision": float(item["metrics"]["precision"]),
        "edf_recall": float(item["metrics"]["recall"]),
        "edf_f1": float(item["metrics"]["f1"]),
        "localized_f1": float(item["localized_metrics"]["f1"]),
        "event_recall": float(item["event_sensitivity"]),
        "event_precision": float(item["event_precision"]),
        "event_f1": float(item["event_f1"]),
        "false_alarms_per_hour": float(item["false_alarms_per_hour"]),
        "detected_events": int(item["detected_seizure_events"]),
        "total_events": int(item["total_seizure_events"]),
        "false_alarms": int(item["false_alarm_events"]),
        "false_positives_edf": int(item["false_positives"]),
        "false_negatives_edf": int(item["false_negatives"]),
    }


def _event_diagnostics(
    y: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    evaluation: dict[str, Any],
    patient_data: list[dict[str, Any]],
    channels: list[str],
) -> list[dict[str, Any]]:
    by_file = {item["file"]: item for item in patient_data}
    output: list[dict[str, Any]] = []
    for file_result in evaluation["files"]:
        file = file_result["arquivo"]
        indices = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == file])
        segments = intervalos_crise_reais(y, metas, indices)
        if not segments:
            continue
        runs = listar_runs(
            scores, metas, indices, evaluation["threshold"],
            min_duration_seconds=evaluation["min_duration_seconds"],
            max_gap_seconds=evaluation["max_gap_seconds"],
        )
        runs = [
            run for run in runs
            if run["duration_seconds"] >= evaluation["min_duration_seconds"]
            and run["score_mean"] >= evaluation["threshold"]
        ]
        full = by_file[file]
        times = np.asarray([float(meta["start_seconds"]) for meta in full["meta"]])
        powers = full["x8"][:, 4, :].reshape(len(times), len(channels), len(TIME_FREQUENCY_FEATURE_NAMES))[:, :, :5]
        for event_idx, event in enumerate(segments, start=1):
            start, end = float(event["start_seconds"]), float(event["end_seconds"])
            inside = (times >= start) & (times < end)
            outside = (times < start - 60.0) | (times > end + 60.0)
            contrast = np.median(powers[inside], axis=0) - np.median(powers[outside], axis=0)
            channel_strength = np.max(np.abs(contrast), axis=1)
            top_channel_idx = int(np.argmax(channel_strength))
            best_run = max(runs, key=lambda run: melhor_overlap_com_crise(run, [event])[0], default=None)
            overlap = melhor_overlap_com_crise(best_run, [event])[0] if best_run else 0.0
            union = (end - start) + float(best_run["duration_seconds"]) - overlap if best_run else 0.0
            primary = file_result["trecho_suspeito"]
            primary_overlap = melhor_overlap_com_crise(primary, [event])[0]
            ictal_mask = np.asarray([
                meta["arquivo"] == file and start <= float(meta["start_seconds"]) < end
                for meta in metas
            ])
            score_ictal = scores[ictal_mask]
            output.append({
                "file": file, "event": event_idx, "real_start_seconds": start,
                "real_end_seconds": end, "real_duration_seconds": end - start,
                "max_ictal_score": float(np.max(score_ictal)) if score_ictal.size else None,
                "mean_ictal_score": float(np.mean(score_ictal)) if score_ictal.size else None,
                "detected": bool(overlap >= 1.0 and overlap / (end - start) >= 0.25),
                "overlap_seconds": float(overlap),
                "temporal_iou": float(overlap / union) if union else 0.0,
                "onset_error_seconds": float(best_run["start_seconds"] - start) if best_run else None,
                "offset_error_seconds": float(best_run["end_seconds"] - end) if best_run else None,
                "latency_seconds": float(max(0.0, best_run["start_seconds"] - start)) if best_run else None,
                "predicted_duration_seconds": float(best_run["duration_seconds"]) if best_run else None,
                "primary_correct": bool(primary_overlap >= 1.0 and primary_overlap / (end - start) >= 0.25),
                "strongest_feature_contrast_channel": channels[top_channel_idx],
                "feature_contrast_by_band": {
                    band: float(value)
                    for band, value in zip(("delta", "theta", "alpha", "beta", "gamma"), contrast[top_channel_idx])
                },
                "feature_contrast_is_model_attribution": False,
            })
    return output


def _train_variant(spec: tuple[str, str, str, bool], data: dict[str, Any],
                   cal_files: list[dict[str, Any]], cache: SequenceDatasetCache,
                   *, seed: int, epochs: int) -> tuple[Any, Any, dict[str, Any]]:
    import tensorflow as tf

    name, architecture, loss_mode, event_sampler = spec
    meta_key = "event_meta" if event_sampler else "meta"
    short_key = "event_x8" if event_sampler else "x8"
    long_key = "event_x24" if event_sampler else "x24"
    meta = data[meta_key]
    y, temporal_meta = _targets(meta, cache.intervalos_por_arquivo)
    y_unique, _empty, _m = agregar_predicoes_temporais(np.zeros_like(y), y, temporal_meta)
    scaler = ajustar_scaler(data[short_key])
    long_context = architecture in {"multiscale", "attention"}
    x = aplicar_scaler(data[long_key] if long_context else data[short_key], scaler)
    class_weight = resolver_class_weight(y_unique, mode="balanced", positive_weight=3.0)
    weights = pesos_temporais(y, temporal_meta, boundary_weight=1.5, class_weight=class_weight)
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    if architecture == "baseline":
        model = criar_modelo_segmentacao(8, x.shape[-1], loss_mode="binary_crossentropy")
    else:
        model = create_multiscale_model(
            input_steps=24 if long_context else 8, output_steps=8,
            n_channels=x.shape[-1] // len(TIME_FREQUENCY_FEATURE_NAMES),
            features_per_channel=len(TIME_FREQUENCY_FEATURE_NAMES),
            variant=architecture, loss_mode=loss_mode,
        )
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_pr_auc", mode="max", patience=4, min_delta=0.001,
        restore_best_weights=True,
    )]
    cal_x = np.concatenate([
        aplicar_scaler(item["x24"] if long_context else item["x8"], scaler)
        for item in cal_files
    ])
    cal_y = np.concatenate([_targets(item["meta"], cache.intervalos_por_arquivo)[0] for item in cal_files])
    history = model.fit(
        x, y, epochs=epochs, batch_size=32, verbose=0, sample_weight=weights,
        validation_data=(cal_x, cal_y), callbacks=callbacks,
    )
    del cal_x, cal_y
    y_cal, scores_cal, meta_cal = _predict_eval(
        model, scaler, cal_files, long_context=long_context,
        intervals=cache.intervalos_por_arquivo,
    )
    best, frontier, feasible = _froc(y_cal, scores_cal, meta_cal)
    result = {
        "variant": name, "architecture": architecture, "loss": loss_mode,
        "event_sampler": event_sampler, "seed": seed,
        "parameters": int(model.count_params()),
        "epochs_trained": len(history.history["loss"]),
        "train_sequences": len(x), "train_positive_steps": int(y.sum()),
        "calibration_constraint_satisfied": feasible,
        "calibration_best": _compact(best), "calibration_froc": frontier,
    }
    print(f"{name}: {result['calibration_best']}", flush=True)
    return model, scaler, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "dataset_amostra/manifests/v23_expanded_reserved_test.txt")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "dataset_amostra")
    parser.add_argument("--output", type=Path, default=ROOT / "modelos/v31_multiscale_ablation.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--resume-calibration", action="store_true")
    parser.add_argument("--resume-ablation", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
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
        train_files=train_files, calibration_files=cal_files, test_files=test_files,
        reserved_test_patients=reserved,
    )
    reference = list(extrair_metadados_edf(args.dataset_dir / "chb01_01.edf")["canais_eeg"])
    cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir, window_seconds=4, step_seconds=2,
        max_normal_windows=48, max_seizure_windows=32, sequence_length=8,
        sequence_stride=2, sequence_target_mode="center", sampling_level="sequence",
        min_window_ictal_overlap_ratio=0.5, feature_normalization="per_edf_robust",
        feature_mode="time_frequency_per_channel", canais_referencia=reference,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )
    print("Loading training and calibration data", flush=True)
    training = _load_training(cache, train_files)
    cal_data = _load_evaluation(cache, cal_files)
    payload: dict[str, Any] = {
        "experiment": "v31_multiscale_architecture_ablation",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(args.manifest.resolve()),
        "train_patients": sorted(set(by_patient) - reserved - set(fixed_cal)),
        "calibration_patients": cal_patients,
        "reserved_test_patients": sorted(reserved),
        "train_files": train_files, "calibration_files": cal_files,
        "test_files": test_files,
        "seed": args.seed, "epochs_limit": args.epochs,
        "first_stage_only": True, "candidate_confirmer": False,
        "normal_sequences_per_edf": 48, "positive_sequences_per_edf": 32,
        "sampling_audit": training["sampling_audit"],
        "ablation": [], "test_accessed_during_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_calibration or args.resume_ablation:
        if not args.output.exists():
            raise FileNotFoundError(f"Calibration artifact is missing: {args.output}")
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload.get("manifest") != str(args.manifest.resolve())
            or payload.get("train_files") != train_files
            or payload.get("calibration_files") != cal_files
            or payload.get("test_files") != test_files
            or payload.get("seed") != args.seed
            or payload.get("epochs_limit") != args.epochs
        ):
            raise RuntimeError("Frozen calibration artifact does not match this evaluation split/configuration.")
        if args.resume_calibration and payload.get("selected_variant") not in {item[0] for item in VARIANTS}:
            raise RuntimeError("The calibration artifact has no frozen selected variant.")
    if not args.resume_calibration:
        for spec in VARIANTS:
            if spec[0] in {item["variant"] for item in payload["ablation"]}:
                continue
            _model, _scaler, result = _train_variant(
                spec, training, cal_data, cache, seed=args.seed, epochs=args.epochs,
            )
            payload["ablation"].append(result)
            args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            del _model, _scaler
        eligible = [item for item in payload["ablation"] if item["calibration_constraint_satisfied"]]
        candidates = eligible or payload["ablation"]
        selected = max(candidates, key=lambda item: (
            item["calibration_best"]["event_recall"],
            item["calibration_best"]["localized_f1"],
            item["calibration_best"]["event_precision"],
            -item["calibration_best"]["false_alarms_per_hour"],
        ))
        payload["selected_variant"] = selected["variant"]
        payload["selection_rule"] = "FA/h<=1, event recall, localized F1, event precision, -FA/h"
        payload["selection_constraint_satisfied"] = bool(eligible)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = next(item for item in payload["ablation"] if item["variant"] == payload["selected_variant"])
    print(f"Selected on calibration: {selected['variant']}", flush=True)
    if args.calibration_only:
        return
    spec = next(item for item in VARIANTS if item[0] == selected["variant"])
    test_results = []
    for fold_idx, patient in enumerate(sorted(reserved), start=1):
        print(f"Frozen evaluation: {patient}", flush=True)
        model, scaler, refit = _train_variant(
            spec, training, cal_data, cache, seed=args.seed + fold_idx, epochs=args.epochs,
        )
        point = refit["calibration_best"]
        patient_files = _load_evaluation(cache, by_patient[patient])
        y, scores, metas = _predict_eval(
            model, scaler, patient_files,
            long_context=spec[1] in {"multiscale", "attention"},
            intervals=cache.intervalos_por_arquivo,
        )
        evaluation = avaliar_combo(
            y, scores, metas, threshold=point["threshold"],
            min_duration_seconds=point["min_duration_seconds"],
            max_gap_seconds=point["max_gap_seconds"],
            min_overlap_ratio=0.25,
        )
        diagnostics = _event_diagnostics(y, scores, metas, evaluation, patient_files, reference)
        if sum(item["detected"] for item in diagnostics) != evaluation["detected_seizure_events"]:
            raise RuntimeError("Per-event diagnostics disagree with the frozen event metric.")
        test_results.append({
            "patient": patient, "seed": args.seed + fold_idx,
            "calibration_point": point, "test_metrics": _compact(evaluation),
            "event_diagnostics": diagnostics,
            "files": evaluation["files"],
        })
        payload["test"] = test_results
        payload["test_accessed_during_selection"] = False
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
