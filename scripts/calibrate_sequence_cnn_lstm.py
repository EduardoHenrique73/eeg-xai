"""Calibra threshold e duracao minima do CNN-LSTM sequencial.

Este script nao retreina o modelo. Ele carrega o modelo sequencial salvo,
gera scores nos EDFs de validacao e testa combinacoes de threshold/duracao
para escolher uma regra de decisao por arquivo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_sequence_cnn_lstm import aplicar_scaler, carregar_dataset  # noqa: E402


def carregar_metadata(model_path: Path, metadata_path: Path | None) -> dict[str, Any]:
    path = metadata_path or model_path.with_name(f"{model_path.stem}_metadata.json")
    if not path.exists():
        raise SystemExit(f"Metadata nao encontrado: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def metricas(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def melhor_run(
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    indices: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    runs = listar_runs(scores, metas, indices, threshold)
    if runs:
        return {**runs[0], "indices": runs[0]["indices"]}

    pico_local = int(np.argmax(scores[indices]))
    run_indices = indices[pico_local : pico_local + 1]
    return montar_run(scores, metas, run_indices)


def montar_run(
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    run_indices: np.ndarray,
) -> dict[str, Any]:
    duration = float(
        max(
            0.0,
            float(metas[int(run_indices[-1])]["end_seconds"])
            - float(metas[int(run_indices[0])]["start_seconds"]),
        )
    )
    return {
        "indices": run_indices,
        "start_seconds": float(metas[int(run_indices[0])]["start_seconds"]),
        "end_seconds": float(metas[int(run_indices[-1])]["end_seconds"]),
        "duration_seconds": duration,
        "n_sequences": int(run_indices.size),
        "score_mean": float(np.mean(scores[run_indices])),
        "score_max": float(np.max(scores[run_indices])),
    }


def listar_runs(
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    indices: np.ndarray,
    threshold: float,
    min_duration_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    acima = scores[indices] >= threshold
    runs: list[dict[str, Any]] = []
    inicio_run: int | None = None

    for local_idx, suspeita in enumerate(acima):
        if (
            inicio_run is not None
            and local_idx > 0
            and float(metas[int(indices[local_idx])]["start_seconds"])
            > float(metas[int(indices[local_idx - 1])]["end_seconds"]) + 1e-6
        ):
            runs.append(montar_run(scores, metas, indices[inicio_run:local_idx]))
            inicio_run = None
        if suspeita and inicio_run is None:
            inicio_run = local_idx
        if (not suspeita or local_idx == len(acima) - 1) and inicio_run is not None:
            fim_run = local_idx + 1 if suspeita and local_idx == len(acima) - 1 else local_idx
            runs.append(montar_run(scores, metas, indices[inicio_run:fim_run]))
            inicio_run = None

    return sorted(
        runs,
        key=lambda run: (
            bool(float(run["duration_seconds"]) >= min_duration_seconds),
            float(run["score_mean"]),
            float(run["score_max"]),
            float(run["duration_seconds"]),
        ),
        reverse=True,
    )


def intervalo_crise_real(y_true: np.ndarray, metas: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any] | None:
    segmentos = intervalos_crise_reais(y_true, metas, indices)
    if not segmentos:
        return None
    return {
        "start_seconds": float(segmentos[0]["start_seconds"]),
        "end_seconds": float(segmentos[-1]["end_seconds"]),
        "n_sequences": int(sum(int(segmento["n_sequences"]) for segmento in segmentos)),
        "segments": segmentos,
    }


def intervalos_crise_reais(
    y_true: np.ndarray,
    metas: list[dict[str, Any]],
    indices: np.ndarray,
) -> list[dict[str, Any]]:
    segmentos: list[dict[str, Any]] = []
    inicio_run: int | None = None

    for local_idx, idx in enumerate(indices):
        positivo = int(y_true[idx]) == 1
        quebra_temporal = (
            local_idx > 0
            and float(metas[int(idx)]["start_seconds"])
            > float(metas[int(indices[local_idx - 1])]["end_seconds"]) + 1e-6
        )
        if inicio_run is not None and (not positivo or quebra_temporal):
            run = indices[inicio_run:local_idx]
            segmentos.append(
                {
                    "start_seconds": float(metas[int(run[0])]["start_seconds"]),
                    "end_seconds": float(metas[int(run[-1])]["end_seconds"]),
                    "n_sequences": int(run.size),
                }
            )
            inicio_run = None
        if positivo and inicio_run is None:
            inicio_run = local_idx

    if inicio_run is not None:
        run = indices[inicio_run:]
        segmentos.append(
            {
                "start_seconds": float(metas[int(run[0])]["start_seconds"]),
                "end_seconds": float(metas[int(run[-1])]["end_seconds"]),
                "n_sequences": int(run.size),
            }
        )
    return segmentos


def melhor_overlap_com_crise(a: dict[str, Any], segmentos: list[dict[str, Any]]) -> tuple[float, float]:
    if not segmentos:
        return 0.0, 0.0
    overlaps = [(overlap_seconds(a, segmento), overlap_ratio(a, segmento)) for segmento in segmentos]
    return max(overlaps, key=lambda item: (item[1], item[0]))


def intervalo_crise_real_unico(y_true: np.ndarray, metas: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any] | None:
    crisis_indices = indices[y_true[indices] == 1]
    if crisis_indices.size == 0:
        return None
    return {
        "start_seconds": float(metas[int(crisis_indices[0])]["start_seconds"]),
        "end_seconds": float(metas[int(crisis_indices[-1])]["end_seconds"]),
        "n_sequences": int(crisis_indices.size),
    }


def overlap_seconds(a: dict[str, Any], b: dict[str, Any] | None) -> float:
    if b is None:
        return 0.0
    inicio = max(float(a["start_seconds"]), float(b["start_seconds"]))
    fim = min(float(a["end_seconds"]), float(b["end_seconds"]))
    return float(max(0.0, fim - inicio))


def overlap_ratio(a: dict[str, Any], b: dict[str, Any] | None) -> float:
    if b is None:
        return 0.0
    duracao_real = float(max(0.0, float(b["end_seconds"]) - float(b["start_seconds"])))
    if duracao_real <= 0:
        return 0.0
    return float(overlap_seconds(a, b) / duracao_real)


def estatisticas_crise_real(
    y_true: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    indices: np.ndarray,
    threshold: float,
) -> dict[str, Any] | None:
    crisis_indices = indices[y_true[indices] == 1]
    if crisis_indices.size == 0:
        return None
    return {
        "start_seconds": float(metas[int(crisis_indices[0])]["start_seconds"]),
        "end_seconds": float(metas[int(crisis_indices[-1])]["end_seconds"]),
        "n_sequences": int(crisis_indices.size),
        "score_mean": float(np.mean(scores[crisis_indices])),
        "score_max": float(np.max(scores[crisis_indices])),
        "n_sequences_above_threshold": int(np.sum(scores[crisis_indices] >= threshold)),
    }


def avaliar_combo(
    y_true: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    *,
    threshold: float,
    min_duration_seconds: float,
    max_suspicious_coverage: float = 0.7,
    min_overlap_seconds: float = 1.0,
    min_overlap_ratio: float = 0.0,
) -> dict[str, Any]:
    arquivos = sorted({str(meta["arquivo"]) for meta in metas})
    y_file_true: list[int] = []
    y_file_pred: list[int] = []
    files: list[dict[str, Any]] = []

    for arquivo in arquivos:
        indices = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo])
        runs = listar_runs(
            scores,
            metas,
            indices,
            threshold,
            min_duration_seconds=min_duration_seconds,
        )
        run = runs[0] if runs else melhor_run(scores, metas, indices, threshold)
        crise_real = intervalo_crise_real(y_true, metas, indices)
        segmentos_crise = list((crise_real or {}).get("segments") or [])
        crise_real_stats = estatisticas_crise_real(y_true, scores, metas, indices, threshold)
        label = int(np.any(y_true[indices] == 1))
        pred_raw = int(run["duration_seconds"] >= min_duration_seconds and run["score_mean"] >= threshold)
        duracao_analisada = float(
            max(
                0.0,
                float(metas[int(indices[-1])]["end_seconds"])
                - float(metas[int(indices[0])]["start_seconds"]),
            )
        )
        coverage_ratio = (
            float(run["duration_seconds"]) / duracao_analisada
            if duracao_analisada > 0
            else 0.0
        )
        cobertura_excessiva = bool(pred_raw and coverage_ratio >= max_suspicious_coverage)
        pred = int(pred_raw and not cobertura_excessiva)
        overlap, overlap_proporcao = melhor_overlap_com_crise(run, segmentos_crise)
        y_file_true.append(label)
        y_file_pred.append(pred)
        files.append(
            {
                "arquivo": arquivo,
                "label": label,
                "pred": pred,
                "pred_raw": pred_raw,
                "resultado_conclusivo": not cobertura_excessiva,
                "cobertura_excessiva": cobertura_excessiva,
                "n_sequences": int(indices.size),
                "score_max": float(np.max(scores[indices])),
                "score_mean": float(np.mean(scores[indices])),
                "trecho_suspeito": {
                    "start_seconds": run["start_seconds"],
                    "end_seconds": run["end_seconds"],
                    "duration_seconds": run["duration_seconds"],
                    "n_sequences": run["n_sequences"],
                    "score_mean": run["score_mean"],
                    "score_max": run["score_max"],
                    "threshold": float(threshold),
                    "coverage_ratio": coverage_ratio,
                    "max_suspicious_coverage": float(max_suspicious_coverage),
                    "cobertura_excessiva": cobertura_excessiva,
                    "overlap_crise_real_seconds": overlap,
                    "overlap_crise_real_ratio": overlap_proporcao,
                },
                "top_trechos_suspeitos": [
                    {
                        "start_seconds": item["start_seconds"],
                        "end_seconds": item["end_seconds"],
                        "duration_seconds": item["duration_seconds"],
                        "n_sequences": item["n_sequences"],
                        "score_mean": item["score_mean"],
                        "score_max": item["score_max"],
                        "threshold": float(threshold),
                        "overlap_crise_real_seconds": melhor_overlap_com_crise(item, segmentos_crise)[0],
                        "overlap_crise_real_ratio": melhor_overlap_com_crise(item, segmentos_crise)[1],
                    }
                    for item in runs[:5]
                ],
                "crise_real": crise_real_stats,
            }
        )

    false_positives = sum(1 for true, pred in zip(y_file_true, y_file_pred) if true == 0 and pred == 1)
    false_negatives = sum(1 for true, pred in zip(y_file_true, y_file_pred) if true == 1 and pred == 0)
    def localizou(file: dict[str, Any]) -> bool:
        trecho = file["trecho_suspeito"]
        return bool(
            trecho["overlap_crise_real_seconds"] >= min_overlap_seconds
            and trecho["overlap_crise_real_ratio"] >= min_overlap_ratio
        )

    y_file_pred_localizado = [
        int(file["pred"] == 1 and (file["label"] == 0 or localizou(file)))
        for file in files
    ]
    localized_false_negatives = sum(
        1
        for file in files
        if file["label"] == 1
        and (file["pred"] == 0 or not localizou(file))
    )
    positive_overlaps = [
        file["trecho_suspeito"]["overlap_crise_real_seconds"]
        for file in files
        if file["label"] == 1 and file["pred"] == 1
    ]
    positivos_preditos = [file for file in files if file["label"] == 1 and file["pred"] == 1]
    positives_with_overlap = sum(
        1
        for file in positivos_preditos
        if localizou(file)
    )
    positive_predictions_without_overlap = sum(
        1
        for file in positivos_preditos
        if not localizou(file)
    )
    indeterminate = sum(1 for file in files if not file["resultado_conclusivo"])

    return {
        "threshold": float(threshold),
        "min_duration_seconds": float(min_duration_seconds),
        "metrics": metricas(y_file_true, y_file_pred),
        "localized_metrics": metricas(y_file_true, y_file_pred_localizado),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "localized_false_negatives": int(localized_false_negatives),
        "indeterminate": int(indeterminate),
        "max_suspicious_coverage": float(max_suspicious_coverage),
        "min_overlap_seconds": float(min_overlap_seconds),
        "min_overlap_ratio": float(min_overlap_ratio),
        "positive_overlap_seconds_sum": float(sum(positive_overlaps)),
        "positive_predictions": int(len(positivos_preditos)),
        "positive_predictions_with_overlap": int(positives_with_overlap),
        "positive_predictions_without_overlap": int(positive_predictions_without_overlap),
        "files": files,
    }


def ranking_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    return (
        -float(item["false_positives"]),
        -float(item["localized_false_negatives"]),
        -float(item["positive_predictions_without_overlap"]),
        -float(item.get("indeterminate", 0)),
        float(item["positive_overlap_seconds_sum"]),
        float(item["localized_metrics"]["f1"]),
        float(item["metrics"]["f1"]),
        float(item["metrics"]["accuracy"]),
    )


def ranking_key_balanced(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
    return (
        float(item["localized_metrics"]["f1"]),
        float(item["localized_metrics"]["recall"]),
        -float(item["positive_predictions_without_overlap"]),
        -float(item["false_positives"]),
        -float(item["localized_false_negatives"]),
        -float(item.get("indeterminate", 0)),
        float(item["positive_overlap_seconds_sum"]),
        float(item["metrics"]["accuracy"]),
    )


def resultado_alinhado(item: dict[str, Any]) -> bool:
    if item["false_positives"] != 0 or item["false_negatives"] != 0:
        return False
    if item.get("indeterminate", 0) != 0:
        return False
    return item["positive_predictions"] == item["positive_predictions_with_overlap"]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra threshold/duracao do CNN-LSTM sequencial.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "modelos" / "sequence_cnn_lstm_features.keras")
    parser.add_argument("--scaler-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--files", nargs="+", default=["chb11_01.edf", "chb11_82.edf"])
    parser.add_argument("--max-normal-windows-per-file", type=int, default=None)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=None)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.98")
    parser.add_argument("--durations", default="10,20,30,60,120,180,240,300")
    parser.add_argument("--max-suspicious-coverage", type=float, default=0.7)
    parser.add_argument("--min-overlap-seconds", type=float, default=1.0)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.0)
    parser.add_argument("--selection-mode", choices=["conservative", "balanced"], default="balanced")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "modelos" / "calibration_sequence_cnn_lstm_chb11.json")
    args = parser.parse_args()

    import tensorflow as tf

    metadata = carregar_metadata(args.model_path, args.metadata_path)
    scaler_path = args.scaler_path or args.model_path.with_name(f"{args.model_path.stem}_scaler.pkl")
    if not args.model_path.exists():
        raise SystemExit(f"Modelo nao encontrado: {args.model_path}")
    if not scaler_path.exists():
        raise SystemExit(f"Scaler nao encontrado: {scaler_path}")

    x, y, metas = carregar_dataset(
        dataset_dir=args.dataset_dir,
        arquivos=args.files,
        window_seconds=float(metadata["window_seconds"]),
        step_seconds=float(metadata["step_seconds"]),
        max_normal_windows=args.max_normal_windows_per_file,
        max_seizure_windows=args.max_seizure_windows_per_file,
        sequence_length=int(metadata["sequence_length"]),
        sequence_stride=int(metadata["sequence_stride"]),
        feature_mode=str(metadata.get("feature_mode") or "mean"),
        canais_referencia=list(metadata.get("canais_referencia") or []) or None,
    )

    with scaler_path.open("rb") as arquivo:
        scaler = pickle.load(arquivo)
    x_scaled = aplicar_scaler(x, scaler)

    model = tf.keras.models.load_model(args.model_path)
    scores = model.predict(x_scaled, verbose=0).reshape(-1)

    thresholds = parse_float_list(args.thresholds)
    durations = parse_float_list(args.durations)
    resultados = [
        avaliar_combo(
            y,
            scores,
            metas,
            threshold=threshold,
            min_duration_seconds=duration,
            max_suspicious_coverage=args.max_suspicious_coverage,
            min_overlap_seconds=args.min_overlap_seconds,
            min_overlap_ratio=args.min_overlap_ratio,
        )
        for threshold in thresholds
        for duration in durations
    ]
    ranking = ranking_key_balanced if args.selection_mode == "balanced" else ranking_key
    resultados_ordenados = sorted(resultados, key=ranking, reverse=True)
    resultados_alinhados = [item for item in resultados_ordenados if resultado_alinhado(item)]

    payload = {
        "model_path": str(args.model_path.resolve()),
        "scaler_path": str(scaler_path.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "files": args.files,
        "metadata": metadata,
        "max_normal_windows_per_file": args.max_normal_windows_per_file,
        "max_seizure_windows_per_file": args.max_seizure_windows_per_file,
        "thresholds": thresholds,
        "durations": durations,
        "max_suspicious_coverage": float(args.max_suspicious_coverage),
        "min_overlap_seconds": float(args.min_overlap_seconds),
        "min_overlap_ratio": float(args.min_overlap_ratio),
        "selection_mode": args.selection_mode,
        "best": resultados_ordenados[0],
        "best_aligned": resultados_alinhados[0] if resultados_alinhados else None,
        "warning": (
            None
            if resultados_alinhados
            else "Nenhuma combinacao da grade acertou os arquivos positivos com trecho suspeito sobrepondo a crise real."
        ),
        "top_10": resultados_ordenados[:10],
        "all_results": resultados_ordenados,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"best": payload["best"], "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
