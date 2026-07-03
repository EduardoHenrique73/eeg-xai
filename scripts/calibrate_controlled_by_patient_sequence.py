"""Calibra threshold/duracao no experimento intra-paciente.

O script treina um modelo por split, guarda os scores e depois testa uma grade
de threshold e duracao minima por paciente. A melhor regra prioriza:
1. nao gerar falso positivo no EDF normal;
2. nao perder o EDF com crise;
3. sobrepor o trecho suspeito com a crise real.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_sequence_cnn_lstm import avaliar_combo, parse_float_list  # noqa: E402
from scripts.train_controlled_by_patient_sequence import agrupar_edfs, montar_splits  # noqa: E402
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    ajustar_scaler,
    aplicar_scaler,
    carregar_dataset,
    class_weights,
    criar_modelo,
    metricas_binarias,
)


def treinar_split_scores(
    *,
    dataset_dir: Path,
    split: dict[str, Any],
    sequence_length: int,
    sequence_stride: int,
    window_seconds: float,
    step_seconds: float,
    max_normal_windows_per_file: int,
    max_seizure_windows_per_file: int,
    epochs: int,
    batch_size: int,
    random_state: int,
) -> dict[str, Any]:
    import tensorflow as tf

    print(f"\n=== Split: {split['nome']} ===")
    print(f"Treino: {', '.join(split['train_files'])}")
    print(f"Teste: {', '.join(split['holdout_files'])}")

    x_train, y_train, _meta_train = carregar_dataset(
        dataset_dir=dataset_dir,
        arquivos=split["train_files"],
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_normal_windows=max_normal_windows_per_file,
        max_seizure_windows=max_seizure_windows_per_file,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )
    x_holdout, y_holdout, meta_holdout = carregar_dataset(
        dataset_dir=dataset_dir,
        arquivos=split["holdout_files"],
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_normal_windows=None,
        max_seizure_windows=None,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )
    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"Split sem duas classes no treino: {split['nome']}")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_state)
    scaler = ajustar_scaler(x_train)
    x_train_scaled = aplicar_scaler(x_train, scaler)
    x_holdout_scaled = aplicar_scaler(x_holdout, scaler)

    model = criar_modelo(sequence_length, x_train.shape[2])
    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=class_weights(y_train),
        validation_split=0.15,
    )
    scores = model.predict(x_holdout_scaled, verbose=0).reshape(-1)
    return {
        "split_name": split["nome"],
        "split_type": split["tipo"],
        "train_files": split["train_files"],
        "holdout_files": split["holdout_files"],
        "y_holdout": y_holdout,
        "meta_holdout": meta_holdout,
        "scores": scores,
        "train": {
            "n_sequences": int(len(y_train)),
            "class_counts": {
                "0": int(np.sum(y_train == 0)),
                "1": int(np.sum(y_train == 1)),
            },
        },
        "holdout": {
            "n_sequences": int(len(y_holdout)),
            "class_counts": {
                "0": int(np.sum(y_holdout == 0)),
                "1": int(np.sum(y_holdout == 1)),
            },
        },
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }


def avaliar_paciente_combo(
    split_scores: list[dict[str, Any]],
    *,
    threshold: float,
    min_duration_seconds: float,
) -> dict[str, Any]:
    split_results: list[dict[str, Any]] = []
    y_true_files: list[int] = []
    y_pred_files: list[int] = []
    false_positives = 0
    false_negatives = 0
    crisis_overlap_sum = 0.0
    crisis_predictions = 0
    crisis_predictions_with_overlap = 0

    for split in split_scores:
        edf_eval = avaliar_combo(
            split["y_holdout"],
            split["scores"],
            split["meta_holdout"],
            threshold=threshold,
            min_duration_seconds=min_duration_seconds,
        )
        file_eval = edf_eval["files"][0]
        y_true_files.append(int(file_eval["label"]))
        y_pred_files.append(int(file_eval["pred"]))
        if file_eval["label"] == 0 and file_eval["pred"] == 1:
            false_positives += 1
        if file_eval["label"] == 1 and file_eval["pred"] == 0:
            false_negatives += 1
        if file_eval["label"] == 1 and file_eval["pred"] == 1:
            crisis_predictions += 1
            overlap = float(file_eval["trecho_suspeito"]["overlap_crise_real_seconds"])
            crisis_overlap_sum += overlap
            if overlap > 0:
                crisis_predictions_with_overlap += 1
        split_results.append(
            {
                "split_name": split["split_name"],
                "split_type": split["split_type"],
                "train_files": split["train_files"],
                "holdout_files": split["holdout_files"],
                "train": split["train"],
                "holdout": {
                    **split["holdout"],
                    "sequence_metrics": metricas_binarias(
                        split["y_holdout"],
                        (split["scores"] >= threshold).astype(np.int64),
                    ),
                    "edf_eval": edf_eval,
                },
            }
        )

    return {
        "threshold": float(threshold),
        "min_duration_seconds": float(min_duration_seconds),
        "file_metrics": metricas_binarias(np.asarray(y_true_files), np.asarray(y_pred_files)),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "crisis_overlap_seconds_sum": float(crisis_overlap_sum),
        "crisis_predictions": int(crisis_predictions),
        "crisis_predictions_with_overlap": int(crisis_predictions_with_overlap),
        "splits": split_results,
    }


def ranking_key(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    aligned_missing = item["crisis_predictions"] - item["crisis_predictions_with_overlap"]
    return (
        -float(item["false_positives"]),
        -float(item["false_negatives"]),
        -float(aligned_missing),
        float(item["crisis_overlap_seconds_sum"]),
        float(item["file_metrics"]["f1"]),
        float(item["file_metrics"]["accuracy"]),
        -float(item["min_duration_seconds"]),
    )


def resumir_best(paciente: str, best: dict[str, Any]) -> dict[str, Any]:
    crisis_split = next((s for s in best["splits"] if s["split_type"] == "crise_holdout"), None)
    normal_split = next((s for s in best["splits"] if s["split_type"] == "normal_holdout"), None)
    crisis_file = crisis_split["holdout"]["edf_eval"]["files"][0] if crisis_split else None
    normal_file = normal_split["holdout"]["edf_eval"]["files"][0] if normal_split else None
    return {
        "paciente": paciente,
        "threshold": best["threshold"],
        "min_duration_seconds": best["min_duration_seconds"],
        "file_accuracy": best["file_metrics"]["accuracy"],
        "false_positives": best["false_positives"],
        "false_negatives": best["false_negatives"],
        "localizou_crise_real": bool(
            crisis_file
            and crisis_file["pred"] == 1
            and crisis_file["trecho_suspeito"]["overlap_crise_real_seconds"] > 0
        ),
        "crise_overlap_seconds": (
            crisis_file["trecho_suspeito"]["overlap_crise_real_seconds"] if crisis_file else None
        ),
        "normal_pred": normal_file["pred"] if normal_file else None,
        "normal_trecho_seconds": (
            normal_file["trecho_suspeito"]["duration_seconds"] if normal_file else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibra CNN-LSTM intra-paciente em varios pacientes.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--patients", nargs="+", default=["chb01", "chb03", "chb13", "chb14", "chb15"])
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=2)
    parser.add_argument("--max-normal-windows-per-file", type=int, default=260)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=80)
    parser.add_argument("--max-train-seizure-files", type=int, default=999)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--thresholds", default="0.5,0.6,0.7,0.8,0.85,0.9,0.95")
    parser.add_argument("--durations", default="10,20,30,45,60,90,120,180")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "modelos" / "controlled_by_patient_calibration.json")
    args = parser.parse_args()

    pacientes = agrupar_edfs(args.dataset_dir)
    thresholds = parse_float_list(args.thresholds)
    durations = parse_float_list(args.durations)
    all_results: dict[str, Any] = {}
    summary: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for paciente in args.patients:
        splits = montar_splits(
            paciente,
            pacientes.get(paciente, {"crise": [], "normal": []}),
            max_train_seizure_files=args.max_train_seizure_files,
        )
        if not splits:
            skipped.append({"paciente": paciente, "motivo": "sem splits suficientes"})
            continue

        trained_splits = [
            treinar_split_scores(
                dataset_dir=args.dataset_dir,
                split=split,
                sequence_length=args.sequence_length,
                sequence_stride=args.sequence_stride,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_normal_windows_per_file=args.max_normal_windows_per_file,
                max_seizure_windows_per_file=args.max_seizure_windows_per_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                random_state=args.random_state,
            )
            for split in splits
        ]
        combos = [
            avaliar_paciente_combo(
                trained_splits,
                threshold=threshold,
                min_duration_seconds=duration,
            )
            for threshold in thresholds
            for duration in durations
        ]
        combos_ordenados = sorted(combos, key=ranking_key, reverse=True)
        best = combos_ordenados[0]
        all_results[paciente] = {
            "best": best,
            "top_10": combos_ordenados[:10],
        }
        summary.append(resumir_best(paciente, best))

    payload = {
        "experiment": "controlled_by_patient_sequence_calibration",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "thresholds": thresholds,
        "durations": durations,
        "epochs": args.epochs,
        "summary": summary,
        "skipped": skipped,
        "patients": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "skipped": skipped, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
