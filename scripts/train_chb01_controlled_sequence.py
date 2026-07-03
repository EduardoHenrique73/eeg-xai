"""Experimento controlado do CNN-LSTM sequencial no paciente chb01.

Objetivo: separar problema de arquitetura/features do problema de
generalizacao entre pacientes. O script treina e testa apenas em EDFs do
chb01, usando splits pequenos e explicitos.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_sequence_cnn_lstm import avaliar_combo  # noqa: E402
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    ajustar_scaler,
    aplicar_scaler,
    carregar_dataset,
    class_weights,
    criar_modelo,
    metricas_binarias,
)


def treinar_e_avaliar_split(
    *,
    dataset_dir: Path,
    nome: str,
    train_files: list[str],
    holdout_files: list[str],
    sequence_length: int,
    sequence_stride: int,
    window_seconds: float,
    step_seconds: float,
    max_normal_windows_per_file: int,
    max_seizure_windows_per_file: int,
    epochs: int,
    batch_size: int,
    threshold: float,
    min_duration_seconds: float,
    random_state: int,
    verbose: int = 1,
) -> dict[str, Any]:
    import tensorflow as tf

    print(f"\n=== Split: {nome} ===")
    print(f"Treino: {', '.join(train_files)}")
    print(f"Teste: {', '.join(holdout_files)}")

    x_train, y_train, meta_train = carregar_dataset(
        dataset_dir=dataset_dir,
        arquivos=train_files,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_normal_windows=max_normal_windows_per_file,
        max_seizure_windows=max_seizure_windows_per_file,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )
    x_holdout, y_holdout, meta_holdout = carregar_dataset(
        dataset_dir=dataset_dir,
        arquivos=holdout_files,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        max_normal_windows=None,
        max_seizure_windows=None,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
    )

    if len(np.unique(y_train)) < 2:
        raise SystemExit(f"Split {nome} nao tem duas classes no treino.")

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
        verbose=verbose,
        class_weight=class_weights(y_train),
        validation_split=0.15,
    )

    scores = model.predict(x_holdout_scaled, verbose=0).reshape(-1)
    pred = (scores >= threshold).astype(np.int64)
    edf_eval = avaliar_combo(
        y_holdout,
        scores,
        meta_holdout,
        threshold=threshold,
        min_duration_seconds=min_duration_seconds,
    )

    return {
        "name": nome,
        "train_files": train_files,
        "holdout_files": holdout_files,
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
            "sequence_metrics": metricas_binarias(y_holdout, pred),
            "edf_eval": edf_eval,
        },
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina CNN-LSTM sequencial controlado no chb01.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--window-seconds", type=float, default=4.0)
    parser.add_argument("--step-seconds", type=float, default=2.0)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--sequence-stride", type=int, default=2)
    parser.add_argument("--max-normal-windows-per-file", type=int, default=260)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-seconds", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "modelos" / "controlled_chb01_sequence_eval.json")
    args = parser.parse_args()

    splits = [
        {
            "nome": "chb01_crise_holdout_15",
            "train_files": ["chb01_01.edf", "chb01_03.edf", "chb01_04.edf"],
            "holdout_files": ["chb01_15.edf"],
        },
        {
            "nome": "chb01_normal_holdout_01",
            "train_files": ["chb01_03.edf", "chb01_04.edf", "chb01_15.edf"],
            "holdout_files": ["chb01_01.edf"],
        },
    ]

    resultados = [
        treinar_e_avaliar_split(
            dataset_dir=args.dataset_dir,
            nome=split["nome"],
            train_files=split["train_files"],
            holdout_files=split["holdout_files"],
            sequence_length=args.sequence_length,
            sequence_stride=args.sequence_stride,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_normal_windows_per_file=args.max_normal_windows_per_file,
            max_seizure_windows_per_file=args.max_seizure_windows_per_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            threshold=args.threshold,
            min_duration_seconds=args.min_duration_seconds,
            random_state=args.random_state,
        )
        for split in splits
    ]

    payload = {
        "experiment": "controlled_chb01_sequence_cnn_lstm",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "threshold": args.threshold,
        "min_duration_seconds": args.min_duration_seconds,
        "splits": resultados,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Resultado salvo em: {args.output}")


if __name__ == "__main__":
    main()
