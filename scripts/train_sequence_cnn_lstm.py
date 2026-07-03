"""Treina CNN-LSTM com sequencias reais de janelas de features.

Entrada do modelo:
    (sequence_length, n_features)

Cada amostra e um bloco temporal de janelas consecutivas. Isso da uma
temporalidade real para a LSTM, diferente do modelo legado que tratava as 19
features de uma janela como se fossem uma sequencia.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai_engine.training import (  # noqa: E402
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    FEATURE_MODE_RAW_SIGNAL,
    FEATURE_MODE_TIME_FREQUENCY,
    FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    carregar_resumos_chbmit,
    extrair_dataset_janelado_edf,
)
from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402
from app.config import get_settings  # noqa: E402


def carregar_manifesto(path: Path) -> list[str]:
    arquivos = [
        linha.strip()
        for linha in path.read_text(encoding="utf-8").splitlines()
        if linha.strip() and not linha.strip().startswith("#")
    ]
    if not arquivos:
        raise SystemExit(f"Manifesto vazio: {path}")
    return arquivos


def paciente_do_arquivo(nome: str) -> str:
    match = re.match(r"(chb\d+)", nome.lower())
    if not match:
        raise ValueError(f"Nao foi possivel identificar paciente em {nome}")
    return match.group(1)


def construir_sequencias(
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
    *,
    sequence_length: int,
    sequence_stride: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length deve ser maior que zero.")
    if sequence_stride <= 0:
        raise ValueError("sequence_stride deve ser maior que zero.")

    if len(y) < sequence_length:
        return (
            np.empty((0, sequence_length, x.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            [],
        )

    def bloco_temporal_continuo(bloco_meta: list[dict[str, Any]]) -> bool:
        for anterior, atual in zip(bloco_meta, bloco_meta[1:]):
            fim_anterior = float(anterior["end_seconds"])
            inicio_atual = float(atual["start_seconds"])
            if inicio_atual > fim_anterior + 1e-6:
                return False
        return True

    x_seq: list[np.ndarray] = []
    y_seq: list[int] = []
    meta_seq: list[dict[str, Any]] = []

    for inicio in range(0, len(y) - sequence_length + 1, sequence_stride):
        fim = inicio + sequence_length
        bloco_y = y[inicio:fim]
        bloco_meta = meta[inicio:fim]
        if not bloco_temporal_continuo(bloco_meta):
            continue
        x_seq.append(x[inicio:fim])
        y_seq.append(int(np.any(bloco_y == 1)))
        meta_seq.append(
            {
                "arquivo": bloco_meta[0].get("arquivo"),
                "paciente": paciente_do_arquivo(str(bloco_meta[0].get("arquivo"))),
                "start_seconds": float(bloco_meta[0]["start_seconds"]),
                "end_seconds": float(bloco_meta[-1]["end_seconds"]),
                "label": int(np.any(bloco_y == 1)),
                "n_janelas_crise": int(np.sum(bloco_y == 1)),
            }
        )

    return (
        np.asarray(x_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.int64),
        meta_seq,
    )


def carregar_dataset(
    *,
    dataset_dir: Path,
    arquivos: list[str],
    window_seconds: float,
    step_seconds: float,
    max_normal_windows: int | None,
    max_seizure_windows: int | None,
    sequence_length: int,
    sequence_stride: int,
    feature_mode: str = FEATURE_MODE_MEAN,
    canais_referencia: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    intervalos_por_arquivo = carregar_resumos_chbmit(dataset_dir)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    for arquivo in arquivos:
        path = dataset_dir / arquivo
        if not path.exists():
            raise SystemExit(f"EDF nao encontrado: {path}")
        x_win, y_win, meta_win = extrair_dataset_janelado_edf(
            path,
            intervalos_por_arquivo.get(arquivo, []),
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            max_windows_per_class=None,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
            canais_selecionados=canais_referencia,
            feature_mode=feature_mode,
        )
        if len(y_win) == 0:
            continue
        x_seq, y_seq, meta_seq = construir_sequencias(
            x_win,
            y_win,
            meta_win,
            sequence_length=sequence_length,
            sequence_stride=sequence_stride,
        )
        if len(y_seq) == 0:
            continue
        x_parts.append(x_seq)
        y_parts.append(y_seq)
        metas.extend(meta_seq)
        print(
            f"{arquivo}: {len(y_seq)} sequencias "
            f"(normal={int(np.sum(y_seq == 0))}, crise={int(np.sum(y_seq == 1))})"
        )

    if not x_parts:
        raise SystemExit("Nenhuma sequencia gerada.")
    return np.vstack(x_parts), np.concatenate(y_parts), metas


def criar_modelo(sequence_length: int, n_features: int) -> Any:
    import tensorflow as tf
    from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            Conv1D(32, 3, activation="relu", padding="same"),
            Dropout(0.15),
            Conv1D(64, 3, activation="relu", padding="same"),
            LSTM(64, return_sequences=True),
            Dropout(0.25),
            LSTM(32, return_sequences=False),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ],
        name="sequence_cnn_lstm_eeg",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {int(cls): float(total / (len(classes) * count)) for cls, count in zip(classes, counts)}


def resolver_class_weight(
    y: np.ndarray,
    *,
    mode: str,
    positive_weight: float,
) -> dict[int, float] | None:
    if mode == "none":
        return None
    if mode == "manual":
        return {0: 1.0, 1: float(positive_weight)}
    return class_weights(y)


def ajustar_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    n_seq, seq_len, n_features = x_train.shape
    scaler.fit(x_train.reshape(n_seq * seq_len, n_features))
    return scaler


def aplicar_scaler(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n_seq, seq_len, n_features = x.shape
    x_scaled = scaler.transform(x.reshape(n_seq * seq_len, n_features))
    return x_scaled.reshape(n_seq, seq_len, n_features).astype(np.float32)


def metricas_binarias(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def avaliar_por_edf(
    y_true: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    *,
    threshold: float,
    min_duration_seconds: float,
) -> dict[str, Any]:
    arquivos = sorted({str(meta["arquivo"]) for meta in metas})
    linhas: list[dict[str, Any]] = []
    y_file_true: list[int] = []
    y_file_pred: list[int] = []

    for arquivo in arquivos:
        indices = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo])
        if indices.size == 0:
            continue
        acima = scores[indices] >= threshold
        melhor_inicio: int | None = None
        melhor_fim: int | None = None
        inicio_run: int | None = None
        for local_idx, suspeita in enumerate(acima):
            if (
                inicio_run is not None
                and local_idx > 0
                and float(metas[int(indices[local_idx])]["start_seconds"])
                > float(metas[int(indices[local_idx - 1])]["end_seconds"]) + 1e-6
            ):
                if melhor_inicio is None or (local_idx - inicio_run) > (melhor_fim - melhor_inicio):  # type: ignore[operator]
                    melhor_inicio, melhor_fim = inicio_run, local_idx
                inicio_run = None
            if suspeita and inicio_run is None:
                inicio_run = local_idx
            if (not suspeita or local_idx == len(acima) - 1) and inicio_run is not None:
                fim_run = local_idx + 1 if suspeita and local_idx == len(acima) - 1 else local_idx
                if melhor_inicio is None or (fim_run - inicio_run) > (melhor_fim - melhor_inicio):  # type: ignore[operator]
                    melhor_inicio, melhor_fim = inicio_run, fim_run
                inicio_run = None

        if melhor_inicio is None or melhor_fim is None:
            pico_local = int(np.argmax(scores[indices]))
            melhor_inicio = pico_local
            melhor_fim = pico_local + 1

        run_indices = indices[melhor_inicio:melhor_fim]
        start = float(metas[int(run_indices[0])]["start_seconds"])
        end = float(metas[int(run_indices[-1])]["end_seconds"])
        duration = float(max(0.0, end - start))
        label = int(np.any(y_true[indices] == 1))
        pred = int(duration >= min_duration_seconds and np.mean(scores[run_indices]) >= threshold)
        y_file_true.append(label)
        y_file_pred.append(pred)
        linhas.append(
            {
                "arquivo": arquivo,
                "label": label,
                "pred": pred,
                "n_sequences": int(indices.size),
                "score_max": float(np.max(scores[indices])),
                "score_mean": float(np.mean(scores[indices])),
                "trecho_suspeito": {
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": duration,
                    "n_sequences": int(run_indices.size),
                    "score_mean": float(np.mean(scores[run_indices])),
                    "score_max": float(np.max(scores[run_indices])),
                    "threshold": float(threshold),
                },
            }
        )

    return {
        "metrics": metricas_binarias(np.asarray(y_file_true), np.asarray(y_file_pred)),
        "files": linhas,
    }


def salvar_metadata(path: Path, payload: dict[str, Any]) -> Path:
    metadata_path = path.with_name(f"{path.stem}_metadata.json")
    metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata_path


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Treina CNN-LSTM sequencial por janelas.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "dataset_amostra" / "curated_train_files_expanded_23ch.txt")
    parser.add_argument("--holdout-files", nargs="+", default=["chb11_01.edf", "chb11_82.edf"])
    parser.add_argument("--window-seconds", type=float, default=settings.ai_sequence_window_seconds)
    parser.add_argument("--step-seconds", type=float, default=settings.ai_sequence_step_seconds)
    parser.add_argument("--max-normal-windows-per-file", type=int, default=settings.ai_sequence_max_normal_windows_per_file)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=settings.ai_sequence_max_seizure_windows_per_file)
    parser.add_argument("--max-holdout-normal-windows-per-file", type=int, default=None)
    parser.add_argument("--max-holdout-seizure-windows-per-file", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=settings.ai_sequence_length)
    parser.add_argument("--sequence-stride", type=int, default=settings.ai_sequence_stride)
    parser.add_argument(
        "--feature-mode",
        choices=[
            FEATURE_MODE_MEAN,
            FEATURE_MODE_PER_CHANNEL,
            FEATURE_MODE_TIME_FREQUENCY,
            FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
            FEATURE_MODE_RAW_SIGNAL,
        ],
        default=settings.ai_sequence_feature_mode,
    )
    parser.add_argument("--channel-reference-edf", default=settings.ai_sequence_channel_reference_edf)
    parser.add_argument("--epochs", type=int, default=settings.ai_sequence_epochs)
    parser.add_argument("--batch-size", type=int, default=settings.ai_sequence_batch_size)
    parser.add_argument("--class-weight-mode", choices=["balanced", "manual", "none"], default="balanced")
    parser.add_argument("--positive-class-weight", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-seconds", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "modelos" / "sequence_cnn_lstm_features.keras")
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "modelos" / "metrics_sequence_cnn_lstm_features.json")
    args = parser.parse_args()

    import tensorflow as tf

    train_files = carregar_manifesto(args.manifest)
    canais_referencia = None
    if args.feature_mode in {FEATURE_MODE_PER_CHANNEL, FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL}:
        ref_path = args.dataset_dir / (args.channel_reference_edf or train_files[0])
        canais_referencia = list(extrair_metadados_edf(ref_path)["canais_eeg"])

    x_train, y_train, meta_train = carregar_dataset(
        dataset_dir=args.dataset_dir,
        arquivos=train_files,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        max_normal_windows=args.max_normal_windows_per_file,
        max_seizure_windows=args.max_seizure_windows_per_file,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        feature_mode=args.feature_mode,
        canais_referencia=canais_referencia,
    )
    x_holdout, y_holdout, meta_holdout = carregar_dataset(
        dataset_dir=args.dataset_dir,
        arquivos=args.holdout_files,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        max_normal_windows=args.max_holdout_normal_windows_per_file,
        max_seizure_windows=args.max_holdout_seizure_windows_per_file,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        feature_mode=args.feature_mode,
        canais_referencia=canais_referencia,
    )

    if len(np.unique(y_train)) < 2:
        raise SystemExit("Treino requer duas classes.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.random_state)
    scaler = ajustar_scaler(x_train)
    x_train_scaled = aplicar_scaler(x_train, scaler)
    x_holdout_scaled = aplicar_scaler(x_holdout, scaler)

    model = criar_modelo(args.sequence_length, x_train.shape[2])
    pesos_classe = resolver_class_weight(
        y_train,
        mode=args.class_weight_mode,
        positive_weight=args.positive_class_weight,
    )
    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        class_weight=pesos_classe,
        validation_split=0.15,
        callbacks=callbacks,
    )

    scores_holdout = model.predict(x_holdout_scaled, verbose=0).reshape(-1)
    pred_holdout = (scores_holdout >= args.threshold).astype(np.int64)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_output)
    scaler_path = args.model_output.with_name(f"{args.model_output.stem}_scaler.pkl")
    with scaler_path.open("wb") as arquivo:
        pickle.dump(scaler, arquivo)
    metadata_path = salvar_metadata(
        args.model_output,
        {
            "model_type": "sequence_cnn_lstm",
            "feature_mode": args.feature_mode,
            "sequence_length": args.sequence_length,
            "sequence_stride": args.sequence_stride,
            "window_seconds": args.window_seconds,
            "step_seconds": args.step_seconds,
            "n_features": int(x_train.shape[2]),
            "canais_referencia": list(canais_referencia or []),
            "aggregation": "continuous_suspicious_sequence",
        },
    )

    metrics = {
        "model_type": "sequence_cnn_lstm",
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "train_files": train_files,
        "holdout_files": args.holdout_files,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "feature_mode": args.feature_mode,
        "canais_referencia": list(canais_referencia or []),
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "max_normal_windows_per_file": args.max_normal_windows_per_file,
        "max_seizure_windows_per_file": args.max_seizure_windows_per_file,
        "max_holdout_normal_windows_per_file": args.max_holdout_normal_windows_per_file,
        "max_holdout_seizure_windows_per_file": args.max_holdout_seizure_windows_per_file,
        "class_weight_mode": args.class_weight_mode,
        "class_weight": (
            {str(k): float(v) for k, v in pesos_classe.items()}
            if pesos_classe is not None
            else None
        ),
        "early_stopping_patience": args.early_stopping_patience,
        "threshold": args.threshold,
        "min_duration_seconds": args.min_duration_seconds,
        "training_history": {
            key: [float(v) for v in values]
            for key, values in history.history.items()
        },
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
            "sequence_metrics": metricas_binarias(y_holdout, pred_holdout),
            "edf_metrics": avaliar_por_edf(
                y_holdout,
                scores_holdout,
                meta_holdout,
                threshold=args.threshold,
                min_duration_seconds=args.min_duration_seconds,
            ),
        },
        "artifacts": {
            "model_path": str(args.model_output.resolve()),
            "scaler_path": str(scaler_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
        },
    }

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics["holdout"], indent=2, ensure_ascii=False))
    print(f"Modelo salvo em: {args.model_output}")
    print(f"Metricas salvas em: {args.metrics_output}")


if __name__ == "__main__":
    main()
