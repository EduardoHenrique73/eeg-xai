"""Treina CNN-LSTM com sequencias reais de janelas de features.

Entrada do modelo:
    (sequence_length, n_features)

Cada amostra e um bloco temporal de janelas consecutivas. Isso da uma
temporalidade real para a LSTM, diferente do modelo legado que tratava as 19
features de uma janela como se fossem uma sequencia.
"""

from __future__ import annotations

import argparse
import hashlib
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
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    carregar_resumos_chbmit,
    extrair_dataset_janelado_edf,
    limitar_dataset_janelado,
)
from app.ai_engine.feature_extractor import (  # noqa: E402
    extrair_metadados_edf,
    normalizar_matriz_features_robusta,
)
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
    target_mode: str = "any",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length deve ser maior que zero.")
    if sequence_stride <= 0:
        raise ValueError("sequence_stride deve ser maior que zero.")
    if target_mode not in {"any", "center"}:
        raise ValueError("target_mode deve ser 'any' ou 'center'.")

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
        target_idx = sequence_length // 2
        target_meta = bloco_meta[target_idx]
        label = int(np.any(bloco_y == 1)) if target_mode == "any" else int(bloco_y[target_idx])
        x_seq.append(x[inicio:fim])
        y_seq.append(label)
        meta_seq.append(
            {
                "arquivo": bloco_meta[0].get("arquivo"),
                "paciente": paciente_do_arquivo(str(bloco_meta[0].get("arquivo"))),
                "start_seconds": float(
                    target_meta["start_seconds"] if target_mode == "center" else bloco_meta[0]["start_seconds"]
                ),
                "end_seconds": float(
                    target_meta["end_seconds"] if target_mode == "center" else bloco_meta[-1]["end_seconds"]
                ),
                "context_start_seconds": float(bloco_meta[0]["start_seconds"]),
                "context_end_seconds": float(bloco_meta[-1]["end_seconds"]),
                "label": label,
                "context": str(target_meta.get("context") or "unknown"),
                "ictal_overlap_ratio": float(target_meta.get("ictal_overlap_ratio") or 0.0),
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
    sequence_target_mode: str = "any",
    min_window_ictal_overlap_ratio: float = 0.0,
    cache_dir: Path | None = None,
    feature_normalization: str = "global_scaler",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    if feature_normalization not in {"global_scaler", "per_edf_robust"}:
        raise ValueError("feature_normalization invalida.")
    intervalos_por_arquivo = carregar_resumos_chbmit(dataset_dir)
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    metas: list[dict[str, Any]] = []

    for arquivo in arquivos:
        path = dataset_dir / arquivo
        if not path.exists():
            raise SystemExit(f"EDF nao encontrado: {path}")
        cache_path: Path | None = None
        if cache_dir is not None:
            cache_key = hashlib.sha1(
                json.dumps(
                    {
                        "arquivo": arquivo,
                        "window_seconds": window_seconds,
                        "step_seconds": step_seconds,
                        "max_normal_windows": max_normal_windows,
                        "max_seizure_windows": max_seizure_windows,
                        "sequence_length": sequence_length,
                        "sequence_stride": sequence_stride,
                        "feature_mode": feature_mode,
                        "canais_referencia": canais_referencia or [],
                        "sequence_target_mode": sequence_target_mode,
                        "min_window_ictal_overlap_ratio": min_window_ictal_overlap_ratio,
                        "feature_normalization": feature_normalization,
                        "robust_reference": "all_edf_windows_before_sampling_v4",
                    },
                    sort_keys=True,
                ).encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()[:16]
            cache_path = cache_dir / f"{path.stem}__{cache_key}.pkl"

        if cache_path is not None and cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            x_seq, y_seq, meta_seq = cached["x"], cached["y"], cached["meta"]
        else:
            normalizacao_edf_completo = feature_normalization == "per_edf_robust"
            x_win, y_win, meta_win = extrair_dataset_janelado_edf(
                path,
                intervalos_por_arquivo.get(arquivo, []),
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                max_windows_per_class=None,
                max_normal_windows=None if normalizacao_edf_completo else max_normal_windows,
                max_seizure_windows=None if normalizacao_edf_completo else max_seizure_windows,
                canais_selecionados=canais_referencia,
                feature_mode=feature_mode,
                min_ictal_overlap_ratio=min_window_ictal_overlap_ratio,
                min_contiguous_windows=sequence_length,
            )
            if len(y_win) == 0:
                continue
            if normalizacao_edf_completo:
                # A inferencia usa o EDF inteiro como referencia robusta. Normalize
                # antes da amostragem para manter a mesma escala no treino.
                x_win = normalizar_matriz_features_robusta(x_win)
                x_win, y_win, meta_win = limitar_dataset_janelado(
                    x_win,
                    y_win,
                    meta_win,
                    max_normal_windows=max_normal_windows,
                    max_seizure_windows=max_seizure_windows,
                    min_contiguous_windows=sequence_length,
                )
            x_seq, y_seq, meta_seq = construir_sequencias(
                x_win,
                y_win,
                meta_win,
                sequence_length=sequence_length,
                sequence_stride=sequence_stride,
                target_mode=sequence_target_mode,
            )
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = cache_path.with_suffix(".tmp")
                with tmp_path.open("wb") as handle:
                    pickle.dump({"x": x_seq, "y": y_seq, "meta": meta_seq}, handle)
                tmp_path.replace(cache_path)
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


def criar_modelo(
    sequence_length: int,
    n_features: int,
    *,
    loss_mode: str = "binary_crossentropy",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> Any:
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
    loss: Any = "binary_crossentropy"
    if loss_mode == "focal":
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def criar_modelo_segmentacao(
    sequence_length: int,
    n_features: int,
    *,
    loss_mode: str = "binary_crossentropy",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> Any:
    """CNN-BiLSTM que retorna uma probabilidade para cada passo temporal."""
    import tensorflow as tf
    from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, Dropout, Input, LSTM, TimeDistributed
    from tensorflow.keras.models import Sequential

    loss: Any = "binary_crossentropy"
    if loss_mode == "focal":
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            alpha=focal_alpha,
            gamma=focal_gamma,
        )
    model = Sequential(
        [
            Input(shape=(sequence_length, n_features)),
            Conv1D(64, 3, activation="relu", padding="same"),
            Dropout(0.2),
            Conv1D(64, 3, activation="relu", padding="same"),
            Bidirectional(LSTM(48, return_sequences=True)),
            Dropout(0.25),
            TimeDistributed(Dense(32, activation="relu")),
            Dropout(0.2),
            TimeDistributed(Dense(1, activation="sigmoid")),
        ],
        name="sequence_cnn_bilstm_segmentation_eeg",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
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


def resolver_sample_weights(
    metas: list[dict[str, Any]],
    *,
    boundary_weight: float,
    y: np.ndarray | None = None,
    class_weight: dict[int, float] | None = None,
) -> np.ndarray:
    if boundary_weight <= 0:
        raise ValueError("boundary_weight deve ser maior que zero.")
    pesos = np.asarray(
        [boundary_weight if meta.get("context") == "boundary" else 1.0 for meta in metas],
        dtype=np.float32,
    )
    if class_weight is not None:
        if y is None or len(y) != len(pesos):
            raise ValueError("y deve ter o mesmo tamanho de metas ao combinar pesos.")
        pesos *= np.asarray([class_weight[int(label)] for label in y], dtype=np.float32)
    return pesos


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
    parser.add_argument("--sequence-target-mode", choices=["any", "center"], default="center")
    parser.add_argument("--min-window-ictal-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--boundary-sample-weight", type=float, default=1.5)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "modelos" / "sequence_dataset_cache",
    )
    parser.add_argument(
        "--feature-normalization",
        choices=["global_scaler", "per_edf_robust"],
        default="global_scaler",
    )
    parser.add_argument(
        "--feature-mode",
        choices=[
            FEATURE_MODE_MEAN,
            FEATURE_MODE_PER_CHANNEL,
            FEATURE_MODE_TIME_FREQUENCY,
            FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
            FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
            FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
            FEATURE_MODE_RAW_SIGNAL,
        ],
        default=settings.ai_sequence_feature_mode,
    )
    parser.add_argument("--channel-reference-edf", default=settings.ai_sequence_channel_reference_edf)
    parser.add_argument("--epochs", type=int, default=settings.ai_sequence_epochs)
    parser.add_argument("--batch-size", type=int, default=settings.ai_sequence_batch_size)
    parser.add_argument("--class-weight-mode", choices=["balanced", "manual", "none"], default="balanced")
    parser.add_argument("--positive-class-weight", type=float, default=2.0)
    parser.add_argument(
        "--loss",
        choices=["binary_crossentropy", "focal"],
        default="focal",
        help="Focal loss prioriza exemplos dificeis e reduz dominio de normais faceis.",
    )
    parser.add_argument("--focal-alpha", type=float, default=0.35)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument(
        "--early-stopping-monitor",
        choices=["val_pr_auc", "val_loss"],
        default="val_pr_auc",
        help="Metrica usada para restaurar o melhor checkpoint do treino.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-duration-seconds", type=float, default=10.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-output", type=Path, default=PROJECT_ROOT / "modelos" / "sequence_cnn_lstm_features.keras")
    parser.add_argument("--metrics-output", type=Path, default=PROJECT_ROOT / "modelos" / "metrics_sequence_cnn_lstm_features.json")
    args = parser.parse_args()

    import tensorflow as tf

    train_files = carregar_manifesto(args.manifest)
    canais_referencia = None
    if args.feature_mode in {
        FEATURE_MODE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    }:
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
        sequence_target_mode=args.sequence_target_mode,
        min_window_ictal_overlap_ratio=args.min_window_ictal_overlap_ratio,
        cache_dir=args.cache_dir,
        feature_normalization=args.feature_normalization,
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
        sequence_target_mode=args.sequence_target_mode,
        min_window_ictal_overlap_ratio=args.min_window_ictal_overlap_ratio,
        cache_dir=args.cache_dir,
        feature_normalization=args.feature_normalization,
    )

    if len(np.unique(y_train)) < 2:
        raise SystemExit("Treino requer duas classes.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(args.random_state)
    scaler = ajustar_scaler(x_train)
    x_train_scaled = aplicar_scaler(x_train, scaler)
    x_holdout_scaled = aplicar_scaler(x_holdout, scaler)

    model = criar_modelo(
        args.sequence_length,
        x_train.shape[2],
        loss_mode=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    pesos_classe = resolver_class_weight(
        y_train,
        mode=args.class_weight_mode,
        positive_weight=args.positive_class_weight,
    )
    pesos_amostras = resolver_sample_weights(
        meta_train,
        boundary_weight=args.boundary_sample_weight,
        y=y_train,
        class_weight=None if args.loss == "focal" else pesos_classe,
    )
    callbacks = []
    if args.early_stopping_patience > 0:
        monitor_mode = "max" if args.early_stopping_monitor == "val_pr_auc" else "min"
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=args.early_stopping_monitor,
                mode=monitor_mode,
                patience=args.early_stopping_patience,
                min_delta=0.001,
                restore_best_weights=True,
            )
        )
    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        sample_weight=pesos_amostras,
        validation_data=(x_holdout_scaled, y_holdout),
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
            "sequence_target_mode": args.sequence_target_mode,
            "min_window_ictal_overlap_ratio": args.min_window_ictal_overlap_ratio,
            "feature_normalization": args.feature_normalization,
            "robust_reference": "all_edf_windows_before_sampling_v4",
            "window_seconds": args.window_seconds,
            "step_seconds": args.step_seconds,
            "n_features": int(x_train.shape[2]),
            "canais_referencia": list(canais_referencia or []),
            "aggregation": "continuous_suspicious_sequence",
            "loss": args.loss,
            "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
            "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
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
        "sequence_target_mode": args.sequence_target_mode,
        "min_window_ictal_overlap_ratio": args.min_window_ictal_overlap_ratio,
        "boundary_sample_weight": args.boundary_sample_weight,
        "validation_source": "holdout_patients",
        "feature_normalization": args.feature_normalization,
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "max_normal_windows_per_file": args.max_normal_windows_per_file,
        "max_seizure_windows_per_file": args.max_seizure_windows_per_file,
        "max_holdout_normal_windows_per_file": args.max_holdout_normal_windows_per_file,
        "max_holdout_seizure_windows_per_file": args.max_holdout_seizure_windows_per_file,
        "class_weight_mode": args.class_weight_mode,
        "loss": args.loss,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "class_weight": (
            {str(k): float(v) for k, v in pesos_classe.items()}
            if pesos_classe is not None
            else None
        ),
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_monitor": args.early_stopping_monitor,
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
