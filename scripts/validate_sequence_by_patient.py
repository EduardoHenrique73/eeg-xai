"""Valida CNN-LSTM sequencial com separacao por paciente.

Este e o experimento alinhado ao objetivo principal do projeto: treinar em
varios pacientes e testar em um paciente nunca visto. A calibracao de
threshold/duracao e feita apenas nos pacientes de treino e aplicada no paciente
de teste.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.calibrate_sequence_cnn_lstm import avaliar_combo, parse_float_list  # noqa: E402
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    FEATURE_MODE_RAW_SIGNAL,
    FEATURE_MODE_TIME_FREQUENCY,
    FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    ajustar_scaler,
    aplicar_scaler,
    construir_sequencias,
    criar_modelo,
    metricas_binarias,
    paciente_do_arquivo,
    resolver_class_weight,
)
from app.ai_engine.feature_extractor import extrair_metadados_edf  # noqa: E402
from app.ai_engine.training import carregar_resumos_chbmit, extrair_dataset_janelado_edf  # noqa: E402
from app.config import get_settings  # noqa: E402


def carregar_arquivos(dataset_dir: Path, manifest: Path | None) -> list[str]:
    if manifest is not None:
        arquivos = [
            linha.strip()
            for linha in manifest.read_text(encoding="utf-8").splitlines()
            if linha.strip() and not linha.strip().startswith("#")
        ]
    else:
        arquivos = sorted(path.name for path in dataset_dir.glob("*.edf"))
    if not arquivos:
        raise SystemExit("Nenhum EDF encontrado para avaliacao.")
    return arquivos


def agrupar_por_paciente(arquivos: list[str]) -> dict[str, list[str]]:
    grupos: dict[str, list[str]] = defaultdict(list)
    for arquivo in arquivos:
        grupos[paciente_do_arquivo(arquivo)].append(arquivo)
    return dict(sorted(grupos.items()))


class SequenceDatasetCache:
    def __init__(
        self,
        *,
        dataset_dir: Path,
        window_seconds: float,
        step_seconds: float,
        max_normal_windows: int | None,
        max_seizure_windows: int | None,
        sequence_length: int,
        sequence_stride: int,
        feature_mode: str,
        canais_referencia: list[str] | None,
        disk_cache_enabled: bool,
        disk_cache_path: Path,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.max_normal_windows = max_normal_windows
        self.max_seizure_windows = max_seizure_windows
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.feature_mode = feature_mode
        self.canais_referencia = canais_referencia
        self.disk_cache_enabled = disk_cache_enabled
        self.disk_cache_path = disk_cache_path
        self.intervalos_por_arquivo = carregar_resumos_chbmit(dataset_dir)
        self._cache: dict[tuple[str, int | None, int | None], tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}

    def _disk_cache_file(
        self,
        arquivo: str,
        *,
        max_normal_windows: int | None,
        max_seizure_windows: int | None,
    ) -> Path:
        canais_hash = hashlib.sha1(
            json.dumps(self.canais_referencia or [], ensure_ascii=True).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:10]
        stem = Path(arquivo).stem
        nome = (
            f"{stem}__{self.feature_mode}"
            f"__w{self.window_seconds:g}_s{self.step_seconds:g}"
            f"__seq{self.sequence_length}_stride{self.sequence_stride}"
            f"__normal{max_normal_windows if max_normal_windows is not None else 'all'}"
            f"__seizure{max_seizure_windows if max_seizure_windows is not None else 'all'}"
            f"__ch{canais_hash}.pkl"
        )
        return self.disk_cache_path / nome

    def _load_disk_cache(
        self,
        arquivo: str,
        *,
        max_normal_windows: int | None,
        max_seizure_windows: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]] | None:
        if not self.disk_cache_enabled:
            return None
        cache_file = self._disk_cache_file(
            arquivo,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
        )
        if not cache_file.exists():
            return None
        with cache_file.open("rb") as handle:
            payload = pickle.load(handle)
        return payload["x"], payload["y"], payload["meta"]

    def _save_disk_cache(
        self,
        arquivo: str,
        resultado: tuple[np.ndarray, np.ndarray, list[dict[str, Any]]],
        *,
        max_normal_windows: int | None,
        max_seizure_windows: int | None,
    ) -> None:
        if not self.disk_cache_enabled:
            return
        self.disk_cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = self._disk_cache_file(
            arquivo,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
        )
        tmp_file = cache_file.with_suffix(".tmp")
        x, y, meta = resultado
        with tmp_file.open("wb") as handle:
            pickle.dump({"x": x, "y": y, "meta": meta}, handle)
        tmp_file.replace(cache_file)

    def carregar_arquivo(
        self,
        arquivo: str,
        *,
        max_normal_windows: int | None,
        max_seizure_windows: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        chave = (arquivo, max_normal_windows, max_seizure_windows)
        if chave in self._cache:
            return self._cache[chave]

        cached = self._load_disk_cache(
            arquivo,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
        )
        if cached is not None:
            x_seq, y_seq, _meta_seq = cached
            print(
                f"{arquivo}: {len(y_seq)} sequencias do cache "
                f"(normal={int(np.sum(y_seq == 0))}, crise={int(np.sum(y_seq == 1))})"
                ,
                flush=True,
            )
            self._cache[chave] = cached
            return cached

        path = self.dataset_dir / arquivo
        if not path.exists():
            raise SystemExit(f"EDF nao encontrado: {path}")
        x_win, y_win, meta_win = extrair_dataset_janelado_edf(
            path,
            self.intervalos_por_arquivo.get(arquivo, []),
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
            max_windows_per_class=None,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
            canais_selecionados=self.canais_referencia,
            feature_mode=self.feature_mode,
        )
        if len(y_win) == 0:
            resultado = (
                np.empty((0, self.sequence_length, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
            )
        else:
            resultado = construir_sequencias(
                x_win,
                y_win,
                meta_win,
                sequence_length=self.sequence_length,
                sequence_stride=self.sequence_stride,
            )

        x_seq, y_seq, meta_seq = resultado
        if len(y_seq) > 0:
            print(
                f"{arquivo}: {len(y_seq)} sequencias "
                f"(normal={int(np.sum(y_seq == 0))}, crise={int(np.sum(y_seq == 1))})"
                ,
                flush=True,
            )
        self._save_disk_cache(
            arquivo,
            resultado,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
        )
        self._cache[chave] = resultado
        return resultado

    def carregar_lista(
        self,
        arquivos: list[str],
        *,
        max_normal_windows: int | None,
        max_seizure_windows: int | None,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        for arquivo in arquivos:
            x, y, meta = self.carregar_arquivo(
                arquivo,
                max_normal_windows=max_normal_windows,
                max_seizure_windows=max_seizure_windows,
            )
            if len(y) == 0:
                continue
            x_parts.append(x)
            y_parts.append(y)
            metas.extend(meta)
        if not x_parts:
            raise SystemExit("Nenhuma sequencia gerada.")
        return np.vstack(x_parts), np.concatenate(y_parts), metas


def separar_treino_calibracao(
    por_paciente: dict[str, list[str]],
    paciente_teste: str,
    *,
    n_calibration_patients: int,
) -> tuple[list[str], list[str], list[str]]:
    pacientes_treino = [paciente for paciente in sorted(por_paciente) if paciente != paciente_teste]
    if len(pacientes_treino) < 2:
        raise RuntimeError("A validacao inter-paciente requer ao menos dois pacientes fora do teste.")

    n_calibracao = min(max(1, n_calibration_patients), len(pacientes_treino) - 1)
    pacientes_calibracao = pacientes_treino[-n_calibracao:]
    pacientes_modelo = [paciente for paciente in pacientes_treino if paciente not in pacientes_calibracao]

    train_files = [arquivo for paciente in pacientes_modelo for arquivo in por_paciente[paciente]]
    calibration_files = [arquivo for paciente in pacientes_calibracao for arquivo in por_paciente[paciente]]
    return train_files, calibration_files, pacientes_calibracao


def contar_classes_por_arquivo(metas: list[dict[str, Any]], y: np.ndarray) -> dict[str, dict[str, int]]:
    resultado: dict[str, dict[str, int]] = {}
    for arquivo in sorted({str(meta["arquivo"]) for meta in metas}):
        indices = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo])
        resultado[arquivo] = {
            "0": int(np.sum(y[indices] == 0)),
            "1": int(np.sum(y[indices] == 1)),
        }
    return resultado


def ranking_calibracao(item: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float, float]:
    faltou_alinhar = item["positive_predictions"] - item["positive_predictions_with_overlap"]
    return (
        -float(item["false_positives"]),
        -float(item.get("localized_false_negatives", item["false_negatives"])),
        -float(faltou_alinhar),
        -float(item.get("indeterminate", 0)),
        float(item["positive_overlap_seconds_sum"]),
        float(item.get("localized_metrics", item["metrics"])["f1"]),
        float(item.get("localized_metrics", item["metrics"])["recall"]),
        float(item["metrics"]["f1"]),
        float(item["metrics"]["accuracy"]),
    )


def calibrar_no_treino(
    y_train: np.ndarray,
    scores_train: np.ndarray,
    meta_train: list[dict[str, Any]],
    *,
    thresholds: list[float],
    durations: list[float],
) -> dict[str, Any]:
    resultados = [
        avaliar_combo(
            y_train,
            scores_train,
            meta_train,
            threshold=threshold,
            min_duration_seconds=duration,
        )
        for threshold in thresholds
        for duration in durations
    ]
    resultados_ordenados = sorted(resultados, key=ranking_calibracao, reverse=True)
    return {
        "best": resultados_ordenados[0],
        "top_10": resultados_ordenados[:10],
    }


def resumir_fold(fold: dict[str, Any]) -> dict[str, Any]:
    edf_eval = fold["test"]["edf_eval"]
    files = edf_eval["files"]
    arquivos_crise = [file for file in files if file["label"] == 1]
    arquivos_normais = [file for file in files if file["label"] == 0]
    crises_localizadas = sum(
        1
        for file in arquivos_crise
        if file["pred"] == 1 and file["trecho_suspeito"]["overlap_crise_real_seconds"] > 0
    )
    return {
        "paciente_teste": fold["paciente_teste"],
        "threshold_calibrado": fold["calibration"]["threshold"],
        "duracao_minima_calibrada": fold["calibration"]["min_duration_seconds"],
        "arquivos_teste": len(files),
        "arquivos_crise": len(arquivos_crise),
        "arquivos_normais": len(arquivos_normais),
        "crises_localizadas": crises_localizadas,
        "falsos_positivos": int(edf_eval["false_positives"]),
        "falsos_negativos": int(edf_eval["false_negatives"]),
        "falsos_negativos_localizacao": int(edf_eval.get("localized_false_negatives", edf_eval["false_negatives"])),
        "edf_accuracy": edf_eval["metrics"]["accuracy"],
        "edf_precision": edf_eval["metrics"]["precision"],
        "edf_recall": edf_eval["metrics"]["recall"],
        "edf_f1": edf_eval["metrics"]["f1"],
        "edf_localized_accuracy": edf_eval.get("localized_metrics", edf_eval["metrics"])["accuracy"],
        "edf_localized_precision": edf_eval.get("localized_metrics", edf_eval["metrics"])["precision"],
        "edf_localized_recall": edf_eval.get("localized_metrics", edf_eval["metrics"])["recall"],
        "edf_localized_f1": edf_eval.get("localized_metrics", edf_eval["metrics"])["f1"],
        "window_f1": fold["test"]["window_metrics"]["f1"],
    }


def montar_payload(
    *,
    args: argparse.Namespace,
    settings: Any,
    thresholds: list[float],
    durations: list[float],
    canais_referencia: list[str] | None,
    folds: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    partial: bool,
    running_patient: str | None = None,
) -> dict[str, Any]:
    resumo = [resumir_fold(fold) for fold in folds]
    metric_keys = [
        "edf_accuracy",
        "edf_precision",
        "edf_recall",
        "edf_f1",
        "edf_localized_accuracy",
        "edf_localized_precision",
        "edf_localized_recall",
        "edf_localized_f1",
        "window_f1",
    ]
    mean = (
        {key: float(np.mean([item[key] for item in resumo])) for key in metric_keys}
        if resumo
        else {key: None for key in metric_keys}
    )
    total = {
        "arquivos_crise": int(sum(item["arquivos_crise"] for item in resumo)),
        "arquivos_normais": int(sum(item["arquivos_normais"] for item in resumo)),
        "crises_localizadas": int(sum(item["crises_localizadas"] for item in resumo)),
        "falsos_positivos": int(sum(item["falsos_positivos"] for item in resumo)),
        "falsos_negativos": int(sum(item["falsos_negativos"] for item in resumo)),
        "falsos_negativos_localizacao": int(sum(item["falsos_negativos_localizacao"] for item in resumo)),
    }
    return {
        "experiment": "sequence_cnn_lstm_leave_one_patient_out",
        "partial": partial,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "running_patient": running_patient,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "test_patients": args.test_patients,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "feature_mode": args.feature_mode,
        "canais_referencia": list(canais_referencia or []),
        "max_normal_windows_per_file": args.max_normal_windows_per_file,
        "max_seizure_windows_per_file": args.max_seizure_windows_per_file,
        "calibration_patients_per_fold": args.calibration_patients_per_fold,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "class_weight_mode": args.class_weight_mode,
        "positive_class_weight": args.positive_class_weight,
        "thresholds": thresholds,
        "durations": durations,
        "disk_cache_enabled": bool(settings.ai_sequence_disk_cache_enabled),
        "disk_cache_path": str(settings.ai_sequence_cache_path.resolve()),
        "summary": resumo,
        "mean": mean,
        "total": total,
        "skipped": skipped,
        "folds": folds,
    }


def salvar_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def avaliar_fold(
    *,
    dataset_cache: SequenceDatasetCache,
    dataset_dir: Path,
    paciente_teste: str,
    train_files: list[str],
    calibration_files: list[str],
    calibration_patients: list[str],
    test_files: list[str],
    window_seconds: float,
    step_seconds: float,
    max_normal_windows_per_file: int,
    max_seizure_windows_per_file: int,
    sequence_length: int,
    sequence_stride: int,
    feature_mode: str,
    canais_referencia: list[str] | None,
    epochs: int,
    batch_size: int,
    class_weight_mode: str,
    positive_class_weight: float,
    thresholds: list[float],
    durations: list[float],
    random_state: int,
    fold_idx: int,
) -> dict[str, Any]:
    import tensorflow as tf

    print(f"\n=== Fold paciente teste: {paciente_teste} ===", flush=True)
    print(
        f"Treino: {len(train_files)} EDFs; "
        f"calibracao: {len(calibration_files)} EDFs ({', '.join(calibration_patients)}); "
        f"teste: {', '.join(test_files)}",
        flush=True,
    )

    x_train, y_train, meta_train = dataset_cache.carregar_lista(
        arquivos=train_files,
        max_normal_windows=max_normal_windows_per_file,
        max_seizure_windows=max_seizure_windows_per_file,
    )
    x_test, y_test, meta_test = dataset_cache.carregar_lista(
        arquivos=test_files,
        max_normal_windows=None,
        max_seizure_windows=None,
    )
    x_cal, y_cal, meta_cal = dataset_cache.carregar_lista(
        arquivos=calibration_files,
        max_normal_windows=None,
        max_seizure_windows=None,
    )

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"Treino do fold {paciente_teste} nao tem duas classes.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_state + fold_idx)

    scaler = ajustar_scaler(x_train)
    x_train_scaled = aplicar_scaler(x_train, scaler)
    x_test_scaled = aplicar_scaler(x_test, scaler)

    model = criar_modelo(sequence_length, x_train.shape[2])
    pesos_classe = resolver_class_weight(
        y_train,
        mode=class_weight_mode,
        positive_weight=positive_class_weight,
    )
    history = model.fit(
        x_train_scaled,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=pesos_classe,
        validation_split=0.15,
    )

    x_cal_scaled = aplicar_scaler(x_cal, scaler)
    scores_cal = model.predict(x_cal_scaled, verbose=0).reshape(-1)
    calibracao = calibrar_no_treino(
        y_cal,
        scores_cal,
        meta_cal,
        thresholds=thresholds,
        durations=durations,
    )
    threshold = float(calibracao["best"]["threshold"])
    duration = float(calibracao["best"]["min_duration_seconds"])

    scores_test = model.predict(x_test_scaled, verbose=0).reshape(-1)
    pred_test = (scores_test >= threshold).astype(np.int64)
    edf_eval = avaliar_combo(
        y_test,
        scores_test,
        meta_test,
        threshold=threshold,
        min_duration_seconds=duration,
    )

    fold = {
        "paciente_teste": paciente_teste,
        "train_files": train_files,
        "calibration_files": calibration_files,
        "calibration_patients": calibration_patients,
        "test_files": test_files,
        "calibration": {
            "threshold": threshold,
            "min_duration_seconds": duration,
            "train_best": calibracao["best"],
            "train_top_10": calibracao["top_10"],
        },
        "train": {
            "n_sequences": int(len(y_train)),
            "class_counts": {
                "0": int(np.sum(y_train == 0)),
                "1": int(np.sum(y_train == 1)),
            },
            "class_weight_mode": class_weight_mode,
            "class_weight": (
                {str(k): float(v) for k, v in pesos_classe.items()}
                if pesos_classe is not None
                else None
            ),
            "class_counts_by_file": contar_classes_por_arquivo(meta_train, y_train),
        },
        "calibration_set": {
            "n_sequences": int(len(y_cal)),
            "class_counts": {
                "0": int(np.sum(y_cal == 0)),
                "1": int(np.sum(y_cal == 1)),
            },
            "class_counts_by_file": contar_classes_por_arquivo(meta_cal, y_cal),
        },
        "test": {
            "n_sequences": int(len(y_test)),
            "class_counts": {
                "0": int(np.sum(y_test == 0)),
                "1": int(np.sum(y_test == 1)),
            },
            "class_counts_by_file": contar_classes_por_arquivo(meta_test, y_test),
            "window_metrics": metricas_binarias(y_test, pred_test),
            "edf_eval": edf_eval,
        },
        "training_history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
    }
    resumo = resumir_fold(fold)
    print(
        f"{paciente_teste}: EDF f1={resumo['edf_f1']:.3f}, "
        f"localizado f1={resumo['edf_localized_f1']:.3f}, "
        f"crises localizadas={resumo['crises_localizadas']}/{resumo['arquivos_crise']}, "
        f"FP={resumo['falsos_positivos']}, FN={resumo['falsos_negativos']}, "
        f"FN_loc={resumo['falsos_negativos_localizacao']}, "
        f"thr={threshold}, dur={duration}",
        flush=True,
    )
    return fold


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validacao CNN-LSTM sequencial leave-one-patient-out.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--test-patients", nargs="+", default=["chb01", "chb03", "chb11", "chb13", "chb14", "chb15"])
    parser.add_argument("--window-seconds", type=float, default=settings.ai_sequence_window_seconds)
    parser.add_argument("--step-seconds", type=float, default=settings.ai_sequence_step_seconds)
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
    parser.add_argument("--max-normal-windows-per-file", type=int, default=settings.ai_sequence_max_normal_windows_per_file)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=settings.ai_sequence_max_seizure_windows_per_file)
    parser.add_argument("--calibration-patients-per-fold", type=int, default=settings.ai_sequence_calibration_patients_per_fold)
    parser.add_argument("--epochs", type=int, default=settings.ai_sequence_epochs)
    parser.add_argument("--batch-size", type=int, default=settings.ai_sequence_batch_size)
    parser.add_argument("--class-weight-mode", choices=["balanced", "manual", "none"], default="balanced")
    parser.add_argument("--positive-class-weight", type=float, default=3.0)
    parser.add_argument("--thresholds", default=settings.ai_sequence_thresholds)
    parser.add_argument("--durations", default=settings.ai_sequence_durations)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "modelos" / "sequence_leave_one_patient_eval.json")
    parser.add_argument("--partial-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Retoma folds ja gravados no partial-output.")
    args = parser.parse_args()

    arquivos = carregar_arquivos(args.dataset_dir, args.manifest)
    por_paciente = agrupar_por_paciente(arquivos)
    thresholds = parse_float_list(args.thresholds)
    durations = parse_float_list(args.durations)
    canais_referencia = None
    if args.feature_mode in {FEATURE_MODE_PER_CHANNEL, FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL}:
        ref_path = args.dataset_dir / (args.channel_reference_edf or arquivos[0])
        canais_referencia = list(extrair_metadados_edf(ref_path)["canais_eeg"])
    dataset_cache = SequenceDatasetCache(
        dataset_dir=args.dataset_dir,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        max_normal_windows=args.max_normal_windows_per_file,
        max_seizure_windows=args.max_seizure_windows_per_file,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        feature_mode=args.feature_mode,
        canais_referencia=canais_referencia,
        disk_cache_enabled=settings.ai_sequence_disk_cache_enabled,
        disk_cache_path=settings.ai_sequence_cache_path,
    )

    folds: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    partial_output = args.partial_output or args.output.with_name(f"{args.output.stem}.partial.json")
    if args.resume and partial_output.exists():
        partial_payload = json.loads(partial_output.read_text(encoding="utf-8"))
        folds = list(partial_payload.get("folds", []))
        skipped = list(partial_payload.get("skipped", []))
        print(
            f"Retomando de {partial_output}: {len(folds)} folds concluidos, "
            f"{len(skipped)} ignorados.",
            flush=True,
        )

    for fold_idx, paciente_teste in enumerate(args.test_patients, start=1):
        pacientes_concluidos = {str(fold["paciente_teste"]) for fold in folds}
        pacientes_ignorados = {str(item["paciente"]) for item in skipped}
        if args.resume and paciente_teste in pacientes_concluidos | pacientes_ignorados:
            print(f"{paciente_teste}: ja registrado no partial, pulando.", flush=True)
            continue

        test_files = por_paciente.get(paciente_teste, [])
        if not test_files:
            skipped.append({"paciente": paciente_teste, "motivo": "sem EDF local"})
            salvar_payload(
                montar_payload(
                    args=args,
                    settings=settings,
                    thresholds=thresholds,
                    durations=durations,
                    canais_referencia=canais_referencia,
                    folds=folds,
                    skipped=skipped,
                    partial=True,
                ),
                partial_output,
            )
            continue
        train_files, calibration_files, calibration_patients = separar_treino_calibracao(
            por_paciente,
            paciente_teste,
            n_calibration_patients=args.calibration_patients_per_fold,
        )
        salvar_payload(
            montar_payload(
                args=args,
                settings=settings,
                thresholds=thresholds,
                durations=durations,
                canais_referencia=canais_referencia,
                folds=folds,
                skipped=skipped,
                partial=True,
                running_patient=paciente_teste,
            ),
            partial_output,
        )
        try:
            fold = avaliar_fold(
                dataset_cache=dataset_cache,
                dataset_dir=args.dataset_dir,
                paciente_teste=paciente_teste,
                train_files=train_files,
                calibration_files=calibration_files,
                calibration_patients=calibration_patients,
                test_files=test_files,
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_normal_windows_per_file=args.max_normal_windows_per_file,
                max_seizure_windows_per_file=args.max_seizure_windows_per_file,
                sequence_length=args.sequence_length,
                sequence_stride=args.sequence_stride,
                feature_mode=args.feature_mode,
                canais_referencia=canais_referencia,
                epochs=args.epochs,
                batch_size=args.batch_size,
                class_weight_mode=args.class_weight_mode,
                positive_class_weight=args.positive_class_weight,
                thresholds=thresholds,
                durations=durations,
                random_state=args.random_state,
                fold_idx=fold_idx,
            )
            folds.append(fold)
            salvar_payload(
                montar_payload(
                    args=args,
                    settings=settings,
                    thresholds=thresholds,
                    durations=durations,
                    canais_referencia=canais_referencia,
                    folds=folds,
                    skipped=skipped,
                    partial=True,
                ),
                partial_output,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{paciente_teste}: fold ignorado - {exc}", flush=True)
            skipped.append({"paciente": paciente_teste, "motivo": str(exc)})
            salvar_payload(
                montar_payload(
                    args=args,
                    settings=settings,
                    thresholds=thresholds,
                    durations=durations,
                    canais_referencia=canais_referencia,
                    folds=folds,
                    skipped=skipped,
                    partial=True,
                ),
                partial_output,
            )

    if not folds:
        raise SystemExit("Nenhum fold inter-paciente foi executado.")

    payload = montar_payload(
        args=args,
        settings=settings,
        thresholds=thresholds,
        durations=durations,
        canais_referencia=canais_referencia,
        folds=folds,
        skipped=skipped,
        partial=False,
    )

    salvar_payload(payload, args.output)
    print(json.dumps({"summary": payload["summary"], "mean": payload["mean"], "total": payload["total"], "skipped": skipped}, indent=2, ensure_ascii=False), flush=True)
    print(f"Metricas salvas em: {args.output}", flush=True)


if __name__ == "__main__":
    main()
