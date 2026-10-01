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

from scripts.calibrate_sequence_cnn_lstm import (  # noqa: E402
    avaliar_combo,
    intervalos_crise_reais,
    listar_runs,
    melhor_overlap_com_crise,
    metricas,
    parse_float_list,
)
from scripts.train_sequence_cnn_lstm import (  # noqa: E402
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    FEATURE_MODE_RAW_SIGNAL,
    FEATURE_MODE_TIME_FREQUENCY,
    FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    ajustar_scaler,
    aplicar_scaler,
    construir_sequencias,
    criar_modelo,
    criar_modelo_segmentacao,
    metricas_binarias,
    paciente_do_arquivo,
    resolver_class_weight,
    resolver_sample_weights,
)
from app.ai_engine.feature_extractor import (  # noqa: E402
    extrair_metadados_edf,
    normalizar_matriz_features_robusta,
)
from app.ai_engine.training import (  # noqa: E402
    carregar_resumos_chbmit,
    extrair_dataset_janelado_edf,
    limitar_dataset_janelado,
)
from app.config import get_settings  # noqa: E402
from app.ai_engine.sequence_segmentation import (  # noqa: E402
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.ai_engine.candidate_confirmer import (  # noqa: E402
    RANKING_POLICIES,
    calibrar_confirmador_oof,
    candidate_feature_vector,
    candidate_ranking_score,
    selecionar_operacao_froc,
)


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
        sequence_target_mode: str,
        sampling_level: str,
        min_window_ictal_overlap_ratio: float,
        feature_normalization: str,
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
        self.sequence_target_mode = sequence_target_mode
        if sampling_level not in {"window", "sequence"}:
            raise ValueError("sampling_level deve ser 'window' ou 'sequence'.")
        self.sampling_level = sampling_level
        self.min_window_ictal_overlap_ratio = min_window_ictal_overlap_ratio
        self.feature_normalization = feature_normalization
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
        sampling_cache_key = (
            "all"
            if max_normal_windows is None and max_seizure_windows is None
            else self.sampling_level
        )
        nome = (
            f"{stem}__{self.feature_mode}"
            f"__w{self.window_seconds:g}_s{self.step_seconds:g}"
            f"__seq{self.sequence_length}_stride{self.sequence_stride}"
            f"__target{self.sequence_target_mode}_overlap{self.min_window_ictal_overlap_ratio:g}"
            f"__sample{sampling_cache_key}"
            f"__norm{self.feature_normalization}"
            "__refallv4"
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
            # Caches completos gerados antes da chave `all` independem do nivel
            # de amostragem e podem ser reutilizados com seguranca.
            legacy_full_cache = Path(str(cache_file).replace("__sampleall", "__samplewindow"))
            if legacy_full_cache.exists():
                cache_file = legacy_full_cache
            elif (
                self.sampling_level == "sequence"
                and max_normal_windows is not None
            ):
                limites_maiores = sorted(
                    {
                        item
                        for item in (self.max_normal_windows, 64, 96, 128, 192, 256)
                        if item is not None and item > max_normal_windows
                    }
                )
                larger_cache = next(
                    (
                        candidate
                        for limit in limites_maiores
                        if (
                            candidate := self._disk_cache_file(
                                arquivo,
                                max_normal_windows=limit,
                                max_seizure_windows=self.max_seizure_windows,
                            )
                        ).exists()
                    ),
                    None,
                )
                if larger_cache is None:
                    return None
                with larger_cache.open("rb") as handle:
                    payload = pickle.load(handle)
                return limitar_dataset_janelado(
                    payload["x"],
                    payload["y"],
                    payload["meta"],
                    max_normal_windows=max_normal_windows,
                    max_seizure_windows=max_seizure_windows,
                    min_contiguous_windows=1,
                )
            else:
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
        normalizacao_edf_completo = self.feature_normalization == "per_edf_robust"
        amostragem_por_sequencia = self.sampling_level == "sequence"
        x_win, y_win, meta_win = extrair_dataset_janelado_edf(
            path,
            self.intervalos_por_arquivo.get(arquivo, []),
            window_seconds=self.window_seconds,
            step_seconds=self.step_seconds,
            max_windows_per_class=None,
            max_normal_windows=None if normalizacao_edf_completo or amostragem_por_sequencia else max_normal_windows,
            max_seizure_windows=None if normalizacao_edf_completo or amostragem_por_sequencia else max_seizure_windows,
            canais_selecionados=self.canais_referencia,
            feature_mode=self.feature_mode,
            min_ictal_overlap_ratio=self.min_window_ictal_overlap_ratio,
            min_contiguous_windows=self.sequence_length,
        )
        if len(y_win) == 0:
            resultado = (
                np.empty((0, self.sequence_length, 0), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
                [],
            )
        else:
            if normalizacao_edf_completo:
                x_win = normalizar_matriz_features_robusta(x_win)
            if normalizacao_edf_completo and not amostragem_por_sequencia:
                x_win, y_win, meta_win = limitar_dataset_janelado(
                    x_win,
                    y_win,
                    meta_win,
                    max_normal_windows=max_normal_windows,
                    max_seizure_windows=max_seizure_windows,
                    min_contiguous_windows=self.sequence_length,
                )
            resultado = construir_sequencias(
                x_win,
                y_win,
                meta_win,
                sequence_length=self.sequence_length,
                sequence_stride=self.sequence_stride,
                target_mode=self.sequence_target_mode,
            )
            if amostragem_por_sequencia:
                x_seq, y_seq, meta_seq = resultado
                resultado = limitar_dataset_janelado(
                    x_seq,
                    y_seq,
                    meta_seq,
                    max_normal_windows=max_normal_windows,
                    max_seizure_windows=max_seizure_windows,
                    min_contiguous_windows=1,
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
    reserved_test_patients: set[str] | None = None,
    fixed_calibration_patients: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    reservados = set(reserved_test_patients or ()) | {paciente_teste}
    pacientes_treino = [paciente for paciente in sorted(por_paciente) if paciente not in reservados]
    if len(pacientes_treino) < 2:
        raise RuntimeError("A validacao inter-paciente requer ao menos dois pacientes fora do teste.")

    if fixed_calibration_patients:
        pacientes_calibracao = list(dict.fromkeys(fixed_calibration_patients))
        invalidos = [paciente for paciente in pacientes_calibracao if paciente not in pacientes_treino]
        if invalidos:
            raise RuntimeError(f"Pacientes de calibracao invalidos: {', '.join(invalidos)}")
        if len(pacientes_calibracao) >= len(pacientes_treino):
            raise RuntimeError("A calibracao fixa nao pode consumir todos os pacientes de treino.")
    else:
        n_calibracao = min(max(1, n_calibration_patients), len(pacientes_treino) - 1)
        pacientes_calibracao = pacientes_treino[-n_calibracao:]
    pacientes_modelo = [paciente for paciente in pacientes_treino if paciente not in pacientes_calibracao]

    train_files = [arquivo for paciente in pacientes_modelo for arquivo in por_paciente[paciente]]
    calibration_files = [arquivo for paciente in pacientes_calibracao for arquivo in por_paciente[paciente]]
    return train_files, calibration_files, pacientes_calibracao


def validar_isolamento_fold(
    *,
    train_files: list[str],
    calibration_files: list[str],
    test_files: list[str],
    reserved_test_patients: set[str],
) -> None:
    """Interrompe a avaliacao se qualquer paciente atravessar particoes."""
    train_patients = {paciente_do_arquivo(item) for item in train_files}
    calibration_patients = {paciente_do_arquivo(item) for item in calibration_files}
    test_patients = {paciente_do_arquivo(item) for item in test_files}
    conflitos = {
        "treino_calibracao": sorted(train_patients & calibration_patients),
        "treino_teste": sorted(train_patients & test_patients),
        "calibracao_teste": sorted(calibration_patients & test_patients),
        "reservados_no_treino": sorted(train_patients & reserved_test_patients),
        "reservados_na_calibracao": sorted(calibration_patients & reserved_test_patients),
    }
    invalidos = {key: value for key, value in conflitos.items() if value}
    if invalidos:
        detalhes = "; ".join(f"{key}={','.join(value)}" for key, value in invalidos.items())
        raise RuntimeError(f"Vazamento de paciente detectado: {detalhes}")


def contar_classes_por_arquivo(metas: list[dict[str, Any]], y: np.ndarray) -> dict[str, dict[str, int]]:
    resultado: dict[str, dict[str, int]] = {}
    for arquivo in sorted({str(meta["arquivo"]) for meta in metas}):
        indices = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo])
        resultado[arquivo] = {
            "0": int(np.sum(y[indices] == 0)),
            "1": int(np.sum(y[indices] == 1)),
        }
    return resultado


def extrair_candidatos_confirmador(
    y_true: np.ndarray,
    scores: np.ndarray,
    metas: list[dict[str, Any]],
    *,
    threshold: float,
    max_gap_seconds: float,
    min_overlap_ratio: float,
    max_candidates_per_file: int,
) -> list[dict[str, Any]]:
    """Converte runs sensiveis em exemplos positivos e hard negatives."""
    candidatos: list[dict[str, Any]] = []
    for arquivo in sorted({str(meta["arquivo"]) for meta in metas}):
        indices = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo])
        runs = listar_runs(
            scores,
            metas,
            indices,
            threshold,
            min_duration_seconds=0.0,
            max_gap_seconds=max_gap_seconds,
        )[:max_candidates_per_file]
        segmentos = intervalos_crise_reais(y_true, metas, indices)
        file_scores = np.asarray(scores[indices], dtype=np.float32)
        posicao_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(indices)}
        file_duration = float(metas[int(indices[-1])]["end_seconds"]) - float(
            metas[int(indices[0])]["start_seconds"]
        )
        for rank, run in enumerate(runs, start=1):
            local_indices = np.asarray(
                [posicao_local[int(global_idx)] for global_idx in run["indices"]],
                dtype=np.int64,
            )
            run_local = {**run, "local_indices": local_indices}
            overlap_seconds, overlap_ratio = melhor_overlap_com_crise(run, segmentos)
            candidatos.append(
                {
                    "arquivo": arquivo,
                    "paciente": paciente_do_arquivo(arquivo),
                    "rank_stage1": rank,
                    "start_seconds": float(run["start_seconds"]),
                    "end_seconds": float(run["end_seconds"]),
                    "duration_seconds": float(run["duration_seconds"]),
                    "score_mean": float(run["score_mean"]),
                    "score_max": float(run["score_max"]),
                    "overlap_seconds": overlap_seconds,
                    "overlap_ratio": overlap_ratio,
                    "label": int(overlap_seconds >= 1.0 and overlap_ratio >= min_overlap_ratio),
                    "segments": segmentos,
                    "features": candidate_feature_vector(
                        run_local,
                        file_scores=file_scores,
                        file_duration_seconds=file_duration,
                        n_runs_file=len(runs),
                        generation_threshold=threshold,
                    ),
                    "file_duration_seconds": file_duration,
                }
            )
    return candidatos


def avaliar_confirmador(
    candidatos: list[dict[str, Any]],
    probabilities: np.ndarray,
    *,
    threshold: float,
    y_true: np.ndarray,
    metas: list[dict[str, Any]],
    ranking_policy: str = "confirmation",
) -> dict[str, Any]:
    """Avalia candidatos confirmados por evento, EDF, localizacao e FA/h."""
    probabilities = np.asarray(probabilities, dtype=float)
    confirmados = [
        {**item, "confirmation_score": float(score)}
        for item, score in zip(candidatos, probabilities)
        if score >= threshold
    ]
    y_true_file: list[int] = []
    y_pred_file: list[int] = []
    y_pred_localizado: list[int] = []
    eventos_total = 0
    eventos_detectados = 0
    falsos_alarmes = 0
    horas_nao_ictais = 0.0
    files: list[dict[str, Any]] = []
    for arquivo in sorted({str(meta["arquivo"]) for meta in metas}):
        indices_arquivo = np.asarray([idx for idx, meta in enumerate(metas) if meta["arquivo"] == arquivo])
        todos_arquivo = [item for item in candidatos if item["arquivo"] == arquivo]
        confirmados_arquivo = [item for item in confirmados if item["arquivo"] == arquivo]
        segmentos = intervalos_crise_reais(y_true, metas, indices_arquivo)
        label = int(bool(segmentos))
        pred = int(bool(confirmados_arquivo))
        max_duration_file = max(
            (float(item["duration_seconds"]) for item in confirmados_arquivo),
            default=1.0,
        )
        principal = max(
            confirmados_arquivo,
            key=lambda item: (
                candidate_ranking_score(
                    item,
                    policy=ranking_policy,
                    max_duration_seconds=max_duration_file,
                ),
                item["confirmation_score"],
            ),
            default=None,
        )
        principal_localizado = bool(principal and principal["label"] == 1)
        detectados_arquivo = sum(
            1
            for segmento in segmentos
            if any(melhor_overlap_com_crise(item, [segmento])[0] >= 1.0 for item in confirmados_arquivo)
        )
        falsos_arquivo = sum(int(item["label"] == 0) for item in confirmados_arquivo)
        duracao = float(metas[int(indices_arquivo[-1])]["end_seconds"]) - float(
            metas[int(indices_arquivo[0])]["start_seconds"]
        )
        duracao_ictal = sum(
            max(0.0, float(item["end_seconds"]) - float(item["start_seconds"]))
            for item in segmentos
        )
        horas_nao_ictais += max(0.0, duracao - duracao_ictal) / 3600.0
        eventos_total += len(segmentos)
        eventos_detectados += detectados_arquivo
        falsos_alarmes += falsos_arquivo
        y_true_file.append(label)
        y_pred_file.append(pred)
        y_pred_localizado.append(int(pred and (label == 0 or principal_localizado)))
        files.append(
            {
                "arquivo": arquivo,
                "label": label,
                "pred": pred,
                "principal_localizado": principal_localizado,
                "eventos_detectados": detectados_arquivo,
                "eventos_total": len(segmentos),
                "falsos_alarmes": falsos_arquivo,
                "principal": (
                    {
                        key: principal[key]
                        for key in (
                            "start_seconds",
                            "end_seconds",
                            "duration_seconds",
                            "score_mean",
                            "score_max",
                            "confirmation_score",
                            "overlap_seconds",
                            "overlap_ratio",
                        )
                    }
                    | {
                        "ranking_score": candidate_ranking_score(
                            principal,
                            policy=ranking_policy,
                            max_duration_seconds=max_duration_file,
                        )
                    }
                    if principal
                    else None
                ),
            }
        )
    event_precision = eventos_detectados / (eventos_detectados + falsos_alarmes) if eventos_detectados + falsos_alarmes else 0.0
    event_recall = eventos_detectados / eventos_total if eventos_total else 0.0
    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if event_precision + event_recall else 0.0
    return {
        "threshold": float(threshold),
        "ranking_policy": ranking_policy,
        "metrics": metricas(y_true_file, y_pred_file),
        "localized_metrics": metricas(y_true_file, y_pred_localizado),
        "event_precision": float(event_precision),
        "event_sensitivity": float(event_recall),
        "event_f1": float(event_f1),
        "detected_seizure_events": int(eventos_detectados),
        "total_seizure_events": int(eventos_total),
        "false_alarm_events": int(falsos_alarmes),
        "normal_hours": float(horas_nao_ictais),
        "false_alarms_per_hour": float(falsos_alarmes / horas_nao_ictais) if horas_nao_ictais else 0.0,
        "files": files,
    }


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


def ranking_calibracao_balanceada(
    item: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    faltou_alinhar = item["positive_predictions"] - item["positive_predictions_with_overlap"]
    return (
        float(item.get("localized_metrics", item["metrics"])["f1"]),
        float(item.get("localized_metrics", item["metrics"])["recall"]),
        float(item["metrics"]["f1"]),
        -float(faltou_alinhar),
        -float(item["false_positives"]),
        -float(item.get("localized_false_negatives", item["false_negatives"])),
        -float(item.get("indeterminate", 0)),
        float(item["positive_overlap_seconds_sum"]),
        float(item["metrics"]["accuracy"]),
    )


def ranking_calibracao_eventos(
    item: dict[str, Any],
) -> tuple[float, float, float, float, float, float, float, float, float]:
    return (
        float(item.get("event_f1", 0.0)),
        float(item.get("event_sensitivity", 0.0)),
        float(item.get("localized_metrics", item["metrics"])["f1"]),
        -float(item.get("false_alarms_per_hour", 0.0)),
        float(item["metrics"]["f1"]),
        -float(item["false_positives"]),
        -float(item.get("localized_false_negatives", item["false_negatives"])),
        float(item.get("positive_overlap_seconds_sum", 0.0)),
        float(item["metrics"]["accuracy"]),
    )


def calibrar_no_treino(
    y_train: np.ndarray,
    scores_train: np.ndarray,
    meta_train: list[dict[str, Any]],
    *,
    thresholds: list[float],
    durations: list[float],
    max_gap_values: list[float],
    hysteresis_ratios: list[float],
    min_overlap_ratio: float,
    selection_mode: str = "balanced",
) -> dict[str, Any]:
    resultados = [
        avaliar_combo(
            y_train,
            scores_train,
            meta_train,
            threshold=threshold,
            min_duration_seconds=duration,
            min_overlap_ratio=min_overlap_ratio,
            max_gap_seconds=max_gap_seconds,
            hysteresis_ratio=hysteresis_ratio,
        )
        for threshold in thresholds
        for duration in durations
        for max_gap_seconds in max_gap_values
        for hysteresis_ratio in hysteresis_ratios
    ]
    if selection_mode == "event_balanced":
        ranking = ranking_calibracao_eventos
    elif selection_mode == "balanced":
        ranking = ranking_calibracao_balanceada
    else:
        ranking = ranking_calibracao
    resultados_ordenados = sorted(resultados, key=ranking, reverse=True)
    return {
        "selection_mode": selection_mode,
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
    confirmadores = [
        fold["test"]["candidate_confirmer"]["test"]
        for fold in folds
        if fold["test"].get("candidate_confirmer") is not None
    ]
    candidate_confirmer_summary = None
    if confirmadores:
        normal_hours = float(sum(item["normal_hours"] for item in confirmadores))
        false_alarms = int(sum(item["false_alarm_events"] for item in confirmadores))
        candidate_confirmer_summary = {
            "mean_edf_precision": float(np.mean([item["metrics"]["precision"] for item in confirmadores])),
            "mean_edf_recall": float(np.mean([item["metrics"]["recall"] for item in confirmadores])),
            "mean_edf_f1": float(np.mean([item["metrics"]["f1"] for item in confirmadores])),
            "mean_localized_f1": float(
                np.mean([item["localized_metrics"]["f1"] for item in confirmadores])
            ),
            "detected_seizure_events": int(
                sum(item["detected_seizure_events"] for item in confirmadores)
            ),
            "total_seizure_events": int(
                sum(item["total_seizure_events"] for item in confirmadores)
            ),
            "correct_primary_candidates": int(
                sum(
                    int(file["principal_localizado"])
                    for item in confirmadores
                    for file in item["files"]
                    if file["label"] == 1
                )
            ),
            "false_alarm_events": false_alarms,
            "normal_hours": normal_hours,
            "false_alarms_per_hour": false_alarms / normal_hours if normal_hours else 0.0,
        }
    return {
        "experiment": "sequence_cnn_lstm_leave_one_patient_out",
        "partial": partial,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "running_patient": running_patient,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest": str(args.manifest.resolve()) if args.manifest else None,
        "test_patients": args.test_patients,
        "reserved_test_patients": args.reserved_test_patients,
        "fixed_calibration_patients": args.fixed_calibration_patients,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sequence_length": args.sequence_length,
        "sequence_stride": args.sequence_stride,
        "sequence_target_mode": args.sequence_target_mode,
        "sampling_level": args.sampling_level,
        "output_mode": args.output_mode,
        "min_window_ictal_overlap_ratio": args.min_window_ictal_overlap_ratio,
        "boundary_sample_weight": args.boundary_sample_weight,
        "event_balanced_weights": args.event_balanced_weights,
        "feature_normalization": args.feature_normalization,
        "feature_mode": args.feature_mode,
        "canais_referencia": list(canais_referencia or []),
        "max_normal_windows_per_file": args.max_normal_windows_per_file,
        "max_seizure_windows_per_file": args.max_seizure_windows_per_file,
        "calibration_patients_per_fold": args.calibration_patients_per_fold,
        "calibration_selection_mode": args.calibration_selection_mode,
        "epochs": args.epochs,
        "ensemble_size": args.ensemble_size,
        "batch_size": args.batch_size,
        "class_weight_mode": args.class_weight_mode,
        "positive_class_weight": args.positive_class_weight,
        "loss": args.loss,
        "focal_alpha": args.focal_alpha if args.loss == "focal" else None,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "early_stopping_patience": args.early_stopping_patience,
        "min_overlap_ratio": args.min_overlap_ratio,
        "thresholds": thresholds,
        "durations": durations,
        "max_gap_values": args.max_gap_values,
        "hysteresis_ratios": args.hysteresis_ratios,
        "candidate_confirmer": args.candidate_confirmer,
        "candidate_generation_threshold": args.candidate_generation_threshold,
        "candidate_generation_gap_seconds": args.candidate_generation_gap_seconds,
        "candidate_max_per_file": args.candidate_max_per_file,
        "candidate_confirmation_thresholds": args.candidate_confirmation_thresholds,
        "candidate_max_false_alarms_per_hour": args.candidate_max_false_alarms_per_hour,
        "random_state": args.random_state,
        "disk_cache_enabled": bool(settings.ai_sequence_disk_cache_enabled),
        "disk_cache_path": str(settings.ai_sequence_cache_path.resolve()),
        "summary": resumo,
        "mean": mean,
        "total": total,
        "candidate_confirmer_summary": candidate_confirmer_summary,
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
    output_mode: str,
    feature_mode: str,
    canais_referencia: list[str] | None,
    epochs: int,
    ensemble_size: int,
    batch_size: int,
    class_weight_mode: str,
    positive_class_weight: float,
    boundary_sample_weight: float,
    event_balanced_weights: bool,
    loss_mode: str,
    focal_alpha: float,
    focal_gamma: float,
    early_stopping_patience: int,
    min_overlap_ratio: float,
    calibration_selection_mode: str,
    thresholds: list[float],
    durations: list[float],
    max_gap_values: list[float],
    hysteresis_ratios: list[float],
    candidate_confirmer: bool,
    candidate_generation_threshold: float,
    candidate_generation_gap_seconds: float,
    candidate_max_per_file: int,
    candidate_confirmation_thresholds: list[float],
    candidate_max_false_alarms_per_hour: float,
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

    segmentacao = output_mode == "segmentation"
    if segmentacao:
        y_train_temporal, meta_train_temporal = construir_alvos_temporais(
            meta_train,
            dataset_cache.intervalos_por_arquivo,
            sequence_length=sequence_length,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            min_ictal_overlap_ratio=dataset_cache.min_window_ictal_overlap_ratio,
        )
        y_cal_temporal, meta_cal_temporal = construir_alvos_temporais(
            meta_cal,
            dataset_cache.intervalos_por_arquivo,
            sequence_length=sequence_length,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            min_ictal_overlap_ratio=dataset_cache.min_window_ictal_overlap_ratio,
        )
        y_test_temporal, meta_test_temporal = construir_alvos_temporais(
            meta_test,
            dataset_cache.intervalos_por_arquivo,
            sequence_length=sequence_length,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            min_ictal_overlap_ratio=dataset_cache.min_window_ictal_overlap_ratio,
        )
        y_train_eval, _scores_train_dummy, meta_train_eval = agregar_predicoes_temporais(
            np.zeros_like(y_train_temporal), y_train_temporal, meta_train_temporal
        )
        y_train_fit: np.ndarray = y_train_temporal
        y_cal_fit: np.ndarray = y_cal_temporal
    else:
        y_train_eval = y_train
        meta_train_eval = meta_train
        y_train_fit = y_train
        y_cal_fit = y_cal

    if len(np.unique(y_train_eval)) < 2:
        raise RuntimeError(f"Treino do fold {paciente_teste} nao tem duas classes.")

    scaler = ajustar_scaler(x_train)
    x_train_scaled = aplicar_scaler(x_train, scaler)
    x_test_scaled = aplicar_scaler(x_test, scaler)
    x_cal_scaled = aplicar_scaler(x_cal, scaler)

    pesos_classe = resolver_class_weight(
        y_train_eval,
        mode=class_weight_mode,
        positive_weight=positive_class_weight,
    )
    if segmentacao:
        pesos_amostras = pesos_temporais(
            y_train_temporal,
            meta_train_temporal,
            boundary_weight=boundary_sample_weight,
            class_weight=None if loss_mode == "focal" else pesos_classe,
            balance_events=event_balanced_weights,
        )
    else:
        pesos_amostras = resolver_sample_weights(
            meta_train,
            boundary_weight=boundary_sample_weight,
            y=y_train,
            class_weight=None if loss_mode == "focal" else pesos_classe,
        )
    criar = criar_modelo_segmentacao if segmentacao else criar_modelo
    scores_cal_membros: list[np.ndarray] = []
    scores_test_membros: list[np.ndarray] = []
    training_histories: list[dict[str, list[float]]] = []
    for member_idx in range(max(1, ensemble_size)):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state + fold_idx + member_idx * 1000)
        model = criar(
            sequence_length,
            x_train.shape[2],
            loss_mode=loss_mode,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        callbacks = []
        if early_stopping_patience > 0:
            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_pr_auc",
                    mode="max",
                    patience=early_stopping_patience,
                    min_delta=0.001,
                    restore_best_weights=True,
                )
            )
        history = model.fit(
            x_train_scaled,
            y_train_fit,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            sample_weight=pesos_amostras,
            validation_data=(x_cal_scaled, y_cal_fit),
            callbacks=callbacks,
        )
        scores_cal_membros.append(np.asarray(model.predict(x_cal_scaled, verbose=0)))
        scores_test_membros.append(np.asarray(model.predict(x_test_scaled, verbose=0)))
        training_histories.append(
            {key: [float(value) for value in values] for key, values in history.history.items()}
        )
    scores_cal_raw = np.mean(np.stack(scores_cal_membros), axis=0)
    scores_test_raw = np.mean(np.stack(scores_test_membros), axis=0)
    if segmentacao:
        y_cal_eval, scores_cal, meta_cal_eval = agregar_predicoes_temporais(
            scores_cal_raw,
            y_cal_temporal,
            meta_cal_temporal,
        )
    else:
        y_cal_eval = y_cal
        scores_cal = scores_cal_raw.reshape(-1)
        meta_cal_eval = meta_cal
    calibracao = calibrar_no_treino(
        y_cal_eval,
        scores_cal,
        meta_cal_eval,
        thresholds=thresholds,
        durations=durations,
        max_gap_values=max_gap_values,
        hysteresis_ratios=hysteresis_ratios,
        min_overlap_ratio=min_overlap_ratio,
        selection_mode=calibration_selection_mode,
    )
    threshold = float(calibracao["best"]["threshold"])
    duration = float(calibracao["best"]["min_duration_seconds"])
    max_gap = float(calibracao["best"].get("max_gap_seconds", 0.0))
    hysteresis_ratio = float(calibracao["best"].get("hysteresis_ratio", 1.0))

    if segmentacao:
        y_test_eval, scores_test, meta_test_eval = agregar_predicoes_temporais(
            scores_test_raw,
            y_test_temporal,
            meta_test_temporal,
        )
    else:
        y_test_eval = y_test
        scores_test = scores_test_raw.reshape(-1)
        meta_test_eval = meta_test
    pred_test = (scores_test >= threshold).astype(np.int64)
    edf_eval = avaliar_combo(
        y_test_eval,
        scores_test,
        meta_test_eval,
        threshold=threshold,
        min_duration_seconds=duration,
        min_overlap_ratio=min_overlap_ratio,
        max_gap_seconds=max_gap,
        hysteresis_ratio=hysteresis_ratio,
    )

    confirmer_eval: dict[str, Any] | None = None
    if candidate_confirmer:
        candidatos_cal = extrair_candidatos_confirmador(
            y_cal_eval,
            scores_cal,
            meta_cal_eval,
            threshold=candidate_generation_threshold,
            max_gap_seconds=candidate_generation_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
            max_candidates_per_file=candidate_max_per_file,
        )
        x_confirmador = np.vstack([item["features"] for item in candidatos_cal])
        y_confirmador = np.asarray([item["label"] for item in candidatos_cal], dtype=np.int64)
        grupos_confirmador = np.asarray([item["paciente"] for item in candidatos_cal])
        pacientes_candidatos = {str(item) for item in grupos_confirmador}
        if pacientes_candidatos != set(calibration_patients):
            raise RuntimeError(
                "Candidatos do confirmador nao correspondem exclusivamente aos pacientes de calibracao."
            )
        if paciente_teste in pacientes_candidatos:
            raise RuntimeError("Vazamento: paciente de teste entrou no confirmador.")
        confirmador = calibrar_confirmador_oof(
            x_confirmador,
            y_confirmador,
            grupos_confirmador,
            thresholds=candidate_confirmation_thresholds,
            random_state=random_state + fold_idx * 100,
        )
        froc_calibracao = [
            avaliar_confirmador(
                candidatos_cal,
                confirmador["oof_scores"],
                threshold=threshold_item,
                y_true=y_cal_eval,
                metas=meta_cal_eval,
                ranking_policy=ranking_policy,
            )
            for threshold_item in candidate_confirmation_thresholds
            for ranking_policy in RANKING_POLICIES
        ]
        limites_froc = (0.50, 0.75, 1.00, 1.25, 1.50)
        pontos_operacao = [
            selecionar_operacao_froc(
                froc_calibracao,
                max_false_alarms_per_hour=limite,
            )
            for limite in limites_froc
        ]
        operacao_principal = selecionar_operacao_froc(
            froc_calibracao,
            max_false_alarms_per_hour=candidate_max_false_alarms_per_hour,
        )
        melhor_calibracao = operacao_principal["selected"]
        confirmador["threshold"] = float(melhor_calibracao["threshold"])
        confirmador["ranking_policy"] = str(melhor_calibracao["ranking_policy"])
        confirmador["froc_constraint_satisfied"] = bool(
            operacao_principal["constraint_satisfied"]
        )
        confirmador["max_false_alarms_per_hour"] = float(candidate_max_false_alarms_per_hour)
        confirmador["best_event_oof"] = {
            key: melhor_calibracao[key]
            for key in (
                "threshold",
                "ranking_policy",
                "event_sensitivity",
                "event_precision",
                "event_f1",
                "false_alarms_per_hour",
            )
        }
        confirmador["hard_negative_source"] = "calibration_only"
        confirmador["candidate_patients"] = sorted(pacientes_candidatos)
        candidatos_test = extrair_candidatos_confirmador(
            y_test_eval,
            scores_test,
            meta_test_eval,
            threshold=candidate_generation_threshold,
            max_gap_seconds=candidate_generation_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
            max_candidates_per_file=candidate_max_per_file,
        )
        x_candidatos_test = np.vstack([item["features"] for item in candidatos_test])
        probabilities = confirmador["model"].predict_proba(x_candidatos_test)[:, 1]
        threshold_confirmador = float(confirmador["threshold"])
        ranking_policy = str(confirmador["ranking_policy"])
        resultado_confirmador = avaliar_confirmador(
            candidatos_test,
            probabilities,
            threshold=threshold_confirmador,
            y_true=y_test_eval,
            metas=meta_test_eval,
            ranking_policy=ranking_policy,
        )
        froc = [
            avaliar_confirmador(
                candidatos_test,
                probabilities,
                threshold=item,
                y_true=y_test_eval,
                metas=meta_test_eval,
                ranking_policy=ranking_policy,
            )
            for item in candidate_confirmation_thresholds
        ]
        confirmer_eval = {
            "candidate_generation_threshold": candidate_generation_threshold,
            "candidate_generation_gap_seconds": candidate_generation_gap_seconds,
            "max_candidates_per_file": candidate_max_per_file,
            "selection_source": "calibration_oof_only",
            "max_false_alarms_per_hour": candidate_max_false_alarms_per_hour,
            "calibration_operating_points": [
                {
                    "max_false_alarms_per_hour": item["max_false_alarms_per_hour"],
                    "constraint_satisfied": item["constraint_satisfied"],
                    "selected": {
                        key: item["selected"][key]
                        for key in (
                            "threshold",
                            "ranking_policy",
                            "event_sensitivity",
                            "event_precision",
                            "event_f1",
                            "false_alarms_per_hour",
                        )
                    },
                }
                for item in pontos_operacao
            ],
            "calibration": {
                key: value
                for key, value in confirmador.items()
                if key not in {"model", "oof_scores"}
            },
            "test": resultado_confirmador,
            "froc": [
                {
                    key: item[key]
                    for key in (
                        "threshold",
                        "event_sensitivity",
                        "event_precision",
                        "event_f1",
                        "false_alarms_per_hour",
                    )
                }
                for item in froc
            ],
        }

    fold = {
        "paciente_teste": paciente_teste,
        "random_state_base": random_state,
        "random_seeds": [
            random_state + fold_idx + member_idx * 1000
            for member_idx in range(max(1, ensemble_size))
        ],
        "train_files": train_files,
        "calibration_files": calibration_files,
        "calibration_patients": calibration_patients,
        "test_files": test_files,
        "output_mode": output_mode,
        "calibration": {
            "selection_mode": calibration_selection_mode,
            "threshold": threshold,
            "min_duration_seconds": duration,
            "max_gap_seconds": max_gap,
            "hysteresis_ratio": hysteresis_ratio,
            "train_best": calibracao["best"],
            "train_top_10": calibracao["top_10"],
        },
        "train": {
            "n_sequences": int(len(x_train)),
            "n_targets": int(len(y_train_eval)),
            "class_counts": {
                "0": int(np.sum(y_train_eval == 0)),
                "1": int(np.sum(y_train_eval == 1)),
            },
            "class_weight_mode": class_weight_mode,
            "loss": loss_mode,
            "early_stopping_patience": early_stopping_patience,
            "ensemble_size": max(1, ensemble_size),
            "class_weight": (
                {str(k): float(v) for k, v in pesos_classe.items()}
                if pesos_classe is not None
                else None
            ),
            "class_counts_by_file": contar_classes_por_arquivo(meta_train_eval, y_train_eval),
        },
        "calibration_set": {
            "n_sequences": int(len(x_cal)),
            "n_targets": int(len(y_cal_eval)),
            "class_counts": {
                "0": int(np.sum(y_cal_eval == 0)),
                "1": int(np.sum(y_cal_eval == 1)),
            },
            "class_counts_by_file": contar_classes_por_arquivo(meta_cal_eval, y_cal_eval),
        },
        "test": {
            "n_sequences": int(len(x_test)),
            "n_targets": int(len(y_test_eval)),
            "class_counts": {
                "0": int(np.sum(y_test_eval == 0)),
                "1": int(np.sum(y_test_eval == 1)),
            },
            "class_counts_by_file": contar_classes_por_arquivo(meta_test_eval, y_test_eval),
            "window_metrics": metricas_binarias(y_test_eval, pred_test),
            "edf_eval": edf_eval,
            "candidate_confirmer": confirmer_eval,
        },
        "training_histories": training_histories,
    }
    resumo = resumir_fold(fold)
    print(
        f"{paciente_teste}: EDF f1={resumo['edf_f1']:.3f}, "
        f"localizado f1={resumo['edf_localized_f1']:.3f}, "
        f"crises localizadas={resumo['crises_localizadas']}/{resumo['arquivos_crise']}, "
        f"FP={resumo['falsos_positivos']}, FN={resumo['falsos_negativos']}, "
        f"FN_loc={resumo['falsos_negativos_localizacao']}, "
        f"thr={threshold}, dur={duration}, gap={max_gap}, hyst={hysteresis_ratio}",
        flush=True,
    )
    if confirmer_eval is not None:
        confirmado = confirmer_eval["test"]
        print(
            f"{paciente_teste} confirmador: event_f1={confirmado['event_f1']:.3f}, "
            f"sens={confirmado['event_sensitivity']:.3f}, "
            f"FA/h={confirmado['false_alarms_per_hour']:.3f}, "
            f"localizado_f1={confirmado['localized_metrics']['f1']:.3f}",
            flush=True,
        )
    return fold


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validacao CNN-LSTM sequencial leave-one-patient-out.")
    parser.add_argument("--dataset-dir", type=Path, default=PROJECT_ROOT / "dataset_amostra")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--test-patients", nargs="+", default=["chb01", "chb03", "chb11", "chb13", "chb14", "chb15"])
    parser.add_argument(
        "--reserved-test-patients",
        nargs="+",
        default=None,
        help="Pacientes excluidos de treino/calibracao em todos os folds.",
    )
    parser.add_argument(
        "--fixed-calibration-patients",
        nargs="+",
        default=None,
        help="Coorte de calibracao fixa, excluida do treino em todos os folds.",
    )
    parser.add_argument("--window-seconds", type=float, default=settings.ai_sequence_window_seconds)
    parser.add_argument("--step-seconds", type=float, default=settings.ai_sequence_step_seconds)
    parser.add_argument("--sequence-length", type=int, default=settings.ai_sequence_length)
    parser.add_argument("--sequence-stride", type=int, default=settings.ai_sequence_stride)
    parser.add_argument("--sequence-target-mode", choices=["any", "center"], default="center")
    parser.add_argument(
        "--sampling-level",
        choices=["window", "sequence"],
        default="window",
        help="Define se o limite por classe e aplicado antes ou depois de formar sequencias.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["center", "segmentation"],
        default="center",
        help="center classifica uma janela; segmentation retorna um score por passo temporal.",
    )
    parser.add_argument("--min-window-ictal-overlap-ratio", type=float, default=0.5)
    parser.add_argument("--boundary-sample-weight", type=float, default=1.5)
    parser.add_argument(
        "--event-balanced-weights",
        action="store_true",
        help="Equaliza a contribuicao das crises independentemente de sua duracao.",
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
    parser.add_argument("--max-normal-windows-per-file", type=int, default=settings.ai_sequence_max_normal_windows_per_file)
    parser.add_argument("--max-seizure-windows-per-file", type=int, default=settings.ai_sequence_max_seizure_windows_per_file)
    parser.add_argument("--calibration-patients-per-fold", type=int, default=settings.ai_sequence_calibration_patients_per_fold)
    parser.add_argument(
        "--calibration-selection-mode",
        choices=["balanced", "conservative", "event_balanced"],
        default="balanced",
        help="event_balanced otimiza eventos; balanced prioriza EDF/localizacao; conservative prioriza FP.",
    )
    parser.add_argument("--epochs", type=int, default=settings.ai_sequence_epochs)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=settings.ai_sequence_batch_size)
    parser.add_argument("--class-weight-mode", choices=["balanced", "manual", "none"], default="balanced")
    parser.add_argument("--positive-class-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=["binary_crossentropy", "focal"], default="binary_crossentropy")
    parser.add_argument("--focal-alpha", type=float, default=0.35)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--min-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--thresholds", default=settings.ai_sequence_thresholds)
    parser.add_argument("--durations", default=settings.ai_sequence_durations)
    parser.add_argument(
        "--max-gap-values",
        default="0",
        help="Lacunas maximas, em segundos, testadas ao unir trechos suspeitos.",
    )
    parser.add_argument(
        "--hysteresis-ratios",
        default="1.0",
        help="Razoes do threshold usadas para sustentar um trecho iniciado por um pico.",
    )
    parser.add_argument(
        "--candidate-confirmer",
        action="store_true",
        help="Treina um segundo estagio com candidatos dos pacientes de calibracao.",
    )
    parser.add_argument("--candidate-generation-threshold", type=float, default=0.1)
    parser.add_argument("--candidate-generation-gap-seconds", type=float, default=4.0)
    parser.add_argument("--candidate-max-per-file", type=int, default=30)
    parser.add_argument(
        "--candidate-confirmation-thresholds",
        default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
    )
    parser.add_argument(
        "--candidate-max-false-alarms-per-hour",
        type=float,
        default=1.0,
        help="Teto de FA/h aplicado somente a predicoes OOF da calibracao.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "modelos" / "sequence_leave_one_patient_eval.json")
    parser.add_argument("--partial-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Retoma folds ja gravados no partial-output.")
    args = parser.parse_args()

    arquivos = carregar_arquivos(args.dataset_dir, args.manifest)
    por_paciente = agrupar_por_paciente(arquivos)
    thresholds = parse_float_list(args.thresholds)
    durations = parse_float_list(args.durations)
    max_gap_values = parse_float_list(args.max_gap_values)
    hysteresis_ratios = parse_float_list(args.hysteresis_ratios)
    candidate_confirmation_thresholds = parse_float_list(args.candidate_confirmation_thresholds)
    canais_referencia = None
    if args.feature_mode in {
        FEATURE_MODE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    }:
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
        sequence_target_mode=args.sequence_target_mode,
        sampling_level=args.sampling_level,
        min_window_ictal_overlap_ratio=args.min_window_ictal_overlap_ratio,
        feature_normalization=args.feature_normalization,
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
            reserved_test_patients=set(args.reserved_test_patients or args.test_patients),
            fixed_calibration_patients=args.fixed_calibration_patients,
        )
        validar_isolamento_fold(
            train_files=train_files,
            calibration_files=calibration_files,
            test_files=test_files,
            reserved_test_patients=set(args.reserved_test_patients or args.test_patients),
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
                output_mode=args.output_mode,
                feature_mode=args.feature_mode,
                canais_referencia=canais_referencia,
                epochs=args.epochs,
                ensemble_size=max(1, args.ensemble_size),
                batch_size=args.batch_size,
                class_weight_mode=args.class_weight_mode,
                positive_class_weight=args.positive_class_weight,
                boundary_sample_weight=args.boundary_sample_weight,
                event_balanced_weights=args.event_balanced_weights,
                loss_mode=args.loss,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                early_stopping_patience=args.early_stopping_patience,
                min_overlap_ratio=args.min_overlap_ratio,
                calibration_selection_mode=args.calibration_selection_mode,
                thresholds=thresholds,
                durations=durations,
                max_gap_values=max_gap_values,
                hysteresis_ratios=hysteresis_ratios,
                candidate_confirmer=args.candidate_confirmer,
                candidate_generation_threshold=args.candidate_generation_threshold,
                candidate_generation_gap_seconds=args.candidate_generation_gap_seconds,
                candidate_max_per_file=args.candidate_max_per_file,
                candidate_confirmation_thresholds=candidate_confirmation_thresholds,
                candidate_max_false_alarms_per_hour=args.candidate_max_false_alarms_per_hour,
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
