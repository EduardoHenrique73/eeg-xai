"""Inferencia com o modelo CNN-LSTM sequencial (global / inter-paciente).

Fluxo:
    EDF -> features por janela -> sequencias de janelas consecutivas ->
    scaler -> score por sequencia -> agregacao em trechos continuos suspeitos.

O modulo carrega o modelo `.keras`, o `scaler.pkl`, o `metadata.json` e,
quando disponivel, o `calibration.json` (threshold/duracao ja escolhidos).
Se o modelo ou o scaler nao existirem, `carregar_recursos_sequenciais`
retorna ``None`` para que o pipeline faca fallback ao fluxo legado.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.ai_engine.feature_extractor import (
    FEATURE_MODE_MEAN,
    extrair_features_edf_janelado,
    normalizar_matriz_features_robusta,
)
from app.config import Settings

logger = logging.getLogger(__name__)

# Defaults usados quando o metadata do modelo nao os define.
SEQUENCE_LENGTH_DEFAULT = 8
SEQUENCE_STRIDE_DEFAULT = 2
WINDOW_SECONDS_DEFAULT = 4.0
STEP_SECONDS_DEFAULT = 2.0


@dataclass(frozen=True)
class RecursosSequenciais:
    """Artefatos necessarios para uma inferencia sequencial."""

    model: Any
    scaler: Any
    metadata: dict[str, Any]
    calibration: dict[str, Any] | None
    model_path: str


def resolver_paths_sequenciais(
    settings: Settings,
) -> tuple[Path, Path, Path, Path | None]:
    """Resolve caminhos do modelo/scaler/metadata/calibracao.

    Scaler e metadata sao derivados do nome do modelo quando nao informados.
    """
    model_path = Path(settings.ai_sequence_model_path)
    scaler_path = (
        Path(settings.ai_sequence_scaler_path)
        if settings.ai_sequence_scaler_path
        else model_path.with_name(f"{model_path.stem}_scaler.pkl")
    )
    metadata_path = (
        Path(settings.ai_sequence_metadata_path)
        if settings.ai_sequence_metadata_path
        else model_path.with_name(f"{model_path.stem}_metadata.json")
    )
    calibration_path = (
        Path(settings.ai_sequence_calibration_path)
        if settings.ai_sequence_calibration_path
        else None
    )
    return model_path, scaler_path, metadata_path, calibration_path


@lru_cache(maxsize=4)
def _carregar_recursos_cached(
    model_path: str,
    scaler_path: str,
    metadata_path: str,
    calibration_path: str | None,
) -> RecursosSequenciais:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    with Path(scaler_path).open("rb") as arquivo:
        scaler = pickle.load(arquivo)

    metadata: dict[str, Any] = {}
    if metadata_path and Path(metadata_path).exists():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))

    calibration: dict[str, Any] | None = None
    if calibration_path and Path(calibration_path).exists():
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))

    return RecursosSequenciais(
        model=model,
        scaler=scaler,
        metadata=metadata,
        calibration=calibration,
        model_path=model_path,
    )


def carregar_recursos_sequenciais(settings: Settings) -> RecursosSequenciais | None:
    """Carrega os artefatos do modelo sequencial; ``None`` se indisponiveis."""
    model_path, scaler_path, metadata_path, calibration_path = resolver_paths_sequenciais(
        settings
    )

    if not model_path.exists():
        logger.warning(
            "Modelo sequencial ausente em %s - fallback para fluxo legado.", model_path
        )
        return None
    if not scaler_path.exists():
        logger.warning(
            "Scaler sequencial ausente em %s - fallback para fluxo legado.", scaler_path
        )
        return None

    try:
        return _carregar_recursos_cached(
            str(model_path.resolve()),
            str(scaler_path.resolve()),
            str(metadata_path.resolve()) if metadata_path else "",
            str(calibration_path.resolve()) if calibration_path else None,
        )
    except Exception:  # pragma: no cover - defensivo
        logger.exception("Falha ao carregar recursos do modelo sequencial.")
        return None


def limpar_cache_sequencial() -> None:
    """Libera os recursos sequenciais em cache."""
    _carregar_recursos_cached.cache_clear()


def resolver_threshold_duracao(
    metadata: dict[str, Any],
    calibration: dict[str, Any] | None,
    settings: Settings,
) -> tuple[float, float]:
    """Decide threshold e duracao minima.

    Prioridade: calibracao (best_aligned > best) > defaults do .env.
    """
    threshold = float(settings.ai_sequence_default_threshold)
    min_duration = float(settings.ai_sequence_default_min_duration_seconds)

    if calibration:
        escolhido = calibration.get("best_aligned") or calibration.get("best")
        if isinstance(escolhido, dict):
            if escolhido.get("threshold") is not None:
                threshold = float(escolhido["threshold"])
            if escolhido.get("min_duration_seconds") is not None:
                min_duration = float(escolhido["min_duration_seconds"])

    return threshold, min_duration


def construir_sequencias(
    janelas: list[dict[str, Any]],
    *,
    sequence_length: int,
    sequence_stride: int,
    target_mode: str = "span",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Monta blocos temporais de janelas consecutivas.

    Garante ao menos uma sequencia quando ha janelas suficientes; quando ha
    menos janelas que ``sequence_length``, completa por repeticao da ultima.
    """
    if not janelas:
        return np.empty((0, sequence_length, 0), dtype=np.float32), []
    if target_mode not in {"span", "center"}:
        raise ValueError("target_mode deve ser 'span' ou 'center'.")

    def metadados_temporais(bloco: list[dict[str, Any]], target_idx: int) -> dict[str, float]:
        alvo = bloco[target_idx]
        return {
            "start_seconds": float(
                alvo.get("window_start_seconds", 0.0)
                if target_mode == "center"
                else bloco[0].get("window_start_seconds", 0.0)
            ),
            "end_seconds": float(
                alvo.get("window_end_seconds", 0.0)
                if target_mode == "center"
                else bloco[-1].get("window_end_seconds", 0.0)
            ),
            "context_start_seconds": float(bloco[0].get("window_start_seconds", 0.0)),
            "context_end_seconds": float(bloco[-1].get("window_end_seconds", 0.0)),
        }

    vetores = [np.asarray(j["feature_vector"], dtype=np.float32) for j in janelas]
    n_features = vetores[0].shape[0]

    if len(vetores) < sequence_length:
        faltam = sequence_length - len(vetores)
        vetores = vetores + [vetores[-1]] * faltam
        janelas_pad = list(janelas) + [janelas[-1]] * faltam
        bloco = np.stack(vetores[:sequence_length])
        target_idx = min(len(janelas) - 1, sequence_length // 2)
        meta = metadados_temporais(janelas_pad[:sequence_length], target_idx)
        return bloco.reshape(1, sequence_length, n_features).astype(np.float32), [meta]

    x_seq: list[np.ndarray] = []
    meta_seq: list[dict[str, Any]] = []
    for inicio in range(0, len(vetores) - sequence_length + 1, sequence_stride):
        fim = inicio + sequence_length
        x_seq.append(np.stack(vetores[inicio:fim]))
        meta_seq.append(metadados_temporais(janelas[inicio:fim], sequence_length // 2))

    return np.asarray(x_seq, dtype=np.float32), meta_seq


def _aplicar_scaler(x: np.ndarray, scaler: Any) -> np.ndarray:
    if scaler is None:
        return x.astype(np.float32)
    n_seq, seq_len, n_features = x.shape
    plano = scaler.transform(x.reshape(n_seq * seq_len, n_features))
    return plano.reshape(n_seq, seq_len, n_features).astype(np.float32)


def _segmento(
    scores: np.ndarray,
    meta: list[dict[str, Any]],
    indices: np.ndarray,
    *,
    threshold: float,
    min_duration_seconds: float,
) -> dict[str, Any]:
    start = float(meta[int(indices[0])]["start_seconds"])
    end = float(meta[int(indices[-1])]["end_seconds"])
    duration = float(max(0.0, end - start))
    score_medio = float(np.mean(scores[indices]))
    score_max = float(np.max(scores[indices]))
    evidencia_acumulada = float(np.sum(np.maximum(scores[indices] - threshold, 0.0)))
    return {
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": duration,
        "n_sequences": int(indices.size),
        "n_janelas": int(indices.size),
        "score_medio": score_medio,
        "score_max": score_max,
        "evidencia_acumulada": evidencia_acumulada,
        "threshold": float(threshold),
        "atingiu_duracao_minima": bool(
            duration >= min_duration_seconds and score_medio >= threshold
        ),
    }


def agregar_trechos_suspeitos(
    scores: np.ndarray,
    meta: list[dict[str, Any]],
    *,
    threshold: float,
    min_duration_seconds: float,
    limite: int,
) -> list[dict[str, Any]]:
    """Agrupa sequencias contiguas com score >= threshold em trechos."""
    acima = scores >= threshold
    segmentos: list[dict[str, Any]] = []
    inicio_run: int | None = None

    for idx, suspeita in enumerate(acima):
        if suspeita and inicio_run is None:
            inicio_run = idx
        fim_de_run = (not suspeita) or idx == len(acima) - 1
        if inicio_run is not None and fim_de_run:
            fim = idx + 1 if suspeita and idx == len(acima) - 1 else idx
            indices = np.arange(inicio_run, fim)
            segmentos.append(
                _segmento(
                    scores,
                    meta,
                    indices,
                    threshold=threshold,
                    min_duration_seconds=min_duration_seconds,
                )
            )
            inicio_run = None

    if not segmentos:
        pico = int(np.argmax(scores))
        segmentos.append(
            _segmento(
                scores,
                meta,
                np.asarray([pico]),
                threshold=threshold,
                min_duration_seconds=min_duration_seconds,
            )
        )

    segmentos.sort(
        key=lambda seg: (
            bool(seg["atingiu_duracao_minima"]),
            float(seg["evidencia_acumulada"]),
            float(seg["score_medio"]),
            float(seg["score_max"]),
            float(seg["duration_seconds"]),
        ),
        reverse=True,
    )
    return segmentos[: max(1, limite)]


def classificar_resultado_sequencial(
    *,
    score_geral: float,
    trecho_principal: dict[str, Any] | None,
    threshold: float,
    cobertura_excessiva: bool = False,
    montagem_incompleta: bool = False,
) -> str:
    """Texto clinico cauteloso (apoio, nao diagnostico final)."""
    if montagem_incompleta:
        return (
            "Resultado limitado: canais de referencia ausentes ou omitidos "
            "reduzem a confiabilidade da analise"
        )
    if cobertura_excessiva:
        return (
            "Resultado indeterminado: o modelo marcou uma area extensa demais "
            "do exame como suspeita (revisao medica necessaria)"
        )
    if trecho_principal and trecho_principal.get("atingiu_duracao_minima"):
        return "Padrao sugestivo de atividade epileptiforme (revisao medica necessaria)"
    if score_geral >= threshold:
        return "Atividade suspeita pontual sem persistencia (requer revisao)"
    return "Sem padrao sugestivo de crise nas janelas analisadas"


def analisar_exame_sequencial(
    arquivo_path: str,
    *,
    recursos: RecursosSequenciais,
    settings: Settings,
    canais_selecionados: list[str] | None = None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    """Executa o pipeline sequencial completo e retorna o resultado agregado."""
    metadata = recursos.metadata
    feature_mode = str(metadata.get("feature_mode") or settings.ai_sequence_feature_mode or FEATURE_MODE_MEAN)
    sequence_length = int(metadata.get("sequence_length") or settings.ai_sequence_length or SEQUENCE_LENGTH_DEFAULT)
    sequence_stride = int(metadata.get("sequence_stride") or settings.ai_sequence_stride or SEQUENCE_STRIDE_DEFAULT)
    sequence_target_mode = str(metadata.get("sequence_target_mode") or "span")
    window_seconds = float(metadata.get("window_seconds") or settings.ai_sequence_window_seconds or WINDOW_SECONDS_DEFAULT)
    step_seconds = float(metadata.get("step_seconds") or settings.ai_sequence_step_seconds or STEP_SECONDS_DEFAULT)
    canais_referencia = metadata.get("canais_referencia") or None

    janelas = extrair_features_edf_janelado(
        arquivo_path,
        max_duration_seconds=settings.max_edf_duration_seconds,
        canais_selecionados=canais_selecionados,
        feature_mode=feature_mode,
        canais_referencia=canais_referencia,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    if not janelas:
        raise ValueError("Nenhuma janela temporal foi gerada para inferencia sequencial.")

    feature_normalization = str(metadata.get("feature_normalization") or "global_scaler")
    if feature_normalization == "per_edf_robust":
        matriz_janelas = normalizar_matriz_features_robusta(
            np.asarray([janela["feature_vector"] for janela in janelas], dtype=np.float32)
        )
        janelas = [
            {**janela, "feature_vector": matriz_janelas[idx].tolist()}
            for idx, janela in enumerate(janelas)
        ]

    x_seq, meta_seq = construir_sequencias(
        janelas,
        sequence_length=sequence_length,
        sequence_stride=sequence_stride,
        target_mode=sequence_target_mode,
    )
    if x_seq.shape[0] == 0:
        raise ValueError("Nao foi possivel montar sequencias para inferencia.")

    x_scaled = _aplicar_scaler(x_seq, recursos.scaler)
    scores = np.asarray(recursos.model.predict(x_scaled, verbose=0)).reshape(-1)
    scores = np.clip(scores, 0.0, 1.0)

    threshold, min_duration = resolver_threshold_duracao(
        metadata, recursos.calibration, settings
    )
    if threshold_override is not None:
        threshold = float(np.clip(threshold_override, 0.0, 1.0))

    trechos = agregar_trechos_suspeitos(
        scores,
        meta_seq,
        threshold=threshold,
        min_duration_seconds=min_duration,
        limite=int(settings.ai_sequence_top_segments_limit),
    )
    trecho_principal = trechos[0] if trechos else None
    duracao_analisada = float(
        max(
            0.0,
            float(meta_seq[-1]["end_seconds"]) - float(meta_seq[0]["start_seconds"]),
        )
    )
    cobertura_excessiva = False
    if trecho_principal and duracao_analisada > 0:
        cobertura = float(trecho_principal["duration_seconds"]) / duracao_analisada
        limite_cobertura = float(settings.ai_sequence_max_suspicious_coverage)
        cobertura_excessiva = bool(
            trecho_principal.get("atingiu_duracao_minima")
            and cobertura >= limite_cobertura
        )
        trecho_principal["coverage_ratio"] = cobertura
        trecho_principal["cobertura_excessiva"] = cobertura_excessiva
        trecho_principal["max_suspicious_coverage"] = limite_cobertura

    indice_pico = int(np.argmax(scores))
    janela_pico = {
        "start_seconds": float(meta_seq[indice_pico]["start_seconds"]),
        "end_seconds": float(meta_seq[indice_pico]["end_seconds"]),
        "score": float(scores[indice_pico]),
    }

    canais_processados = list(janelas[0].get("canais_processados", []))
    canais_omitidos = list(janelas[0].get("canais_omitidos", []))
    montagem_incompleta = bool(feature_mode != FEATURE_MODE_MEAN and canais_omitidos)
    resultado_conclusivo = bool(not montagem_incompleta and not cobertura_excessiva)
    resultado_positivo_conclusivo = bool(
        resultado_conclusivo
        and trecho_principal
        and trecho_principal.get("atingiu_duracao_minima")
    )

    score_geral = float(np.max(scores))
    classificacao = classificar_resultado_sequencial(
        score_geral=score_geral,
        trecho_principal=trecho_principal,
        threshold=threshold,
        cobertura_excessiva=cobertura_excessiva,
        montagem_incompleta=montagem_incompleta,
    )

    return {
        "model_type": "sequence_cnn_lstm",
        "score_geral": score_geral,
        "classificacao_clinica": classificacao,
        "threshold": float(threshold),
        "min_duration_seconds": float(min_duration),
        "max_suspicious_coverage": float(settings.ai_sequence_max_suspicious_coverage),
        "feature_mode": feature_mode,
        "feature_normalization": feature_normalization,
        "canais_processados": canais_processados,
        "canais_omitidos": canais_omitidos,
        "montagem_incompleta": montagem_incompleta,
        "cobertura_excessiva": cobertura_excessiva,
        "resultado_conclusivo": resultado_conclusivo,
        "resultado_positivo_conclusivo": resultado_positivo_conclusivo,
        "n_sequences_analisadas": int(scores.shape[0]),
        "n_janelas_analisadas": int(len(janelas)),
        "score_agregacao": "continuous_suspicious_sequence",
        "janela_pico": janela_pico,
        "trecho_suspeito": trecho_principal,
        "top_trechos_suspeitos": trechos,
        "xai_method": "gradient_shap",
        "_xai_input": x_scaled[indice_pico : indice_pico + 1],
        "_xai_background": x_scaled[
            np.linspace(0, len(x_scaled) - 1, min(16, len(x_scaled)), dtype=int)
        ],
    }
