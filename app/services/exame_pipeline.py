"""Pipeline assincrono de processamento de exames (IA + XAI)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select

from app.ai_engine.feature_extractor import (
    FEATURE_MODE_MEAN,
    FEATURE_MODE_PER_CHANNEL,
    FEATURE_NAMES,
    extrair_features_edf,
    extrair_features_edf_janelado,
)
from app.ai_engine.inference import (
    criar_modelo_cnn_lstm_dummy,
    obter_metadados_modelo,
    obter_modelo_keras,
    obter_scaler_modelo,
    obter_vetor_escalonado,
    realizar_inferencia,
)
from app.ai_engine.sequence_inference import (
    RecursosSequenciais,
    analisar_exame_sequencial,
    carregar_recursos_sequenciais,
)
from app.ai_engine.sequence_shap import gerar_mapa_shap_sequencial
from app.ai_engine.shap_explainer import gerar_mapa_shap
from app.config import Settings, get_settings
from app.database import AsyncSessionLocal
from app.models import Exame, PredicaoIA

logger = logging.getLogger(__name__)

FeatureExtractorFn = Callable[..., dict[str, Any]]
InferenciaFn = Callable[[dict[str, Any], str], Awaitable[float]]
ShapFn = Callable[..., str]

WINDOW_SECONDS = 4.0
STEP_SECONDS = 2.0
SUSPICIOUS_WINDOW_THRESHOLD = 0.9
MIN_SUSPICIOUS_DURATION_SECONDS = 10.0


def _resolver_modelo_path(settings: Settings) -> str:
    """Resolve o caminho do modelo; em dev cria dummy se ausente."""
    path = Path(settings.keras_model_path)
    if path.exists():
        return str(path.resolve())

    if settings.is_development:
        path.parent.mkdir(parents=True, exist_ok=True)
        criar_modelo_cnn_lstm_dummy().save(path)
        logger.warning("Modelo ausente - dummy CNN-LSTM criado em %s", path)
        return str(path.resolve())

    raise FileNotFoundError(f"Modelo CNN-LSTM nao encontrado: {path}")


def _resumir_canais_por_magnitude(
    vetor_escalonado: list[float] | Any,
    nomes_features: list[str],
    top_n: int = 5,
) -> list[dict[str, float | str]]:
    scores: dict[str, list[float]] = {}
    for nome, valor in zip(nomes_features, vetor_escalonado, strict=False):
        if "::" not in nome:
            continue
        canal, _feature = nome.split("::", 1)
        scores.setdefault(canal, []).append(abs(float(valor)))

    ranking = [
        {"canal": canal, "score": float(sum(valores) / max(len(valores), 1))}
        for canal, valores in scores.items()
    ]
    ranking.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranking[:top_n]


def _predizer_tensor(modelo: Any, vetor_escalonado: Any) -> float:
    tensor = (
        np.asarray(vetor_escalonado, dtype=np.float32)
        .reshape(1, -1, 1)
    )
    saida = modelo.predict(tensor, verbose=0)
    return float(np.asarray(saida).reshape(-1)[0])


def _predizer_janelas(
    *,
    modelo: Any,
    scaler: Any,
    janelas: list[dict[str, Any]],
) -> tuple[float, dict[str, Any], list[dict[str, float]], dict[str, float | int | bool]]:
    vetores = np.asarray([janela["feature_vector"] for janela in janelas], dtype=np.float32)
    if scaler is not None:
        vetores = scaler.transform(vetores)

    tensores = vetores.reshape(vetores.shape[0], vetores.shape[1], 1)
    scores = np.asarray(modelo.predict(tensores, verbose=0)).reshape(-1)
    scores = np.clip(scores, 0.0, 1.0)
    indices_top = np.argsort(scores)[::-1][:5]
    janelas_top = [
        {
            "start_seconds": float(janelas[int(i)].get("window_start_seconds", 0.0)),
            "end_seconds": float(janelas[int(i)].get("window_end_seconds", 0.0)),
            "score": float(scores[int(i)]),
        }
        for i in indices_top
    ]
    indice_pico = int(indices_top[0])
    trecho_suspeito = _encontrar_melhor_trecho_suspeito(janelas, scores)
    duracao_trecho = float(trecho_suspeito["duration_seconds"])
    fator_continuidade = min(1.0, duracao_trecho / MIN_SUSPICIOUS_DURATION_SECONDS)
    score_agregado = float(float(trecho_suspeito["score_medio"]) * fator_continuidade)

    return score_agregado, janelas[indice_pico], janelas_top, trecho_suspeito


def _encontrar_melhor_trecho_suspeito(
    janelas: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float = SUSPICIOUS_WINDOW_THRESHOLD,
) -> dict[str, float | int | bool]:
    melhor: dict[str, float | int | bool] | None = None
    inicio_run: int | None = None

    def fechar_run(fim_exclusivo: int) -> None:
        nonlocal melhor, inicio_run
        if inicio_run is None:
            return

        indices = np.arange(inicio_run, fim_exclusivo)
        inicio = float(janelas[int(indices[0])].get("window_start_seconds", 0.0))
        fim = float(janelas[int(indices[-1])].get("window_end_seconds", inicio))
        score_medio = float(np.mean(scores[indices]))
        score_max = float(np.max(scores[indices]))
        candidato = {
            "start_seconds": inicio,
            "end_seconds": fim,
            "duration_seconds": float(max(0.0, fim - inicio)),
            "n_janelas": int(len(indices)),
            "score_medio": score_medio,
            "score_max": score_max,
            "threshold": float(threshold),
            "atingiu_duracao_minima": bool((fim - inicio) >= MIN_SUSPICIOUS_DURATION_SECONDS),
        }
        if melhor is None:
            melhor = candidato
        else:
            chave_candidato = (
                float(candidato["duration_seconds"]),
                float(candidato["score_medio"]),
                float(candidato["score_max"]),
            )
            chave_melhor = (
                float(melhor["duration_seconds"]),
                float(melhor["score_medio"]),
                float(melhor["score_max"]),
            )
            if chave_candidato > chave_melhor:
                melhor = candidato
        inicio_run = None

    for idx, score in enumerate(scores):
        if float(score) >= threshold:
            if inicio_run is None:
                inicio_run = idx
        else:
            fechar_run(idx)
    fechar_run(len(scores))

    if melhor is not None:
        return melhor

    indice_pico = int(np.argmax(scores))
    inicio = float(janelas[indice_pico].get("window_start_seconds", 0.0))
    fim = float(janelas[indice_pico].get("window_end_seconds", inicio))
    return {
        "start_seconds": inicio,
        "end_seconds": fim,
        "duration_seconds": float(max(0.0, fim - inicio)),
        "n_janelas": 1,
        "score_medio": float(scores[indice_pico]),
        "score_max": float(scores[indice_pico]),
        "threshold": float(threshold),
        "atingiu_duracao_minima": False,
    }


def _resumir_canais_por_ablacao(
    *,
    modelo: Any,
    vetor_bruto: Any,
    scaler: Any,
    nomes_features: list[str],
    top_n: int = 5,
) -> list[dict[str, float | str]]:
    """
    Mede o impacto de cada canal zerando apenas o bloco de features dele.

    Isso produz um ranking mais fiel ao comportamento real do modelo do que
    usar a magnitude das features escalonadas.
    """
    vetor = np.asarray(vetor_bruto, dtype=np.float32).ravel()
    if vetor.size != len(nomes_features):
        return []

    vetor_base = vetor.reshape(1, -1)
    if scaler is not None:
        vetor_base = scaler.transform(vetor_base)
    base_score = _predizer_tensor(modelo, vetor_base.ravel())
    indices_por_canal: dict[str, list[int]] = {}
    for idx, nome in enumerate(nomes_features):
        if "::" not in nome:
            continue
        canal, _feature = nome.split("::", 1)
        indices_por_canal.setdefault(canal, []).append(idx)

    ranking: list[dict[str, float | str]] = []
    for canal, indices in indices_por_canal.items():
        vetor_abladado = vetor.copy()
        vetor_abladado[indices] = 0.0
        vetor_abladado_2d = vetor_abladado.reshape(1, -1)
        if scaler is not None:
            vetor_abladado_2d = scaler.transform(vetor_abladado_2d)
        score_abladado = _predizer_tensor(modelo, vetor_abladado_2d.ravel())
        impacto = base_score - score_abladado
        ranking.append(
            {
                "canal": canal,
                "score": float(abs(impacto)),
                "impacto": float(impacto),
                "score_sem_canal": float(score_abladado),
            }
        )

    ranking.sort(key=lambda item: float(item["score"]), reverse=True)
    return ranking[:top_n]


def _montar_detalhes_predicao(
    *,
    features: dict[str, Any],
    modelo: Any,
    scaler: Any,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    feature_names = list(features.get("feature_names", FEATURE_NAMES))
    feature_mode = str(features.get("feature_mode") or model_metadata.get("feature_mode") or FEATURE_MODE_MEAN)
    detalhes = {
        "feature_mode": feature_mode,
        "canais_processados": list(features.get("canais_processados", [])),
        "canais_omitidos": list(features.get("canais_omitidos", [])),
        "canais_referencia": list(features.get("canais_referencia", [])),
    }
    if "window_start_seconds" in features and "window_end_seconds" in features:
        detalhes["janela_pico"] = {
            "start_seconds": float(features.get("window_start_seconds", 0.0)),
            "end_seconds": float(features.get("window_end_seconds", 0.0)),
            "score": float(features.get("score_pico", 0.0)),
        }
    if "janelas_top" in features:
        detalhes["janelas_top"] = list(features.get("janelas_top", []))
    if "trecho_suspeito" in features:
        detalhes["trecho_suspeito"] = dict(features.get("trecho_suspeito", {}))
    if "n_janelas_analisadas" in features:
        detalhes["n_janelas_analisadas"] = int(features.get("n_janelas_analisadas", 0))
    if "score_agregacao" in features:
        detalhes["score_agregacao"] = str(features.get("score_agregacao"))
    if feature_mode == FEATURE_MODE_PER_CHANNEL:
        detalhes["canais_destaque"] = _resumir_canais_por_ablacao(
            modelo=modelo,
            vetor_bruto=features.get("feature_vector", []),
            scaler=scaler,
            nomes_features=feature_names,
        )
    else:
        detalhes["canais_destaque"] = []
    return detalhes


def _detalhes_sequenciais(resultado: dict[str, Any]) -> dict[str, Any]:
    """Mapeia o resultado do pipeline sequencial para o JSON de detalhes."""
    return {
        "model_type": resultado.get("model_type", "sequence_cnn_lstm"),
        "classificacao_clinica": resultado.get("classificacao_clinica"),
        "feature_mode": resultado.get("feature_mode"),
        "canais_processados": list(resultado.get("canais_processados", [])),
        "canais_omitidos": list(resultado.get("canais_omitidos", [])),
        "montagem_incompleta": bool(resultado.get("montagem_incompleta", False)),
        "canais_destaque": [],
        "threshold": resultado.get("threshold"),
        "min_duration_seconds": resultado.get("min_duration_seconds"),
        "max_suspicious_coverage": resultado.get("max_suspicious_coverage"),
        "n_janelas_analisadas": resultado.get("n_janelas_analisadas"),
        "n_sequences_analisadas": resultado.get("n_sequences_analisadas"),
        "score_agregacao": resultado.get("score_agregacao"),
        "xai_method": resultado.get("xai_method"),
        "janela_pico": resultado.get("janela_pico"),
        "trecho_suspeito": resultado.get("trecho_suspeito"),
        "top_trechos_suspeitos": list(resultado.get("top_trechos_suspeitos", [])),
    }


async def _processar_exame_sequencial(
    session: Any,
    exame: Exame,
    exame_id: int,
    *,
    recursos: RecursosSequenciais,
    settings: Settings,
    canais_selecionados: list[str] | None,
    threshold_override: float | None,
    gerar_shap: bool,
) -> None:
    """Fluxo do modelo sequencial global com trechos e Gradient SHAP."""
    logger.info("Usando modelo sequencial (global) para exame %s", exame_id)
    resultado = await asyncio.to_thread(
        analisar_exame_sequencial,
        exame.arquivo_path,
        recursos=recursos,
        settings=settings,
        canais_selecionados=canais_selecionados,
        threshold_override=threshold_override,
    )

    score = float(resultado["score_geral"])
    detalhes = _detalhes_sequenciais(resultado)
    logger.info(
        "Inferencia sequencial concluida para exame %s - score=%.4f, %d sequencias",
        exame_id,
        score,
        int(resultado.get("n_sequences_analisadas", 0)),
    )

    mapa_shap_path = ""
    if gerar_shap:
        try:
            pico = resultado.get("janela_pico") or {}
            mapa_shap_path, shap_overlay = await asyncio.to_thread(
                gerar_mapa_shap_sequencial,
                recursos.model,
                resultado["_xai_background"],
                resultado["_xai_input"],
                exame_id=exame_id,
                canais=list(resultado.get("_xai_channels", [])),
                janelas=list(resultado.get("_xai_windows", [])),
                canais_omitidos=list(resultado.get("canais_omitidos", [])),
                inicio_seconds=float(pico.get("start_seconds", 0.0)),
                fim_seconds=float(pico.get("end_seconds", 0.0)),
            )
            if shap_overlay is not None:
                detalhes["shap_overlay"] = shap_overlay
        except Exception:
            logger.exception("Falha ao gerar Gradient SHAP sequencial para exame %s", exame_id)

    predicao = PredicaoIA(
        id_exame=exame_id,
        resultado_score=score,
        mapa_shap_path=mapa_shap_path,
        detalhes_json=json.dumps(detalhes, ensure_ascii=False),
    )
    session.add(predicao)
    await session.commit()
    logger.info("PredicaoIA (sequencial) salva para exame %s", exame_id)


async def processar_exame_ia(
    exame_id: int,
    *,
    canais_selecionados: list[str] | None = None,
    extrair_features: FeatureExtractorFn | None = None,
    inferir: InferenciaFn | None = None,
    gerar_shap: ShapFn | None = None,
    threshold_override: float | None = None,
    gerar_shap_sequencial: bool = True,
) -> None:
    """
    Pipeline completo em background: features -> CNN-LSTM -> SHAP -> PredicaoIA.
    """
    settings = get_settings()
    extrair = extrair_features or extrair_features_edf
    inferir_fn = inferir or realizar_inferencia
    shap_fn = gerar_shap or gerar_mapa_shap

    logger.info("Iniciando processamento IA do exame %s", exame_id)

    async with AsyncSessionLocal() as session:
        try:
            exame = await session.get(Exame, exame_id)
            if exame is None:
                logger.error("Exame %s nao encontrado no banco", exame_id)
                return

            if not Path(exame.arquivo_path).exists():
                logger.error("Arquivo EDF ausente para exame %s: %s", exame_id, exame.arquivo_path)
                return

            usar_sequencial = (
                extrair_features is None
                and inferir is None
                and settings.usar_modelo_sequencial
            )
            if usar_sequencial:
                recursos = await asyncio.to_thread(carregar_recursos_sequenciais, settings)
                if recursos is not None:
                    await _processar_exame_sequencial(
                        session,
                        exame,
                        exame_id,
                        recursos=recursos,
                        settings=settings,
                        canais_selecionados=canais_selecionados,
                        threshold_override=threshold_override,
                        gerar_shap=gerar_shap_sequencial,
                    )
                    return
                logger.warning(
                    "Modelo sequencial indisponivel - fallback legado para exame %s",
                    exame_id,
                )

            modelo_path = _resolver_modelo_path(settings)
            model_metadata = obter_metadados_modelo(modelo_path)
            feature_mode = str(model_metadata.get("feature_mode") or FEATURE_MODE_MEAN)
            canais_referencia = model_metadata.get("canais_referencia")

            modelo = obter_modelo_keras(modelo_path)
            scaler = obter_scaler_modelo(modelo_path)

            if extrair_features is None:
                janelas = await asyncio.to_thread(
                    extrair_features_edf_janelado,
                    exame.arquivo_path,
                    max_duration_seconds=settings.max_edf_duration_seconds,
                    canais_selecionados=canais_selecionados,
                    feature_mode=feature_mode,
                    canais_referencia=canais_referencia,
                    window_seconds=WINDOW_SECONDS,
                    step_seconds=STEP_SECONDS,
                )
                if not janelas:
                    raise ValueError("Nenhuma janela temporal foi gerada para inferencia.")

                score, features, janelas_top, trecho_suspeito = await asyncio.to_thread(
                    _predizer_janelas,
                    modelo=modelo,
                    scaler=scaler,
                    janelas=janelas,
                )
                features["n_janelas_analisadas"] = len(janelas)
                features["janelas_top"] = janelas_top
                features["trecho_suspeito"] = trecho_suspeito
                features["score_pico"] = float(janelas_top[0]["score"])
                features["score_agregacao"] = "continuous_suspicious_segment"
            else:
                features = await asyncio.to_thread(
                    extrair,
                    exame.arquivo_path,
                    max_duration_seconds=settings.max_edf_duration_seconds,
                    canais_selecionados=canais_selecionados,
                    feature_mode=feature_mode,
                    canais_referencia=canais_referencia,
                )
                score = await inferir_fn(features, modelo_path)
            logger.info(
                "Features extraidas para exame %s (%d amostras brutas, modo=%s)",
                exame_id,
                features.get("total_amostras_brutas", 0),
                feature_mode,
            )

            logger.info("Inferencia concluida para exame %s - score=%.4f", exame_id, score)

            vetor = obter_vetor_escalonado(features, modelo_path)
            nomes_features = list(features.get("feature_names", FEATURE_NAMES))

            mapa_shap_path = await asyncio.to_thread(
                shap_fn,
                modelo,
                vetor,
                nomes_features,
                exame_id,
            )
            logger.info("Mapa SHAP gerado para exame %s: %s", exame_id, mapa_shap_path)

            detalhes = _montar_detalhes_predicao(
                features=features,
                modelo=modelo,
                scaler=scaler,
                model_metadata=model_metadata,
            )

            predicao = PredicaoIA(
                id_exame=exame_id,
                resultado_score=score,
                mapa_shap_path=mapa_shap_path,
                detalhes_json=json.dumps(detalhes, ensure_ascii=False),
            )
            session.add(predicao)
            await session.commit()

            logger.info("PredicaoIA salva para exame %s", exame_id)

        except Exception:
            await session.rollback()
            logger.exception("Falha no processamento IA do exame %s", exame_id)
            raise


async def buscar_predicao_exame(exame_id: int) -> PredicaoIA | None:
    """Retorna a predicao mais recente de um exame, se existir."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PredicaoIA)
            .where(PredicaoIA.id_exame == exame_id)
            .order_by(PredicaoIA.data_analise.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
