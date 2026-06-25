"""Pipeline assíncrono de processamento de exames (IA + XAI)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.ai_engine.feature_extractor import FEATURE_NAMES, extrair_features_edf
from app.ai_engine.inference import (
    criar_modelo_cnn_lstm_dummy,
    obter_modelo_keras,
    obter_vetor_escalonado,
    realizar_inferencia,
)
from app.ai_engine.shap_explainer import gerar_mapa_shap
from app.config import Settings, get_settings
from app.database import AsyncSessionLocal
from app.models import Exame, PredicaoIA

logger = logging.getLogger(__name__)

FeatureExtractorFn = Callable[..., dict[str, Any]]
InferenciaFn = Callable[[dict[str, Any], str], Awaitable[float]]
ShapFn = Callable[..., str]


def _resolver_modelo_path(settings: Settings) -> str:
    """Resolve o caminho do modelo; em dev cria dummy se ausente."""
    path = Path(settings.keras_model_path)
    if path.exists():
        return str(path.resolve())

    if settings.is_development:
        path.parent.mkdir(parents=True, exist_ok=True)
        criar_modelo_cnn_lstm_dummy().save(path)
        logger.warning("Modelo ausente — dummy CNN-LSTM criado em %s", path)
        return str(path.resolve())

    raise FileNotFoundError(f"Modelo CNN-LSTM não encontrado: {path}")


async def processar_exame_ia(
    exame_id: int,
    *,
    canais_selecionados: list[str] | None = None,
    extrair_features: FeatureExtractorFn | None = None,
    inferir: InferenciaFn | None = None,
    gerar_shap: ShapFn | None = None,
) -> None:
    """
    Pipeline completo em background: features → CNN-LSTM → SHAP → PredicaoIA.

    1. Carrega o exame do banco (arquivo_path).
    2. Extrai 19 features do .edf via mne + dinâmica simbólica.
    3. Executa inferência CNN-LSTM (probabilidade de crise).
    4. Gera mapa SHAP explicativo e salva PNG em storage/shap/.
    5. Persiste o laudo em `predicoes_ia`.
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
                logger.error("Exame %s não encontrado no banco", exame_id)
                return

            if not Path(exame.arquivo_path).exists():
                logger.error("Arquivo EDF ausente para exame %s: %s", exame_id, exame.arquivo_path)
                return

            modelo_path = _resolver_modelo_path(settings)

            features = await asyncio.to_thread(
                extrair,
                exame.arquivo_path,
                max_duration_seconds=settings.max_edf_duration_seconds,
                canais_selecionados=canais_selecionados,
            )
            logger.info(
                "Features extraídas para exame %s (%d amostras)",
                exame_id,
                features.get("total_amostras_brutas", 0),
            )

            score = await inferir_fn(features, modelo_path)
            logger.info("Inferência concluída para exame %s — score=%.4f", exame_id, score)

            modelo = obter_modelo_keras(modelo_path)
            vetor = obter_vetor_escalonado(features, modelo_path)

            mapa_shap_path = await asyncio.to_thread(
                shap_fn,
                modelo,
                vetor,
                FEATURE_NAMES,
                exame_id,
            )
            logger.info("Mapa SHAP gerado para exame %s: %s", exame_id, mapa_shap_path)

            predicao = PredicaoIA(
                id_exame=exame_id,
                resultado_score=score,
                mapa_shap_path=mapa_shap_path,
            )
            session.add(predicao)
            await session.commit()

            logger.info("PredicaoIA salva para exame %s", exame_id)

        except Exception:
            await session.rollback()
            logger.exception("Falha no processamento IA do exame %s", exame_id)
            raise


async def buscar_predicao_exame(exame_id: int) -> PredicaoIA | None:
    """Retorna a predição mais recente de um exame, se existir."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(PredicaoIA)
            .where(PredicaoIA.id_exame == exame_id)
            .order_by(PredicaoIA.data_analise.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
