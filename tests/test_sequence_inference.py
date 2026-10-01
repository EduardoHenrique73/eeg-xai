"""Testes da logica de inferencia sequencial (sem dependencia de TensorFlow)."""

from __future__ import annotations

import numpy as np

from app.config import Settings
from app.ai_engine.sequence_inference import (
    agregar_trechos_suspeitos,
    classificar_resultado_sequencial,
    construir_sequencias,
    resolver_threshold_duracao,
)


def _janela(inicio: float, fim: float, valor: float) -> dict:
    return {
        "feature_vector": [valor, valor + 1.0, valor + 2.0],
        "window_start_seconds": inicio,
        "window_end_seconds": fim,
    }


def test_construir_sequencias_gera_blocos_consecutivos():
    janelas = [_janela(i * 2.0, i * 2.0 + 4.0, float(i)) for i in range(6)]
    x, meta = construir_sequencias(janelas, sequence_length=3, sequence_stride=1)

    assert x.shape == (4, 3, 3)
    assert meta[0]["start_seconds"] == 0.0
    assert meta[0]["end_seconds"] == 8.0
    assert meta[-1]["start_seconds"] == 6.0


def test_construir_sequencias_center_retorna_janela_central():
    janelas = [_janela(i * 2.0, i * 2.0 + 4.0, float(i)) for i in range(6)]
    x, meta = construir_sequencias(
        janelas,
        sequence_length=4,
        sequence_stride=1,
        target_mode="center",
    )

    assert x.shape == (3, 4, 3)
    assert meta[0]["start_seconds"] == 4.0
    assert meta[0]["end_seconds"] == 8.0
    assert meta[0]["context_start_seconds"] == 0.0
    assert meta[0]["context_end_seconds"] == 10.0


def test_construir_sequencias_completa_quando_poucas_janelas():
    janelas = [_janela(0.0, 4.0, 1.0), _janela(2.0, 6.0, 2.0)]
    x, meta = construir_sequencias(janelas, sequence_length=4, sequence_stride=2)

    assert x.shape == (1, 4, 3)
    assert len(meta) == 1
    assert meta[0]["start_seconds"] == 0.0


def test_agregar_trechos_agrupa_sequencias_contiguas():
    scores = np.array([0.1, 0.9, 0.95, 0.92, 0.2, 0.88, 0.91])
    meta = [{"start_seconds": float(i * 2), "end_seconds": float(i * 2 + 4)} for i in range(len(scores))]

    trechos = agregar_trechos_suspeitos(
        scores, meta, threshold=0.8, min_duration_seconds=4.0, limite=5
    )

    assert len(trechos) == 2
    principal = trechos[0]
    assert principal["n_sequences"] == 3
    assert principal["score_max"] >= 0.92
    assert principal["atingiu_duracao_minima"] is True


def test_agregar_trechos_prioriza_evidencia_sustentada():
    scores = np.array([0.92, 0.92, 0.92, 0.2, 0.96, 0.97])
    meta = [{"start_seconds": float(i * 2), "end_seconds": float(i * 2 + 4)} for i in range(len(scores))]

    trechos = agregar_trechos_suspeitos(
        scores, meta, threshold=0.8, min_duration_seconds=4.0, limite=5
    )

    assert len(trechos) == 2
    assert trechos[0]["n_sequences"] == 3
    assert trechos[0]["evidencia_acumulada"] > trechos[1]["evidencia_acumulada"]
    assert trechos[0]["duration_seconds"] > trechos[1]["duration_seconds"]


def test_agregar_trechos_usa_pico_quando_nada_acima_do_threshold():
    scores = np.array([0.1, 0.2, 0.45, 0.3])
    meta = [{"start_seconds": float(i), "end_seconds": float(i + 1)} for i in range(len(scores))]

    trechos = agregar_trechos_suspeitos(
        scores, meta, threshold=0.9, min_duration_seconds=4.0, limite=5
    )

    assert len(trechos) == 1
    assert trechos[0]["score_max"] == 0.45
    assert trechos[0]["atingiu_duracao_minima"] is False


def test_resolver_threshold_prioriza_calibracao_alinhada():
    settings = Settings(ai_sequence_default_threshold=0.5, ai_sequence_default_min_duration_seconds=30.0)
    calibration = {
        "best": {"threshold": 0.9, "min_duration_seconds": 120.0},
        "best_aligned": {"threshold": 0.85, "min_duration_seconds": 60.0},
    }

    threshold, duracao = resolver_threshold_duracao({}, calibration, settings)

    assert threshold == 0.85
    assert duracao == 60.0


def test_resolver_threshold_usa_defaults_sem_calibracao():
    settings = Settings(ai_sequence_default_threshold=0.7, ai_sequence_default_min_duration_seconds=45.0)

    threshold, duracao = resolver_threshold_duracao({}, None, settings)

    assert threshold == 0.7
    assert duracao == 45.0


def test_classificacao_clinica_e_cautelosa():
    trecho_relevante = {"atingiu_duracao_minima": True}
    assert "epileptiforme" in classificar_resultado_sequencial(
        score_geral=0.95, trecho_principal=trecho_relevante, threshold=0.8
    )

    assert "indeterminado" in classificar_resultado_sequencial(
        score_geral=0.99,
        trecho_principal=trecho_relevante,
        threshold=0.8,
        cobertura_excessiva=True,
    )

    assert "limitado" in classificar_resultado_sequencial(
        score_geral=0.2,
        trecho_principal=None,
        threshold=0.8,
        montagem_incompleta=True,
    )

    assert "pontual" in classificar_resultado_sequencial(
        score_geral=0.9, trecho_principal={"atingiu_duracao_minima": False}, threshold=0.8
    )

    assert "Sem padrao" in classificar_resultado_sequencial(
        score_geral=0.3, trecho_principal=None, threshold=0.8
    )
