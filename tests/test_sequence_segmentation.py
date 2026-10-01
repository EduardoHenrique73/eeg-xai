"""Testes dos alvos e da agregacao da segmentacao temporal."""

from __future__ import annotations

import numpy as np
import pytest

from app.ai_engine.sequence_segmentation import (
    agregar_predicoes_temporais,
    construir_alvos_temporais,
    pesos_temporais,
)
from app.ai_engine.training import SeizureInterval


def test_construir_alvos_temporais_rotula_cada_passo():
    metas = [
        {
            "arquivo": "chb01_03.edf",
            "paciente": "chb01",
            "context_start_seconds": 0.0,
        }
    ]
    y, metas_temporais = construir_alvos_temporais(
        metas,
        {"chb01_03.edf": [SeizureInterval(4.0, 8.0)]},
        sequence_length=4,
        window_seconds=4.0,
        step_seconds=2.0,
        min_ictal_overlap_ratio=0.5,
    )

    assert y.shape == (1, 4, 1)
    assert y[0, :, 0].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert metas_temporais[0][0]["context"] == "interictal"


def test_agregar_predicoes_temporais_tira_media_das_janelas_repetidas():
    metas = [
        [
            {"arquivo": "a.edf", "start_seconds": 0.0, "end_seconds": 4.0},
            {"arquivo": "a.edf", "start_seconds": 2.0, "end_seconds": 6.0},
        ],
        [
            {"arquivo": "a.edf", "start_seconds": 2.0, "end_seconds": 6.0},
            {"arquivo": "a.edf", "start_seconds": 4.0, "end_seconds": 8.0},
        ],
    ]
    y = np.asarray([[[0], [1]], [[1], [0]]], dtype=np.float32)
    scores = np.asarray([[[0.1], [0.6]], [[0.8], [0.2]]], dtype=np.float32)

    y_flat, scores_flat, meta_flat = agregar_predicoes_temporais(scores, y, metas)

    assert y_flat.tolist() == [0, 1, 0]
    assert scores_flat.tolist() == pytest.approx([0.1, 0.7, 0.2])
    assert [item["start_seconds"] for item in meta_flat] == [0.0, 2.0, 4.0]


def test_pesos_temporais_aplica_borda_e_classe():
    y = np.asarray([[[0], [1]]], dtype=np.float32)
    metas = [[{"context": "boundary"}, {"context": "ictal"}]]

    pesos = pesos_temporais(y, metas, boundary_weight=1.5, class_weight={0: 0.5, 1: 2.0})

    assert pesos.tolist() == [[0.75, 2.0]]


def test_pesos_temporais_balanceia_eventos_longos_e_curtos():
    y = np.asarray([[[1], [1], [1]], [[1], [0], [0]]], dtype=np.float32)
    metas = [
        [{"event_key": "a"}, {"event_key": "a"}, {"event_key": "a"}],
        [{"event_key": "b"}, {"event_key": None}, {"event_key": None}],
    ]

    pesos = pesos_temporais(
        y,
        metas,
        boundary_weight=1.0,
        class_weight=None,
        balance_events=True,
    )

    assert float(np.sum(pesos[0])) == pytest.approx(float(pesos[1, 0]))
