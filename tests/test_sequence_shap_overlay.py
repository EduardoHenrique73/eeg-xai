"""Mapeamento conservador das atribuicoes SHAP para tempo e canal."""

import numpy as np
import pytest
from unittest.mock import patch

from app.ai_engine.sequence_shap import _resumir_por_canal, gerar_mapa_shap_sequencial


def test_overlay_usa_ordem_de_canais_e_intervalos_reais():
    matriz = np.array([[1, 3, 2, 2], [4, 4, 1, 1]], dtype=np.float32)
    janelas = [
        {"start_seconds": 10.0, "end_seconds": 14.0},
        {"start_seconds": 12.0, "end_seconds": 16.0},
    ]

    por_canal, overlay = _resumir_por_canal(matriz, ["A", "B"], janelas, ["B"])

    assert por_canal.tolist() == [[2.0, 4.0], [2.0, 1.0]]
    assert overlay == {
        "scope": "peak_sequence",
        "basis": "window_features",
        "cells": [
            {"canal": "A", "start_seconds": 10.0, "end_seconds": 14.0, "intensity": 0.5},
            {"canal": "A", "start_seconds": 12.0, "end_seconds": 16.0, "intensity": 1.0},
        ],
    }


def test_overlay_nao_atribui_canal_quando_formato_nao_confere():
    _, overlay = _resumir_por_canal(
        np.ones((2, 3), dtype=np.float32),
        ["A", "B"],
        [{"start_seconds": 0, "end_seconds": 4}] * 2,
        [],
    )
    assert overlay is None


def test_overlay_rejeita_intervalo_invalido():
    with pytest.raises(ValueError, match="Janela temporal"):
        _resumir_por_canal(
            np.ones((1, 2), dtype=np.float32),
            ["A"],
            [{"start_seconds": 4, "end_seconds": 4}],
            [],
        )


def test_overlay_nao_pinta_atribuicao_contraria_a_crise():
    por_canal, overlay = _resumir_por_canal(
        np.array([[-2, -1, 2, 2]], dtype=np.float32),
        ["A", "B"],
        [{"start_seconds": 0, "end_seconds": 4}],
        [],
    )
    assert por_canal.tolist() == [[0.0], [2.0]]
    assert overlay["cells"][0]["intensity"] == 0.0
    assert overlay["cells"][1]["intensity"] == 1.0


def test_overlay_agrega_janelas_repetidas_de_exame_curto():
    _, overlay = _resumir_por_canal(
        np.ones((3, 1), dtype=np.float32),
        ["A"],
        [{"start_seconds": 2, "end_seconds": 6}] * 3,
        [],
    )
    assert overlay["cells"] == [
        {"canal": "A", "start_seconds": 2.0, "end_seconds": 6.0, "intensity": 1.0}
    ]


def test_gerador_salva_png_e_devolve_overlay(tmp_path):
    class ExplainerFake:
        def __init__(self, _modelo, _background):
            pass

        def shap_values(self, _amostra):
            return np.array([[[[1.0], [-1.0]], [[2.0], [0.0]]]], dtype=np.float32)

    with patch("app.ai_engine.sequence_shap.shap.GradientExplainer", ExplainerFake):
        caminho, overlay = gerar_mapa_shap_sequencial(
            object(),
            np.zeros((2, 2, 2), dtype=np.float32),
            np.ones((1, 2, 2), dtype=np.float32),
            exame_id=7,
            canais=["A", "B"],
            janelas=[
                {"start_seconds": 10, "end_seconds": 14},
                {"start_seconds": 12, "end_seconds": 16},
            ],
            inicio_seconds=12,
            fim_seconds=16,
            output_dir=tmp_path,
        )

    assert caminho.endswith("exame_7_sequence_shap.png")
    assert (tmp_path / "exame_7_sequence_shap.png").read_bytes().startswith(b"\x89PNG")
    assert overlay["cells"][0]["start_seconds"] == 10.0
    assert overlay["cells"][2]["intensity"] == 0.0
