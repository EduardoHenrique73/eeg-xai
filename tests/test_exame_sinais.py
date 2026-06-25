"""Testes do serviço de extração de sinais para visualização."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.exame_sinais import extrair_sinais_para_visualizacao


def _mock_canais(n: int) -> list[str]:
    return [f"CH{i + 1}" for i in range(n)]


def test_extrair_sinais_downsample_para_max_pontos(monkeypatch) -> None:
    taxa = 256.0
    sinal_longo = np.sin(np.linspace(0, 40 * np.pi, 10_000))

    monkeypatch.setattr(
        "app.services.exame_sinais.carregar_sinal_edf",
        lambda *_args, **_kwargs: (sinal_longo, taxa, 4),
    )
    monkeypatch.setattr(
        "app.services.exame_sinais.listar_canais_eeg_edf",
        lambda *_args, **_kwargs: _mock_canais(4),
    )

    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=1500)

    assert resultado["n_pontos_original"] == 10_000
    assert resultado["n_pontos_retornados"] == 1500
    assert resultado["n_canais_eeg"] == 4
    assert resultado["canais_eeg"] == _mock_canais(4)
    assert len(resultado["pontos"]) == 1500
    assert resultado["pontos"][0]["tempo"] == 0.0
    assert resultado["pontos"][-1]["tempo"] > 0


def test_extrair_sinais_mantem_todos_pontos_quando_curto(monkeypatch) -> None:
    taxa = 128.0
    # Valores em Volts (como retorna o MNE)
    sinal_curto = np.array([1e-6, 2e-6, 3e-6, 4e-6])

    monkeypatch.setattr(
        "app.services.exame_sinais.carregar_sinal_edf",
        lambda *_args, **_kwargs: (sinal_curto, taxa, 2),
    )
    monkeypatch.setattr(
        "app.services.exame_sinais.listar_canais_eeg_edf",
        lambda *_args, **_kwargs: _mock_canais(2),
    )

    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=1500)

    assert resultado["n_pontos_retornados"] == 4
    assert [p["amplitude"] for p in resultado["pontos"]] == [1.0, 2.0, 3.0, 4.0]


def test_extrair_sinais_converte_volts_para_microvolts(monkeypatch) -> None:
    taxa = 256.0
    sinal = np.array([50e-6, -30e-6])

    monkeypatch.setattr(
        "app.services.exame_sinais.carregar_sinal_edf",
        lambda *_args, **_kwargs: (sinal, taxa, 1),
    )
    monkeypatch.setattr(
        "app.services.exame_sinais.listar_canais_eeg_edf",
        lambda *_args, **_kwargs: ["FP1"],
    )

    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=1500)

    assert resultado["pontos"][0]["amplitude"] == 50.0
    assert resultado["pontos"][1]["amplitude"] == -30.0
    assert resultado["canais_eeg"] == ["FP1"]
