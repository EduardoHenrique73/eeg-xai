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
        "app.services.exame_sinais.carregar_sinais_edf",
        lambda *_args, **_kwargs: (
            np.vstack([sinal_longo] * 4),
            taxa,
            _mock_canais(4),
        ),
    )

    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=1500)

    assert resultado["n_pontos_original"] == 10_000
    assert resultado["n_pontos_retornados"] <= 1500
    assert resultado["n_canais_eeg"] == 4
    assert resultado["canais_eeg"] == _mock_canais(4)
    assert len(resultado["series"]) == 4
    assert len(resultado["pontos"]) == resultado["n_pontos_retornados"]
    assert resultado["pontos"][0]["tempo"] == 0.0
    assert resultado["pontos"][-1]["tempo"] > 0


def test_preserva_picos_por_canal(monkeypatch):
    sinais = np.zeros((2, 10000))
    sinais[0, 123] = 90e-6
    sinais[1, 8765] = -70e-6
    monkeypatch.setattr(
        "app.services.exame_sinais.carregar_sinais_edf",
        lambda *args, **kwargs: (sinais, 256.0, ["A", "B"]),
    )
    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=100)
    a, b = resultado["series"]
    assert max(p["amplitude"] for p in a["pontos"]) == 90.0
    assert min(p["amplitude"] for p in b["pontos"]) == -70.0
    assert next(p["tempo"] for p in a["pontos"] if p["amplitude"] == 90) == round(123 / 256, 4)
    assert [p["tempo"] for p in a["pontos"]] == [p["tempo"] for p in b["pontos"]]
    assert resultado["pontos"][-1]["tempo"] == round(9999 / 256, 4)


def test_extrair_sinais_mantem_todos_pontos_quando_curto(monkeypatch) -> None:
    taxa = 128.0
    # Valores em Volts (como retorna o MNE)
    sinal_curto = np.array([1e-6, 2e-6, 3e-6, 4e-6])

    monkeypatch.setattr(
        "app.services.exame_sinais.carregar_sinais_edf",
        lambda *_args, **_kwargs: (
            np.vstack([sinal_curto, sinal_curto]),
            taxa,
            _mock_canais(2),
        ),
    )

    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=1500)

    assert resultado["n_pontos_retornados"] == 4
    assert [p["amplitude"] for p in resultado["pontos"]] == [1.0, 2.0, 3.0, 4.0]


def test_extrair_sinais_converte_volts_para_microvolts(monkeypatch) -> None:
    taxa = 256.0
    sinal = np.array([50e-6, -30e-6])

    monkeypatch.setattr(
        "app.services.exame_sinais.carregar_sinais_edf",
        lambda *_args, **_kwargs: (sinal.reshape(1, -1), taxa, ["FP1"]),
    )

    resultado = extrair_sinais_para_visualizacao("fake.edf", max_pontos=1500)

    assert resultado["pontos"][0]["amplitude"] == 50.0
    assert resultado["pontos"][1]["amplitude"] == -30.0
    assert resultado["canais_eeg"] == ["FP1"]


def test_recorte_temporal_mantem_tempo_absoluto_e_mais_pontos(monkeypatch) -> None:
    taxa = 10.0
    sinal = np.arange(1000, dtype=float) * 1e-6
    class RawFalso:
        info = {"sfreq": taxa}
        n_times = len(sinal)
        ch_names = ["A"]

        def pick(self, canais):
            assert canais == ["A"]

        def get_data(self, *, start, stop):
            assert (start, stop) == (200, 240)
            return sinal[start:stop].reshape(1, -1)

    def ler_edf(_path, *, preload, verbose):
        assert preload is False
        assert verbose is False
        return RawFalso()

    monkeypatch.setattr("app.services.exame_sinais.mne.io.read_raw_edf", ler_edf)
    monkeypatch.setattr(
        "app.services.exame_sinais.selecionar_canais_eeg_validos",
        lambda _raw, _canais: ["A"],
    )

    resultado = extrair_sinais_para_visualizacao(
        "fake.edf", max_pontos=1500, start_seconds=20, end_seconds=24
    )

    assert resultado["n_pontos_retornados"] == 40
    assert resultado["pontos"][0]["tempo"] == 20.0
    assert resultado["pontos"][-1]["tempo"] == 23.9
    assert resultado["pontos"][0]["amplitude"] == 200.0


def test_recorte_temporal_rejeita_intervalo_excessivo() -> None:
    with pytest.raises(ValueError, match="120 segundos"):
        extrair_sinais_para_visualizacao(
            "fake.edf", start_seconds=0, end_seconds=121
        )
