"""Testes do extrator de features (dinâmica simbólica + EDF)."""

from pathlib import Path

import mne
import numpy as np
import pytest

from app.ai_engine.feature_extractor import (
    FEATURE_NAMES,
    carregar_sinal_edf,
    extrair_features_de_valores,
    extrair_features_edf,
)
from app.ai_engine.symbolic_dynamics import (
    aplicar_dinamica_simbolica,
    calcular_entropia_shannon,
    calcular_frequencia,
    converter_para_decimal,
    gerar_grupos_deslizantes,
    gerar_sequencia_binaria,
    obter_limiar,
)


class TestDinamicaSimbolicaLegado:
    """Valida a matemática portada do dinamica_simbolica.py."""

    def test_entropia_distribuicao_uniforme(self):
        freq = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        assert calcular_entropia_shannon(freq) == pytest.approx(1.0, abs=1e-6)

    def test_entropia_simbolo_unico(self):
        assert calcular_entropia_shannon({0: 1.0}) == 0.0

    def test_entropia_distribuicao_concentrada(self):
        freq = {0: 0.9, 1: 0.05, 2: 0.03, 3: 0.02}
        entropia = calcular_entropia_shannon(freq)
        assert 0.0 < entropia < 0.5

    def test_pipeline_sinal_conhecido_8_amostras(self):
        sinal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        limiar = obter_limiar(sinal)
        assert limiar == pytest.approx(4.5)

        sequencia = gerar_sequencia_binaria(sinal, limiar)
        assert sequencia == ["0", "0", "0", "0", "1", "1", "1", "1"]

        grupos = gerar_grupos_deslizantes(sequencia, m=3)
        assert grupos == ["000", "000", "001", "011", "111", "111"]

        decimais = converter_para_decimal(grupos)
        assert decimais == [0, 0, 1, 3, 7, 7]

        frequencias = calcular_frequencia(decimais)
        assert frequencias[0] == pytest.approx(1 / 3)
        assert frequencias[1] == pytest.approx(1 / 6)
        assert calcular_entropia_shannon(frequencias) == pytest.approx(0.959148, abs=1e-5)

        resultado = aplicar_dinamica_simbolica(sinal, m=3)
        assert resultado["entropia"] == pytest.approx(0.959148, abs=1e-5)
        assert len(resultado["palavras_decimais"]) == 6


class TestExtrairFeatures:
    """Valida as 19 features do ml_classifier legado."""

    SINAL_REFERENCIA = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    def test_retorna_19_features_mais_metadados(self):
        features = extrair_features_de_valores(self.SINAL_REFERENCIA)

        for nome in FEATURE_NAMES:
            assert nome in features
            assert isinstance(features[nome], (int, float))

        assert features["feature_names"] == FEATURE_NAMES
        assert len(features["feature_vector"]) == 19

    def test_valores_esperados_sinal_referencia(self):
        features = extrair_features_de_valores(self.SINAL_REFERENCIA)

        assert features["entropia_shannon"] == pytest.approx(0.959148, abs=1e-5)
        assert features["limiar"] == pytest.approx(4.5)
        assert features["total_amostras"] == 8
        assert features["total_padroes"] == 6
        assert features["padroes_unicos"] == 4
        assert features["media_valores"] == pytest.approx(4.5)
        assert features["proporcao_uns"] == pytest.approx(0.5)
        assert features["transicoes"] == 1
        assert features["comprimento_sequencia"] == 8
        assert features["max_frequencia"] == pytest.approx(1 / 3, abs=1e-6)
        assert features["min_frequencia"] == pytest.approx(1 / 6, abs=1e-6)
        assert features["entropia_frequencias"] == pytest.approx(1.918296, abs=1e-5)

    def test_sinal_curto_demais_levanta_erro(self):
        with pytest.raises(ValueError, match="Frequências de padrões vazias"):
            extrair_features_de_valores(np.array([1.0, 2.0]))


class TestExtrairFeaturesEdf:
    """Integração com mne-python via arquivo .edf sintético."""

    @pytest.fixture
    def edf_sintetico(self, tmp_path) -> tuple[Path, np.ndarray, float]:
        sfreq = 256.0
        # Onda com alternância acima/abaixo da média para padrões ricos
        t = np.arange(0, 2.0, 1.0 / sfreq)
        sinal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 3 * t)
        data = sinal.reshape(1, -1)

        info = mne.create_info(ch_names=["EEG"], sfreq=sfreq, ch_types=["eeg"])
        raw = mne.io.RawArray(data, info, verbose=False)
        caminho = tmp_path / "teste_sintetico.edf"
        raw.export(caminho, fmt="edf", overwrite=True)
        return caminho, sinal, sfreq

    def test_extrair_features_edf_arquivo_real(self, edf_sintetico):
        caminho, _sinal_original, sfreq = edf_sintetico

        features_edf = extrair_features_edf(str(caminho), max_duration_seconds=None)
        sinal_carregado, sfreq_carregado, _ = carregar_sinal_edf(caminho, max_duration_seconds=None)
        features_array = extrair_features_de_valores(sinal_carregado)

        assert sfreq_carregado == pytest.approx(sfreq)
        for nome in FEATURE_NAMES:
            assert features_edf[nome] == pytest.approx(features_array[nome], rel=1e-6)

        assert features_edf["arquivo_path"] == str(caminho.resolve())
        assert features_edf["taxa_amostragem"] == sfreq
        assert features_edf["total_amostras_brutas"] == sinal_carregado.size

    def test_arquivo_inexistente_levanta_erro(self):
        with pytest.raises(FileNotFoundError):
            extrair_features_edf("/caminho/inexistente.edf")
