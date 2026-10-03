"""Testes do extrator de features."""

from pathlib import Path

import mne
import numpy as np
import pytest

from app.ai_engine.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    RAW_SIGNAL_FEATURE_NAMES,
    TIME_FREQUENCY_FEATURE_NAMES,
    TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES,
    TIME_FREQUENCY_RELATIVE_FEATURE_NAMES,
    carregar_sinal_edf,
    extrair_features_por_modo_de_valores,
    extrair_features_de_valores,
    extrair_features_edf,
    nomes_features_por_modo,
    normalizar_matriz_features_robusta,
    selecionar_canais_eeg_validos,
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
        with pytest.raises(ValueError, match="Frequencias de padroes vazias"):
            extrair_features_de_valores(np.array([1.0, 2.0]))

    def test_features_tempo_frequencia_tem_dimensao_fixa(self):
        sfreq = 128.0
        t = np.arange(0, 4.0, 1.0 / sfreq)
        sinal = np.sin(2 * np.pi * 10 * t)

        features = extrair_features_por_modo_de_valores(
            sinal,
            feature_mode="time_frequency",
            sfreq=sfreq,
        )

        assert features["feature_names"] == TIME_FREQUENCY_FEATURE_NAMES
        assert len(features["feature_vector"]) == len(TIME_FREQUENCY_FEATURE_NAMES)
        assert features["potencia_alpha"] > features["potencia_delta"]

    def test_features_sinal_bruto_tem_dimensao_fixa_e_normalizada(self):
        sinal = np.linspace(-1.0, 1.0, 512)

        features = extrair_features_por_modo_de_valores(
            sinal,
            feature_mode="raw_signal",
            sfreq=128.0,
        )

        vetor = np.asarray(features["feature_vector"])
        assert features["feature_names"] == RAW_SIGNAL_FEATURE_NAMES
        assert vetor.shape == (128,)
        assert np.std(vetor) == pytest.approx(1.0)

    def test_features_tempo_frequencia_relativas_sao_invariantes_a_amplitude(self):
        sfreq = 128.0
        t = np.arange(0, 4.0, 1.0 / sfreq)
        sinal = np.sin(2 * np.pi * 10 * t) + 0.4 * np.sin(2 * np.pi * 5 * t)

        original = extrair_features_por_modo_de_valores(
            sinal,
            feature_mode="time_frequency_relative_per_channel",
            sfreq=sfreq,
        )
        escalado = extrair_features_por_modo_de_valores(
            sinal * 25.0,
            feature_mode="time_frequency_relative_per_channel",
            sfreq=sfreq,
        )

        assert original["feature_names"] == TIME_FREQUENCY_RELATIVE_FEATURE_NAMES
        assert np.asarray(original["feature_vector"]) == pytest.approx(
            np.asarray(escalado["feature_vector"]), rel=1e-8, abs=1e-8
        )
        assert all(0.0 <= original[name] <= 1.0 for name in [
            "frequencia_dominante_normalizada",
            "centroide_espectral_normalizado",
            "entropia_espectral_normalizada",
        ])

    def test_features_morfologicas_complementam_espectro_e_sao_invariantes_a_escala(self):
        sfreq = 128.0
        t = np.arange(0, 4.0, 1.0 / sfreq)
        sinal = np.sin(2 * np.pi * 8 * t) + 0.25 * np.sin(2 * np.pi * 19 * t)

        original = extrair_features_por_modo_de_valores(
            sinal,
            feature_mode=FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
            sfreq=sfreq,
        )
        escalado = extrair_features_por_modo_de_valores(
            sinal * 40.0,
            feature_mode=FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
            sfreq=sfreq,
        )

        assert original["feature_names"] == TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES
        assert len(original["feature_vector"]) == 23
        np.testing.assert_allclose(
            original["feature_vector"][15:], escalado["feature_vector"][15:], rtol=1e-6, atol=1e-6
        )
        assert np.all(np.isfinite(original["feature_vector"]))

    def test_nomes_features_tempo_frequencia_por_canal_usa_base_espectral(self):
        assert nomes_features_por_modo(FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL) == TIME_FREQUENCY_FEATURE_NAMES
        assert (
            nomes_features_por_modo(FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL)
            == TIME_FREQUENCY_RELATIVE_FEATURE_NAMES
        )

    def test_normalizacao_robusta_remove_baseline_do_edf(self):
        matriz = np.asarray(
            [[10.0, 100.0], [12.0, 100.0], [14.0, 100.0], [1000.0, 100.0]],
            dtype=np.float32,
        )

        normalizada = normalizar_matriz_features_robusta(matriz)

        assert np.median(normalizada[:, 0]) == pytest.approx(0.0)
        assert normalizada[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert np.max(np.abs(normalizada)) <= 10.0

    def test_normalizacao_robusta_aceita_referencia_normal_separada(self):
        normais = np.asarray([[9.0], [10.0], [11.0]], dtype=np.float32)
        todas = np.asarray([[9.0], [10.0], [11.0], [100.0]], dtype=np.float32)

        normalizada = normalizar_matriz_features_robusta(todas, referencia=normais)

        assert normalizada[1, 0] == pytest.approx(0.0)
        assert normalizada[-1, 0] == pytest.approx(10.0)


class TestExtrairFeaturesEdf:
    @pytest.fixture
    def edf_sintetico(self, tmp_path) -> tuple[Path, np.ndarray, float]:
        sfreq = 256.0
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


def test_selecionar_canais_eeg_validos_descarta_placeholders():
    info = mne.create_info(
        ch_names=["FP1-F7", "-", "T8-P8", "--0"],
        sfreq=256.0,
        ch_types=["eeg", "eeg", "eeg", "eeg"],
    )
    raw = mne.io.RawArray(np.zeros((4, 64)), info, verbose=False)

    canais = selecionar_canais_eeg_validos(raw)

    assert canais == ["FP1-F7", "T8-P8"]
