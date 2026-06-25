"""
Extrator de features EEG — dinâmica simbólica + estatísticas (porte PIBITI).

Carrega arquivos .edf reais via mne-python (canais EEG) e produz as 19 features
do ml_classifier legado para alimentar a rede CNN-LSTM híbrida.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np

from app.ai_engine.symbolic_dynamics import (
    SYMBOLIC_M_DEFAULT,
    aplicar_dinamica_simbolica,
)

# Ordem estável para alimentar a rede neural híbrida
FEATURE_NAMES: list[str] = [
    "entropia_shannon",
    "limiar",
    "total_amostras",
    "total_padroes",
    "padroes_unicos",
    "media_valores",
    "desvio_padrao",
    "variancia",
    "skewness",
    "kurtosis",
    "amplitude",
    "rms",
    "proporcao_uns",
    "transicoes",
    "comprimento_sequencia",
    "max_frequencia",
    "min_frequencia",
    "std_frequencias",
    "entropia_frequencias",
]

# Limite opcional para gravações longas (evita pico de memória em exames de horas)
MAX_DURATION_SECONDS: float | None = None


def _calcular_skewness(data: np.ndarray) -> float:
    """Assimetria — porte de ml_classifier.py."""
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3))


def _calcular_kurtosis(data: np.ndarray) -> float:
    """Curtose — porte de ml_classifier.py."""
    n = len(data)
    if n < 4:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float(
        (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(((data - mean) / std) ** 4)
        - (3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    )


def _contar_transicoes(sequencia: np.ndarray) -> int:
    """Conta transições 0→1 e 1→0 na sequência binária."""
    if len(sequencia) < 2:
        return 0
    return int(np.sum(sequencia[1:] != sequencia[:-1]))


def _calcular_entropia_frequencias(valores: list[float]) -> float:
    """
    Entropia das frequências dos padrões (log₂) — porte de EEGClassifier._calcular_entropia_shannon.
    """
    if not valores:
        return 0.0
    total = sum(valores)
    if total == 0:
        return 0.0
    entropia = 0.0
    for valor in valores:
        p = valor / total
        if p > 0:
            entropia -= p * np.log2(p)
    return float(entropia)


def _preparar_raw_edf(
    arquivo_path: str | Path,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
) -> mne.io.BaseRaw:
    """Carrega .edf, filtra canais EEG e aplica recorte temporal opcional."""
    path = Path(arquivo_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo EDF não encontrado: {path}")
    if path.suffix.lower() != ".edf":
        raise ValueError(f"Extensão inválida (esperado .edf): {path.suffix}")

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick_types(eeg=True)

    if len(raw.ch_names) == 0:
        raise ValueError(f"Nenhum canal EEG encontrado em {path}")

    if max_duration_seconds is not None and max_duration_seconds > 0:
        duracao = float(raw.times[-1])
        if duracao > max_duration_seconds:
            raw.crop(tmax=max_duration_seconds)

    return raw


def listar_canais_eeg_edf(
    arquivo_path: str | Path,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
) -> list[str]:
    """Retorna os nomes dos canais EEG presentes no arquivo .edf."""
    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    return list(raw.ch_names)


def carregar_sinal_edf(
    arquivo_path: str | Path,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
) -> tuple[np.ndarray, float, int]:
    """
    Carrega canais EEG de um .edf e retorna sinal 1D (média entre canais).

    Returns:
        tuple: (sinal_1d, taxa_amostragem_hz, n_canais_eeg)
    """
    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    n_canais = len(raw.ch_names)
    dados = raw.get_data()
    sinal = np.mean(dados, axis=0)
    taxa_amostragem = float(raw.info["sfreq"])

    return np.asarray(sinal, dtype=float), taxa_amostragem, n_canais


def extrair_features_de_valores(
    valores: np.ndarray,
    m: int = SYMBOLIC_M_DEFAULT,
) -> dict[str, Any]:
    """
    Extrai as 19 features a partir de um array 1D (útil para testes e pipelines internos).
    """
    resultado = aplicar_dinamica_simbolica(valores, m=m)
    if not resultado["sequencia_binaria"]:
        raise ValueError("Sequência binária vazia após dinâmica simbólica.")

    sinal = np.asarray(valores, dtype=float).ravel()
    sequencia_binaria = np.array([int(b) for b in resultado["sequencia_binaria"]])
    freq_values = list(resultado["frequencias"].values())
    if not freq_values:
        raise ValueError("Frequências de padrões vazias.")

    features: dict[str, Any] = {
        "entropia_shannon": resultado["entropia"],
        "limiar": resultado["limiar"],
        "total_amostras": len(resultado["sequencia_binaria"]),
        "total_padroes": len(resultado["palavras_decimais"]),
        "padroes_unicos": len(resultado["frequencias"]),
        "media_valores": float(np.mean(sinal)),
        "desvio_padrao": float(np.std(sinal)),
        "variancia": float(np.var(sinal)),
        "skewness": _calcular_skewness(sinal),
        "kurtosis": _calcular_kurtosis(sinal),
        "amplitude": float(np.max(sinal) - np.min(sinal)),
        "rms": float(np.sqrt(np.mean(sinal**2))),
        "proporcao_uns": float(np.mean(sequencia_binaria)),
        "transicoes": _contar_transicoes(sequencia_binaria),
        "comprimento_sequencia": len(sequencia_binaria),
        "max_frequencia": float(np.max(freq_values)),
        "min_frequencia": float(np.min(freq_values)),
        "std_frequencias": float(np.std(freq_values)),
        "entropia_frequencias": _calcular_entropia_frequencias(freq_values),
    }

    features["feature_names"] = FEATURE_NAMES
    features["feature_vector"] = [features[name] for name in FEATURE_NAMES]
    return features


def extrair_features_edf(
    arquivo_path: str,
    m: int = SYMBOLIC_M_DEFAULT,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
    canais_selecionados: list[str] | None = None,
) -> dict[str, Any]:
    """
    Carrega um .edf real com mne e retorna dicionário consolidado de features.

    Para múltiplos canais, extrai as 19 features de cada canal e retorna a média
    (compatível com a CNN-LSTM atual que espera vetor único de 19 dimensões).
    """
    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    canais_disponiveis = list(raw.ch_names)

    alvos = canais_selecionados if canais_selecionados else canais_disponiveis
    invalidos = [c for c in alvos if c not in canais_disponiveis]
    if invalidos:
        raise ValueError(f"Canais EEG inválidos: {', '.join(invalidos)}")
    if not alvos:
        raise ValueError("Nenhum canal EEG selecionado para extração.")

    vetores: list[list[float]] = []
    for canal in alvos:
        idx = canais_disponiveis.index(canal)
        sinal_canal = np.asarray(raw.get_data(picks=[idx])[0], dtype=float)
        feat = extrair_features_de_valores(sinal_canal, m=m)
        vetores.append(feat["feature_vector"])

    vetor_medio = np.mean(np.array(vetores), axis=0)
    features = extrair_features_de_valores(
        np.asarray(raw.get_data(picks=[canais_disponiveis.index(alvos[0])])[0], dtype=float),
        m=m,
    )
    for i, name in enumerate(FEATURE_NAMES):
        features[name] = float(vetor_medio[i])
    features["feature_vector"] = [float(v) for v in vetor_medio]

    features["arquivo_path"] = str(Path(arquivo_path).resolve())
    features["taxa_amostragem"] = float(raw.info["sfreq"])
    features["n_canais_eeg"] = len(alvos)
    features["canais_processados"] = alvos
    features["total_amostras_brutas"] = int(raw.n_times)
    return features


class FeatureExtractor:
    """Wrapper orientado a objetos para o pipeline de features."""

    def __init__(
        self,
        symbolic_m: int = SYMBOLIC_M_DEFAULT,
        max_duration_seconds: float | None = MAX_DURATION_SECONDS,
    ) -> None:
        self.symbolic_m = symbolic_m
        self.max_duration_seconds = max_duration_seconds

    def extract_from_edf(self, edf_path: Path | str) -> dict[str, Any]:
        return extrair_features_edf(
            str(edf_path),
            m=self.symbolic_m,
            max_duration_seconds=self.max_duration_seconds,
        )

    def extract_from_array(self, valores: np.ndarray) -> dict[str, Any]:
        return extrair_features_de_valores(valores, m=self.symbolic_m)
