"""
Extrator de features EEG baseado em dinamica simbolica e estatisticas.

Carrega arquivos .edf via mne-python e produz as 19 features do pipeline
legado para alimentar a rede CNN-LSTM hibrida.
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

FEATURE_MODE_MEAN = "mean"
FEATURE_MODE_PER_CHANNEL = "per_channel"

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

MAX_DURATION_SECONDS: float | None = None


def _canal_eeg_valido(nome: str) -> bool:
    nome_limpo = nome.strip()
    return bool(nome_limpo) and nome_limpo != "-" and not nome_limpo.startswith("--")


def selecionar_canais_eeg_validos(
    raw: mne.io.BaseRaw,
    canais_selecionados: list[str] | None = None,
) -> list[str]:
    raw.pick("eeg")
    canais_validos = [canal for canal in raw.ch_names if _canal_eeg_valido(canal)]
    if not canais_validos:
        raise ValueError("Nenhum canal EEG valido encontrado no arquivo EDF.")

    if canais_selecionados:
        invalidos = [canal for canal in canais_selecionados if canal not in canais_validos]
        if invalidos:
            raise ValueError(f"Canais EEG invalidos: {', '.join(invalidos)}")
        return canais_selecionados

    return canais_validos


def _calcular_skewness(data: np.ndarray) -> float:
    n = len(data)
    if n < 3:
        return 0.0
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3))


def _calcular_kurtosis(data: np.ndarray) -> float:
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
    if len(sequencia) < 2:
        return 0
    return int(np.sum(sequencia[1:] != sequencia[:-1]))


def _calcular_entropia_frequencias(valores: list[float]) -> float:
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
    path = Path(arquivo_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo EDF nao encontrado: {path}")
    if path.suffix.lower() != ".edf":
        raise ValueError(f"Extensao invalida (esperado .edf): {path.suffix}")

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    raw.pick(selecionar_canais_eeg_validos(raw))

    if max_duration_seconds is not None and max_duration_seconds > 0:
        duracao = float(raw.times[-1])
        if duracao > max_duration_seconds:
            raw.crop(tmax=max_duration_seconds)

    return raw


def listar_canais_eeg_edf(
    arquivo_path: str | Path,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
) -> list[str]:
    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    return list(raw.ch_names)


def extrair_metadados_edf(arquivo_path: str | Path) -> dict[str, object]:
    path = Path(arquivo_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo EDF nao encontrado: {path}")
    if path.suffix.lower() != ".edf":
        raise ValueError(f"Extensao invalida (esperado .edf): {path.suffix}")

    raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
    canais_eeg = selecionar_canais_eeg_validos(raw)

    return {
        "taxa_amostragem": float(raw.info["sfreq"]),
        "canais_eeg": canais_eeg,
    }


def carregar_sinal_edf(
    arquivo_path: str | Path,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
) -> tuple[np.ndarray, float, int]:
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
    resultado = aplicar_dinamica_simbolica(valores, m=m)
    if not resultado["sequencia_binaria"]:
        raise ValueError("Sequencia binaria vazia apos dinamica simbolica.")

    sinal = np.asarray(valores, dtype=float).ravel()
    sequencia_binaria = np.array([int(b) for b in resultado["sequencia_binaria"]])
    freq_values = list(resultado["frequencias"].values())
    if not freq_values:
        raise ValueError("Frequencias de padroes vazias.")

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
    feature_mode: str = FEATURE_MODE_MEAN,
    canais_referencia: list[str] | None = None,
) -> dict[str, Any]:
    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    canais_disponiveis = list(raw.ch_names)
    canais_validos = selecionar_canais_eeg_validos(raw)

    if canais_selecionados is not None:
        invalidos = [canal for canal in canais_selecionados if canal not in canais_validos]
        if invalidos:
            raise ValueError(f"Canais EEG invalidos: {', '.join(invalidos)}")
        canais_usuario = list(canais_selecionados)
    else:
        canais_usuario = list(canais_validos)

    if feature_mode == FEATURE_MODE_MEAN:
        alvos = canais_usuario
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
        features["feature_names"] = FEATURE_NAMES
        canais_omitidos: list[str] = []
    elif feature_mode == FEATURE_MODE_PER_CHANNEL:
        referencia = list(canais_referencia or canais_usuario)
        if not referencia:
            raise ValueError("Nenhum canal de referencia disponivel para inferencia per_channel.")

        vetores_expandidos: list[float] = []
        nomes_expandidos: list[str] = []
        canais_processados: list[str] = []
        canais_omitidos = []

        for canal in referencia:
            nomes_expandidos.extend([f"{canal}::{nome}" for nome in FEATURE_NAMES])
            if canal not in canais_usuario or canal not in canais_validos:
                vetores_expandidos.extend([0.0] * len(FEATURE_NAMES))
                canais_omitidos.append(canal)
                continue

            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx])[0], dtype=float)
            feat = extrair_features_de_valores(sinal_canal, m=m)
            vetores_expandidos.extend(float(v) for v in feat["feature_vector"])
            canais_processados.append(canal)

        features = {
            "feature_names": nomes_expandidos,
            "feature_vector": vetores_expandidos,
        }
        alvos = canais_processados
    else:
        raise ValueError(f"feature_mode invalido: {feature_mode}")

    features["arquivo_path"] = str(Path(arquivo_path).resolve())
    features["taxa_amostragem"] = float(raw.info["sfreq"])
    features["n_canais_eeg"] = len(alvos)
    features["canais_processados"] = list(alvos)
    features["canais_omitidos"] = canais_omitidos
    features["feature_mode"] = feature_mode
    features["canais_referencia"] = list(canais_referencia or alvos)
    features["total_amostras_brutas"] = int(raw.n_times)
    return features


def _montar_features_de_janela(
    *,
    raw: mne.io.BaseRaw,
    canais_usuario: list[str],
    canais_validos: list[str],
    canais_referencia: list[str] | None,
    inicio: int,
    fim: int,
    feature_mode: str,
    m: int,
) -> dict[str, Any]:
    canais_disponiveis = list(raw.ch_names)

    if feature_mode == FEATURE_MODE_MEAN:
        alvos = canais_usuario
        vetores: list[list[float]] = []
        for canal in alvos:
            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx], start=inicio, stop=fim)[0], dtype=float)
            feat = extrair_features_de_valores(sinal_canal, m=m)
            vetores.append(feat["feature_vector"])

        vetor_medio = np.mean(np.array(vetores), axis=0)
        features: dict[str, Any] = {
            name: float(vetor_medio[i])
            for i, name in enumerate(FEATURE_NAMES)
        }
        features["feature_names"] = FEATURE_NAMES
        features["feature_vector"] = [float(v) for v in vetor_medio]
        canais_processados = list(alvos)
        canais_omitidos: list[str] = []
    elif feature_mode == FEATURE_MODE_PER_CHANNEL:
        referencia = list(canais_referencia or canais_usuario)
        if not referencia:
            raise ValueError("Nenhum canal de referencia disponivel para inferencia per_channel.")

        vetores_expandidos: list[float] = []
        nomes_expandidos: list[str] = []
        canais_processados = []
        canais_omitidos = []

        for canal in referencia:
            nomes_expandidos.extend([f"{canal}::{nome}" for nome in FEATURE_NAMES])
            if canal not in canais_usuario or canal not in canais_validos:
                vetores_expandidos.extend([0.0] * len(FEATURE_NAMES))
                canais_omitidos.append(canal)
                continue

            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx], start=inicio, stop=fim)[0], dtype=float)
            feat = extrair_features_de_valores(sinal_canal, m=m)
            vetores_expandidos.extend(float(v) for v in feat["feature_vector"])
            canais_processados.append(canal)

        features = {
            "feature_names": nomes_expandidos,
            "feature_vector": vetores_expandidos,
        }
    else:
        raise ValueError(f"feature_mode invalido: {feature_mode}")

    features["taxa_amostragem"] = float(raw.info["sfreq"])
    features["n_canais_eeg"] = len(canais_processados)
    features["canais_processados"] = list(canais_processados)
    features["canais_omitidos"] = canais_omitidos
    features["feature_mode"] = feature_mode
    features["canais_referencia"] = list(canais_referencia or canais_processados)
    features["total_amostras_brutas"] = int(raw.n_times)
    return features


def extrair_features_edf_janelado(
    arquivo_path: str,
    m: int = SYMBOLIC_M_DEFAULT,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
    canais_selecionados: list[str] | None = None,
    feature_mode: str = FEATURE_MODE_MEAN,
    canais_referencia: list[str] | None = None,
    window_seconds: float = 4.0,
    step_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Extrai features em janelas temporais, alinhado ao treino dos modelos."""
    if window_seconds <= 0:
        raise ValueError("window_seconds deve ser maior que zero.")
    if step_seconds <= 0:
        raise ValueError("step_seconds deve ser maior que zero.")

    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    canais_validos = list(raw.ch_names)

    if canais_selecionados is not None:
        invalidos = [canal for canal in canais_selecionados if canal not in canais_validos]
        if invalidos:
            raise ValueError(f"Canais EEG invalidos: {', '.join(invalidos)}")
        canais_usuario = list(canais_selecionados)
    else:
        canais_usuario = list(canais_validos)

    sfreq = float(raw.info["sfreq"])
    janela_amostras = int(round(window_seconds * sfreq))
    passo_amostras = int(round(step_seconds * sfreq))
    if raw.n_times < janela_amostras:
        return [
            _montar_features_de_janela(
                raw=raw,
                canais_usuario=canais_usuario,
                canais_validos=canais_validos,
                canais_referencia=canais_referencia,
                inicio=0,
                fim=raw.n_times,
                feature_mode=feature_mode,
                m=m,
            )
        ]

    janelas: list[dict[str, Any]] = []
    inicio = 0
    while inicio + janela_amostras <= raw.n_times:
        fim = inicio + janela_amostras
        features = _montar_features_de_janela(
            raw=raw,
            canais_usuario=canais_usuario,
            canais_validos=canais_validos,
            canais_referencia=canais_referencia,
            inicio=inicio,
            fim=fim,
            feature_mode=feature_mode,
            m=m,
        )
        features["arquivo_path"] = str(Path(arquivo_path).resolve())
        features["window_start_seconds"] = float(inicio / sfreq)
        features["window_end_seconds"] = float(fim / sfreq)
        janelas.append(features)
        inicio += passo_amostras

    return janelas


class FeatureExtractor:
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
