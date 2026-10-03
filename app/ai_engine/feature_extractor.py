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
FEATURE_MODE_TIME_FREQUENCY = "time_frequency"
FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL = "time_frequency_per_channel"
FEATURE_MODE_TIME_FREQUENCY_RELATIVE = "time_frequency_relative"
FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL = "time_frequency_relative_per_channel"
FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL = "time_frequency_morphology_per_channel"
FEATURE_MODE_RAW_SIGNAL = "raw_signal"

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
TIME_FREQUENCY_BANDS: list[tuple[str, float, float]] = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 45.0),
]
TIME_FREQUENCY_FEATURE_NAMES: list[str] = [
    *(f"potencia_{nome}" for nome, _low, _high in TIME_FREQUENCY_BANDS),
    *(f"potencia_relativa_{nome}" for nome, _low, _high in TIME_FREQUENCY_BANDS),
    "frequencia_dominante",
    "centroide_espectral",
    "entropia_espectral",
    "razao_theta_alpha",
    "razao_beta_alpha",
]
TIME_FREQUENCY_RELATIVE_FEATURE_NAMES: list[str] = [
    *(f"potencia_relativa_{nome}" for nome, _low, _high in TIME_FREQUENCY_BANDS),
    "frequencia_dominante_normalizada",
    "centroide_espectral_normalizado",
    "entropia_espectral_normalizada",
    "log_razao_theta_alpha",
    "log_razao_beta_alpha",
]
MORPHOLOGY_FEATURE_NAMES: list[str] = [
    "line_length_normalized",
    "hjorth_mobility",
    "hjorth_complexity",
    "zero_crossing_rate",
    "crest_factor",
    "skewness_normalized",
    "kurtosis_normalized",
    "symbolic_entropy",
]
TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES = (
    TIME_FREQUENCY_FEATURE_NAMES + MORPHOLOGY_FEATURE_NAMES
)
RAW_SIGNAL_POINTS = 128
RAW_SIGNAL_FEATURE_NAMES: list[str] = [f"raw_{idx:03d}" for idx in range(RAW_SIGNAL_POINTS)]


def normalizar_matriz_features_robusta(
    valores: np.ndarray,
    *,
    referencia: np.ndarray | None = None,
    clip: float = 10.0,
) -> np.ndarray:
    """Centraliza features pelo EDF usando mediana e intervalo interquartil."""
    matriz = np.asarray(valores, dtype=np.float32)
    if matriz.ndim != 2:
        raise ValueError("valores deve ter shape (n_janelas, n_features).")
    if matriz.shape[0] == 0:
        return matriz.copy()

    base = matriz if referencia is None else np.asarray(referencia, dtype=np.float32)
    if base.ndim != 2 or base.shape[1] != matriz.shape[1] or base.shape[0] == 0:
        raise ValueError("referencia deve ter shape (n_referencias, n_features).")

    mediana = np.median(base, axis=0)
    q25, q75 = np.percentile(base, [25.0, 75.0], axis=0)
    escala = q75 - q25
    escala = np.where(escala > 1e-8, escala, 1.0)
    normalizada = (matriz - mediana) / escala
    return np.clip(normalizada, -abs(clip), abs(clip)).astype(np.float32)


def nomes_features_por_modo(feature_mode: str) -> list[str]:
    if feature_mode in {FEATURE_MODE_MEAN, FEATURE_MODE_PER_CHANNEL}:
        return FEATURE_NAMES
    if feature_mode in {FEATURE_MODE_TIME_FREQUENCY, FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL}:
        return TIME_FREQUENCY_FEATURE_NAMES
    if feature_mode == FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL:
        return TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES
    if feature_mode in {
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    }:
        return TIME_FREQUENCY_RELATIVE_FEATURE_NAMES
    if feature_mode == FEATURE_MODE_RAW_SIGNAL:
        return RAW_SIGNAL_FEATURE_NAMES
    raise ValueError(f"feature_mode invalido: {feature_mode}")

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
    sinais, taxa_amostragem, canais = carregar_sinais_edf(
        arquivo_path,
        max_duration_seconds=max_duration_seconds,
    )
    sinal = np.mean(sinais, axis=0)
    return np.asarray(sinal, dtype=float), taxa_amostragem, len(canais)


def carregar_sinais_edf(
    arquivo_path: str | Path,
    max_duration_seconds: float | None = MAX_DURATION_SECONDS,
    canais_selecionados: list[str] | None = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Carrega a matriz EEG preservando os canais para visualizacao clinica."""
    raw = _preparar_raw_edf(arquivo_path, max_duration_seconds=max_duration_seconds)
    if canais_selecionados:
        invalidos = [canal for canal in canais_selecionados if canal not in raw.ch_names]
        if invalidos:
            raise ValueError(f"Canais EEG invalidos: {', '.join(invalidos)}")
        raw.pick(canais_selecionados)
    dados = np.asarray(raw.get_data(), dtype=float)
    taxa_amostragem = float(raw.info["sfreq"])
    return dados, taxa_amostragem, list(raw.ch_names)


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


def extrair_features_tempo_frequencia_de_valores(
    valores: np.ndarray,
    sfreq: float,
) -> dict[str, Any]:
    sinal = np.asarray(valores, dtype=float).ravel()
    if sinal.size < 4:
        raise ValueError("Sinal curto demais para features tempo-frequencia.")

    sinal = sinal - float(np.mean(sinal))
    janela = np.hanning(sinal.size)
    espectro = np.fft.rfft(sinal * janela)
    frequencias = np.fft.rfftfreq(sinal.size, d=1.0 / sfreq)
    psd = np.abs(espectro) ** 2
    mascara_util = frequencias > 0
    total_power = float(np.sum(psd[mascara_util]))
    if total_power <= 0:
        total_power = 1e-12

    features: dict[str, Any] = {}
    band_powers: dict[str, float] = {}
    for nome, low, high in TIME_FREQUENCY_BANDS:
        mascara = (frequencias >= low) & (frequencias < high)
        power = float(np.sum(psd[mascara]))
        band_powers[nome] = power
        features[f"potencia_{nome}"] = float(np.log1p(power))
    for nome, _low, _high in TIME_FREQUENCY_BANDS:
        features[f"potencia_relativa_{nome}"] = float(band_powers[nome] / total_power)

    idx_dom = int(np.argmax(psd[mascara_util])) if np.any(mascara_util) else 0
    freqs_util = frequencias[mascara_util]
    psd_util = psd[mascara_util]
    probs = psd_util / total_power
    probs = probs[probs > 0]
    features["frequencia_dominante"] = float(freqs_util[idx_dom]) if freqs_util.size else 0.0
    features["centroide_espectral"] = float(np.sum(freqs_util * psd_util) / total_power) if freqs_util.size else 0.0
    features["entropia_espectral"] = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    features["razao_theta_alpha"] = float(band_powers["theta"] / max(band_powers["alpha"], 1e-12))
    features["razao_beta_alpha"] = float(band_powers["beta"] / max(band_powers["alpha"], 1e-12))
    features["feature_names"] = TIME_FREQUENCY_FEATURE_NAMES
    features["feature_vector"] = [float(features[name]) for name in TIME_FREQUENCY_FEATURE_NAMES]
    return features


def extrair_features_tempo_frequencia_relativas_de_valores(
    valores: np.ndarray,
    sfreq: float,
) -> dict[str, Any]:
    """Features espectrais invariantes a escala de amplitude do sinal."""
    sinal = np.asarray(valores, dtype=float).ravel()
    base = extrair_features_tempo_frequencia_de_valores(sinal, sfreq=sfreq)
    nyquist = max(float(sfreq) / 2.0, 1e-12)
    n_bins_uteis = max(2, int(np.fft.rfftfreq(sinal.size, d=1.0 / sfreq)[1:].size))

    features: dict[str, Any] = {
        f"potencia_relativa_{nome}": float(base[f"potencia_relativa_{nome}"])
        for nome, _low, _high in TIME_FREQUENCY_BANDS
    }
    features["frequencia_dominante_normalizada"] = float(base["frequencia_dominante"] / nyquist)
    features["centroide_espectral_normalizado"] = float(base["centroide_espectral"] / nyquist)
    features["entropia_espectral_normalizada"] = float(
        base["entropia_espectral"] / np.log2(n_bins_uteis)
    )
    features["log_razao_theta_alpha"] = float(
        np.clip(np.log(max(float(base["razao_theta_alpha"]), 1e-12)), -12.0, 12.0)
    )
    features["log_razao_beta_alpha"] = float(
        np.clip(np.log(max(float(base["razao_beta_alpha"]), 1e-12)), -12.0, 12.0)
    )
    features["feature_names"] = TIME_FREQUENCY_RELATIVE_FEATURE_NAMES
    features["feature_vector"] = [
        float(features[name]) for name in TIME_FREQUENCY_RELATIVE_FEATURE_NAMES
    ]
    return features


def extrair_features_morfologicas_de_valores(
    valores: np.ndarray,
    *,
    m: int = SYMBOLIC_M_DEFAULT,
) -> dict[str, Any]:
    """Extract amplitude-invariant waveform morphology descriptors."""
    signal = np.asarray(valores, dtype=float).ravel()
    if signal.size < 4:
        raise ValueError("Sinal curto demais para features morfologicas.")
    centered = signal - float(np.mean(signal))
    scale = max(float(np.std(centered)), 1e-12)
    normalized = centered / scale
    first = np.diff(normalized)
    second = np.diff(first)
    var_signal = max(float(np.var(normalized)), 1e-12)
    var_first = max(float(np.var(first)), 1e-12)
    mobility = float(np.sqrt(var_first / var_signal))
    mobility_first = float(np.sqrt(max(float(np.var(second)), 0.0) / var_first))
    symbolic = aplicar_dinamica_simbolica(normalized, m=m)
    features: dict[str, float | list[str] | list[float]] = {
        "line_length_normalized": float(np.mean(np.abs(first))),
        "hjorth_mobility": mobility,
        "hjorth_complexity": float(mobility_first / max(mobility, 1e-12)),
        "zero_crossing_rate": float(np.mean(normalized[1:] * normalized[:-1] < 0)),
        "crest_factor": float(np.max(np.abs(normalized))),
        "skewness_normalized": _calcular_skewness(normalized),
        "kurtosis_normalized": _calcular_kurtosis(normalized),
        "symbolic_entropy": float(symbolic["entropia"]),
    }
    features["feature_names"] = MORPHOLOGY_FEATURE_NAMES
    features["feature_vector"] = [float(features[name]) for name in MORPHOLOGY_FEATURE_NAMES]
    return features


def extrair_features_tempo_frequencia_morfologia_de_valores(
    valores: np.ndarray,
    *,
    sfreq: float,
    m: int = SYMBOLIC_M_DEFAULT,
) -> dict[str, Any]:
    spectral = extrair_features_tempo_frequencia_de_valores(valores, sfreq=sfreq)
    morphology = extrair_features_morfologicas_de_valores(valores, m=m)
    merged = {
        name: float(value)
        for name, value in zip(spectral["feature_names"], spectral["feature_vector"])
    }
    merged.update({
        name: float(value)
        for name, value in zip(morphology["feature_names"], morphology["feature_vector"])
    })
    merged["feature_names"] = TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES
    merged["feature_vector"] = [
        float(merged[name]) for name in TIME_FREQUENCY_MORPHOLOGY_FEATURE_NAMES
    ]
    return merged


def extrair_features_sinal_bruto_de_valores(valores: np.ndarray) -> dict[str, Any]:
    sinal = np.asarray(valores, dtype=float).ravel()
    if sinal.size < 2:
        raise ValueError("Sinal curto demais para vetor bruto.")

    origem = np.linspace(0.0, 1.0, sinal.size)
    destino = np.linspace(0.0, 1.0, RAW_SIGNAL_POINTS)
    bruto = np.interp(destino, origem, sinal)
    desvio = float(np.std(bruto))
    if desvio > 0:
        bruto = (bruto - float(np.mean(bruto))) / desvio
    else:
        bruto = bruto * 0.0
    features = {
        "feature_names": RAW_SIGNAL_FEATURE_NAMES,
        "feature_vector": [float(v) for v in bruto],
    }
    return features


def extrair_features_por_modo_de_valores(
    valores: np.ndarray,
    *,
    feature_mode: str,
    sfreq: float,
    m: int = SYMBOLIC_M_DEFAULT,
) -> dict[str, Any]:
    if feature_mode in {FEATURE_MODE_MEAN, FEATURE_MODE_PER_CHANNEL}:
        return extrair_features_de_valores(valores, m=m)
    if feature_mode in {FEATURE_MODE_TIME_FREQUENCY, FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL}:
        return extrair_features_tempo_frequencia_de_valores(valores, sfreq=sfreq)
    if feature_mode in {
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    }:
        return extrair_features_tempo_frequencia_relativas_de_valores(valores, sfreq=sfreq)
    if feature_mode == FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL:
        return extrair_features_tempo_frequencia_morfologia_de_valores(
            valores, sfreq=sfreq, m=m,
        )
    if feature_mode == FEATURE_MODE_RAW_SIGNAL:
        return extrair_features_sinal_bruto_de_valores(valores)
    raise ValueError(f"feature_mode invalido: {feature_mode}")


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

    if feature_mode in {
        FEATURE_MODE_MEAN,
        FEATURE_MODE_TIME_FREQUENCY,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
        FEATURE_MODE_RAW_SIGNAL,
    }:
        alvos = canais_usuario
        vetores: list[list[float]] = []
        for canal in alvos:
            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx])[0], dtype=float)
            feat = extrair_features_por_modo_de_valores(
                sinal_canal,
                feature_mode=feature_mode,
                sfreq=float(raw.info["sfreq"]),
                m=m,
            )
            vetores.append(feat["feature_vector"])

        vetor_medio = np.mean(np.array(vetores), axis=0)
        nomes_features = nomes_features_por_modo(feature_mode)
        features = {
            "feature_names": nomes_features,
            "feature_vector": [float(v) for v in vetor_medio],
        }
        for i, name in enumerate(nomes_features):
            features[name] = float(vetor_medio[i])
        canais_omitidos: list[str] = []
    elif feature_mode in {
        FEATURE_MODE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
    }:
        referencia = list(canais_referencia or canais_usuario)
        if not referencia:
            raise ValueError("Nenhum canal de referencia disponivel para inferencia por canal.")

        vetores_expandidos: list[float] = []
        nomes_expandidos: list[str] = []
        canais_processados: list[str] = []
        canais_omitidos = []
        nomes_base = nomes_features_por_modo(feature_mode)

        for canal in referencia:
            nomes_expandidos.extend([f"{canal}::{nome}" for nome in nomes_base])
            if canal not in canais_usuario or canal not in canais_validos:
                vetores_expandidos.extend([0.0] * len(nomes_base))
                canais_omitidos.append(canal)
                continue

            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx])[0], dtype=float)
            feat = extrair_features_por_modo_de_valores(
                sinal_canal,
                feature_mode=feature_mode,
                sfreq=float(raw.info["sfreq"]),
                m=m,
            )
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

    if feature_mode in {
        FEATURE_MODE_MEAN,
        FEATURE_MODE_TIME_FREQUENCY,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
        FEATURE_MODE_RAW_SIGNAL,
    }:
        alvos = canais_usuario
        vetores: list[list[float]] = []
        for canal in alvos:
            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx], start=inicio, stop=fim)[0], dtype=float)
            feat = extrair_features_por_modo_de_valores(
                sinal_canal,
                feature_mode=feature_mode,
                sfreq=float(raw.info["sfreq"]),
                m=m,
            )
            vetores.append(feat["feature_vector"])

        vetor_medio = np.mean(np.array(vetores), axis=0)
        nomes_features = nomes_features_por_modo(feature_mode)
        features: dict[str, Any] = {
            name: float(vetor_medio[i])
            for i, name in enumerate(nomes_features)
        }
        features["feature_names"] = nomes_features
        features["feature_vector"] = [float(v) for v in vetor_medio]
        canais_processados = list(alvos)
        canais_omitidos: list[str] = []
    elif feature_mode in {
        FEATURE_MODE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
    }:
        referencia = list(canais_referencia or canais_usuario)
        if not referencia:
            raise ValueError("Nenhum canal de referencia disponivel para inferencia por canal.")

        vetores_expandidos: list[float] = []
        nomes_expandidos: list[str] = []
        canais_processados = []
        canais_omitidos = []
        nomes_base = nomes_features_por_modo(feature_mode)

        for canal in referencia:
            nomes_expandidos.extend([f"{canal}::{nome}" for nome in nomes_base])
            if canal not in canais_usuario or canal not in canais_validos:
                vetores_expandidos.extend([0.0] * len(nomes_base))
                canais_omitidos.append(canal)
                continue

            idx = canais_disponiveis.index(canal)
            sinal_canal = np.asarray(raw.get_data(picks=[idx], start=inicio, stop=fim)[0], dtype=float)
            feat = extrair_features_por_modo_de_valores(
                sinal_canal,
                feature_mode=feature_mode,
                sfreq=float(raw.info["sfreq"]),
                m=m,
            )
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
        features = _montar_features_de_janela(
            raw=raw,
            canais_usuario=canais_usuario,
            canais_validos=canais_validos,
            canais_referencia=canais_referencia,
            inicio=0,
            fim=raw.n_times,
            feature_mode=feature_mode,
            m=m,
        )
        features["window_start_seconds"] = 0.0
        features["window_end_seconds"] = float(raw.n_times / sfreq)
        return [features]

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
