"""
Dinâmica simbólica e entropia de Shannon — porte fiel do sistema legado PIBITI.

Referência: eeg-neural-network/dinamica_simbolica.py
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

SYMBOLIC_M_DEFAULT = 3


def obter_limiar(sinal: np.ndarray) -> float:
    """Limiar = média aritmética do sinal (legado: AVG no SQL)."""
    if sinal.size == 0:
        raise ValueError("Sinal vazio: não é possível calcular o limiar.")
    return float(np.mean(sinal))


def gerar_sequencia_binaria(sinal: np.ndarray, limiar: float) -> list[str]:
    """Gera sequência de '0'/'1' conforme comparação com o limiar (>= limiar → '1')."""
    return ["1" if float(x) >= limiar else "0" for x in sinal]


def gerar_grupos_deslizantes(sequencia: Sequence[str], m: int = SYMBOLIC_M_DEFAULT) -> list[str]:
    """Janelas deslizantes de tamanho m sobre a sequência binária."""
    if len(sequencia) < m:
        return []
    return ["".join(sequencia[i : i + m]) for i in range(len(sequencia) - m + 1)]


def converter_para_decimal(grupos: Sequence[str]) -> list[int]:
    """Converte grupos binários em valores decimais (0–7 para m=3)."""
    if not grupos:
        return []
    return [int(grupo, 2) for grupo in grupos]


def calcular_frequencia(palavras_decimais: Sequence[int]) -> dict[int, float]:
    """Frequência relativa de cada padrão decimal."""
    if not palavras_decimais:
        return {}
    contagem = Counter(palavras_decimais)
    total = sum(contagem.values())
    return {k: v / total for k, v in contagem.items()}


def calcular_entropia_shannon(frequencias: dict[int, float]) -> float:
    """
    Entropia de Shannon normalizada (ln), entre 0 e 1.

    Porte exato de dinamica_simbolica.py — ignora probabilidades 0 ou 1.
    """
    if not frequencias:
        return 0.0

    probabilidades = np.array(list(frequencias.values()), dtype=float)
    probabilidades_filtradas = probabilidades[(probabilidades > 0) & (probabilidades < 1)]
    if len(probabilidades_filtradas) == 0:
        return 0.0

    probabilidades_norm = probabilidades_filtradas / np.sum(probabilidades_filtradas)
    entropia_bruta = -np.sum(probabilidades_norm * np.log(probabilidades_norm))
    n_simbolos = len(probabilidades_filtradas)
    if n_simbolos > 1:
        entropia_maxima = np.log(n_simbolos)
        entropia_normalizada = entropia_bruta / entropia_maxima
    else:
        entropia_normalizada = 0.0

    return float(max(0.0, min(1.0, entropia_normalizada)))


def aplicar_dinamica_simbolica(
    valores: np.ndarray,
    m: int = SYMBOLIC_M_DEFAULT,
) -> dict:
    """
    Pipeline completo de dinâmica simbólica sobre array 1D de amostras.
    """
    sinal = np.asarray(valores, dtype=float).ravel()
    if sinal.size == 0:
        raise ValueError("Sinal vazio.")

    limiar = obter_limiar(sinal)
    sequencia_binaria = gerar_sequencia_binaria(sinal, limiar)
    grupos_binarios = gerar_grupos_deslizantes(sequencia_binaria, m)
    palavras_decimais = converter_para_decimal(grupos_binarios)
    frequencias = calcular_frequencia(palavras_decimais)
    entropia = calcular_entropia_shannon(frequencias)

    return {
        "sequencia_binaria": sequencia_binaria,
        "grupos_binarios": grupos_binarios,
        "palavras_decimais": palavras_decimais,
        "limiar": limiar,
        "entropia": entropia,
        "frequencias": frequencias,
    }
