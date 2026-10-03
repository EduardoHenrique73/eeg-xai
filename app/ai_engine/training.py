"""Pipeline simples de janelamento, data augmentation e k-fold.

Este modulo e propositalmente leve: ele cria janelas temporais sobre sinais EEG,
rotula cada janela por sobreposicao com intervalos de crise e avalia as 19
features ja usadas pelo sistema com validacao cruzada estratificada.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any

import mne
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.ai_engine.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_MODE_RAW_SIGNAL,
    FEATURE_MODE_TIME_FREQUENCY,
    FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
    FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
    FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
    nomes_features_por_modo,
    extrair_features_por_modo_de_valores,
    extrair_features_de_valores,
    selecionar_canais_eeg_validos,
)
from app.ai_engine.inference import criar_modelo_cnn_lstm_hibrido


@dataclass(frozen=True)
class SeizureInterval:
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class WindowSpec:
    start_seconds: float
    end_seconds: float
    label: int
    context: str = "unknown"
    ictal_overlap_ratio: float = 0.0


FEATURE_MODE_MEAN = "mean"
FEATURE_MODE_PER_CHANNEL = "per_channel"


def parse_chbmit_summary(summary_path: str | Path) -> dict[str, list[SeizureInterval]]:
    """Extrai intervalos de crise do arquivo `chbXX-summary.txt` do CHB-MIT."""
    texto = Path(summary_path).read_text(encoding="utf-8", errors="ignore")
    resultado: dict[str, list[SeizureInterval]] = {}
    arquivo_atual: str | None = None
    inicio_atual: float | None = None

    for linha in texto.splitlines():
        nome = re.search(r"File Name:\s*(\S+)", linha)
        if nome:
            arquivo_atual = nome.group(1)
            resultado.setdefault(arquivo_atual, [])
            inicio_atual = None
            continue

        inicio = re.search(
            r"Seizure(?:\s+\d+)? Start Time:\s*(\d+(?:\.\d+)?)\s*seconds",
            linha,
        )
        if inicio and arquivo_atual:
            inicio_atual = float(inicio.group(1))
            continue

        fim = re.search(
            r"Seizure(?:\s+\d+)? End Time:\s*(\d+(?:\.\d+)?)\s*seconds",
            linha,
        )
        if fim and arquivo_atual and inicio_atual is not None:
            resultado[arquivo_atual].append(
                SeizureInterval(
                    start_seconds=inicio_atual,
                    end_seconds=float(fim.group(1)),
                )
            )
            inicio_atual = None

    return resultado


def carregar_resumos_chbmit(
    dataset_dir: str | Path,
) -> dict[str, list[SeizureInterval]]:
    """
    Carrega e combina todos os arquivos `chbXX-summary.txt` de um diretorio.

    Quando ha mais de um paciente no mesmo dataset local, cada EDF precisa ser
    rotulado pelo summary correspondente ao seu proprio prefixo `chbXX`.
    """
    base = Path(dataset_dir)
    summaries = sorted(base.glob("*-summary.txt"))
    if not summaries:
        raise FileNotFoundError(f"Nenhum arquivo *-summary.txt encontrado em {base}")

    resultado: dict[str, list[SeizureInterval]] = {}
    for summary_path in summaries:
        resultado.update(parse_chbmit_summary(summary_path))
    return resultado


def window_overlaps_seizure(
    start_seconds: float,
    end_seconds: float,
    intervals: list[SeizureInterval],
) -> bool:
    """Retorna True quando a janela cruza qualquer intervalo de crise."""
    return any(
        start_seconds < interval.end_seconds and end_seconds > interval.start_seconds
        for interval in intervals
    )


def seizure_overlap_seconds(
    start_seconds: float,
    end_seconds: float,
    intervals: list[SeizureInterval],
) -> float:
    """Soma a sobreposicao da janela com intervalos de crise."""
    return float(
        sum(
            max(
                0.0,
                min(end_seconds, interval.end_seconds)
                - max(start_seconds, interval.start_seconds),
            )
            for interval in intervals
        )
    )


def seizure_overlap_ratio(
    start_seconds: float,
    end_seconds: float,
    intervals: list[SeizureInterval],
) -> float:
    duration = max(0.0, end_seconds - start_seconds)
    if duration <= 0:
        return 0.0
    return float(min(1.0, seizure_overlap_seconds(start_seconds, end_seconds, intervals) / duration))


def classificar_contexto_janela(
    start_seconds: float,
    end_seconds: float,
    intervals: list[SeizureInterval],
    *,
    context_margin_seconds: float = 60.0,
) -> str:
    if not intervals:
        return "normal_file"
    if window_overlaps_seizure(start_seconds, end_seconds, intervals):
        return "ictal"

    for interval in intervals:
        if interval.start_seconds - context_margin_seconds <= end_seconds <= interval.start_seconds:
            return "pre_ictal"
        if interval.end_seconds <= start_seconds <= interval.end_seconds + context_margin_seconds:
            return "post_ictal"
    return "interictal"


def gerar_janelas_temporais(
    duration_seconds: float,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    min_ictal_overlap_ratio: float = 0.0,
) -> list[WindowSpec]:
    """
    Gera janelas sobrepostas. Esse janelamento e a forma inicial de data augmentation.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds deve ser maior que zero.")
    if step_seconds <= 0:
        raise ValueError("step_seconds deve ser maior que zero.")
    if not 0.0 <= min_ictal_overlap_ratio <= 1.0:
        raise ValueError("min_ictal_overlap_ratio deve estar entre 0 e 1.")
    if duration_seconds < window_seconds:
        return []

    specs: list[WindowSpec] = []
    start = 0.0
    while start + window_seconds <= duration_seconds:
        end = start + window_seconds
        overlap_ratio = seizure_overlap_ratio(start, end, intervals)
        label = int(overlap_ratio > 0.0 and overlap_ratio >= min_ictal_overlap_ratio)
        context = classificar_contexto_janela(start, end, intervals)
        if label == 0 and overlap_ratio > 0.0:
            context = "boundary"
        specs.append(
            WindowSpec(
                start_seconds=start,
                end_seconds=end,
                label=label,
                context=context,
                ictal_overlap_ratio=overlap_ratio,
            )
        )
        start += step_seconds
    return specs


def _limitar_janelas_por_classe(
    specs: list[WindowSpec],
    max_windows_per_class: int | None,
    *,
    max_normal_windows: int | None = None,
    max_seizure_windows: int | None = None,
    min_contiguous_windows: int = 1,
    balance_positive_events: bool = False,
) -> list[WindowSpec]:
    if max_windows_per_class is None and max_normal_windows is None and max_seizure_windows is None:
        return specs

    def selecionar_blocos_uniformes(classe: list[WindowSpec], limite: int) -> list[WindowSpec]:
        if len(classe) <= limite:
            return classe
        if limite <= 0:
            return []
        if limite == 1:
            return [classe[len(classe) // 2]]

        n_blocos = min(
            4,
            max(1, limite // max(1, min_contiguous_windows)),
            limite,
            len(classe),
        )
        tamanho_bloco = max(1, limite // n_blocos)
        sobras = limite - (tamanho_bloco * n_blocos)
        inicios = np.linspace(0, len(classe) - 1, n_blocos, dtype=int)
        selecionadas: list[WindowSpec] = []
        usados: set[tuple[float, float]] = set()
        for bloco_idx, centro in enumerate(inicios):
            atual_tamanho = tamanho_bloco + (1 if bloco_idx < sobras else 0)
            inicio = int(max(0, min(len(classe) - atual_tamanho, centro - atual_tamanho // 2)))
            for spec in classe[inicio : inicio + atual_tamanho]:
                chave = (spec.start_seconds, spec.end_seconds)
                if chave not in usados:
                    selecionadas.append(spec)
                    usados.add(chave)

        if len(selecionadas) < limite:
            for spec in classe:
                chave = (spec.start_seconds, spec.end_seconds)
                if chave in usados:
                    continue
                selecionadas.append(spec)
                usados.add(chave)
                if len(selecionadas) >= limite:
                    break
        return selecionadas[:limite]

    def selecionar_normais_contextuais(classe: list[WindowSpec], positivos: list[WindowSpec], limite: int) -> list[WindowSpec]:
        if len(classe) <= limite:
            return classe
        if not positivos:
            return selecionar_blocos_uniformes(classe, limite)

        grupos = {
            "boundary": sorted(
                [spec for spec in classe if spec.context == "boundary"],
                key=lambda spec: spec.ictal_overlap_ratio,
                reverse=True,
            ),
            "pre_ictal": sorted(
                [spec for spec in classe if spec.context == "pre_ictal"],
                key=lambda spec: spec.start_seconds,
                reverse=True,
            ),
            "post_ictal": sorted(
                [spec for spec in classe if spec.context == "post_ictal"],
                key=lambda spec: spec.start_seconds,
            ),
            "interictal": [spec for spec in classe if spec.context == "interictal"],
        }
        pesos = {"boundary": 0.25, "pre_ictal": 0.25, "post_ictal": 0.25, "interictal": 0.25}
        selecionadas: list[WindowSpec] = []
        usados: set[tuple[float, float]] = set()

        for contexto, peso in pesos.items():
            alvo = max(1, int(round(limite * peso))) if grupos[contexto] else 0
            if contexto == "interictal":
                candidatos = selecionar_blocos_uniformes(grupos[contexto], alvo)
            else:
                candidatos = grupos[contexto][:alvo]
            for spec in candidatos:
                chave = (spec.start_seconds, spec.end_seconds)
                if chave not in usados:
                    selecionadas.append(spec)
                    usados.add(chave)

        if len(selecionadas) < limite:
            for contexto in ("boundary", "pre_ictal", "post_ictal", "interictal"):
                for spec in grupos[contexto]:
                    chave = (spec.start_seconds, spec.end_seconds)
                    if chave in usados:
                        continue
                    selecionadas.append(spec)
                    usados.add(chave)
                    if len(selecionadas) >= limite:
                        return selecionadas
        return selecionadas[:limite]

    def selecionar_positivos_por_evento(
        classe: list[WindowSpec], limite: int,
    ) -> list[WindowSpec]:
        """Distribute the positive budget across contiguous seizure events."""
        ordenados = sorted(classe, key=lambda spec: spec.start_seconds)
        grupos: list[list[WindowSpec]] = []
        for spec in ordenados:
            if not grupos or spec.start_seconds > grupos[-1][-1].end_seconds + 1e-6:
                grupos.append([spec])
            else:
                grupos[-1].append(spec)
        if len(ordenados) <= limite or len(grupos) <= 1:
            return selecionar_blocos_uniformes(ordenados, limite)

        base, extra = divmod(limite, len(grupos))
        quotas = [min(len(grupo), base + (indice < extra)) for indice, grupo in enumerate(grupos)]
        while sum(quotas) < limite:
            alterou = False
            for indice, grupo in enumerate(grupos):
                if quotas[indice] < len(grupo):
                    quotas[indice] += 1
                    alterou = True
                    if sum(quotas) >= limite:
                        break
            if not alterou:
                break

        selecionadas_eventos: list[WindowSpec] = []
        for grupo, quota in zip(grupos, quotas):
            if quota <= 0:
                continue
            indices = np.linspace(0, len(grupo) - 1, quota, dtype=int)
            selecionadas_eventos.extend(grupo[int(indice)] for indice in indices)
        return selecionadas_eventos

    positivos = [spec for spec in specs if spec.label == 1]
    selecionadas: list[WindowSpec] = []
    for label in (0, 1):
        limite = max_normal_windows if label == 0 else max_seizure_windows
        if limite is None:
            limite = max_windows_per_class
        if limite is None:
            selecionadas.extend(spec for spec in specs if spec.label == label)
            continue

        classe = [spec for spec in specs if spec.label == label]
        if len(classe) <= limite:
            selecionadas.extend(classe)
            continue
        if label == 0:
            selecionadas.extend(selecionar_normais_contextuais(classe, positivos, limite))
        elif balance_positive_events:
            selecionadas.extend(selecionar_positivos_por_evento(classe, limite))
        else:
            selecionadas.extend(selecionar_blocos_uniformes(classe, limite))

    return sorted(selecionadas, key=lambda spec: spec.start_seconds)


def limitar_dataset_janelado(
    x: np.ndarray,
    y: np.ndarray,
    metadados: list[dict[str, Any]],
    *,
    max_normal_windows: int | None = None,
    max_seizure_windows: int | None = None,
    min_contiguous_windows: int = 1,
    balance_positive_events: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Aplica a amostragem depois da extracao ou normalizacao das janelas."""
    matriz = np.asarray(x)
    rotulos = np.asarray(y)
    if len(matriz) != len(rotulos) or len(rotulos) != len(metadados):
        raise ValueError("x, y e metadados devem ter o mesmo tamanho.")
    if max_normal_windows is None and max_seizure_windows is None:
        return matriz, rotulos, metadados

    specs = [
        WindowSpec(
            start_seconds=float(meta["start_seconds"]),
            end_seconds=float(meta["end_seconds"]),
            label=int(label),
            context=str(meta.get("context") or "interictal"),
            ictal_overlap_ratio=float(meta.get("ictal_overlap_ratio") or 0.0),
        )
        for label, meta in zip(rotulos, metadados)
    ]
    selecionadas = _limitar_janelas_por_classe(
        specs,
        None,
        max_normal_windows=max_normal_windows,
        max_seizure_windows=max_seizure_windows,
        min_contiguous_windows=min_contiguous_windows,
        balance_positive_events=balance_positive_events,
    )
    chaves = {
        (spec.start_seconds, spec.end_seconds, spec.label)
        for spec in selecionadas
    }
    indices = [
        indice
        for indice, (label, meta) in enumerate(zip(rotulos, metadados))
        if (float(meta["start_seconds"]), float(meta["end_seconds"]), int(label)) in chaves
    ]
    return matriz[indices], rotulos[indices], [metadados[indice] for indice in indices]


def extrair_dataset_janelado_de_sinal(
    signal: np.ndarray,
    sfreq: float,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    max_windows_per_class: int | None = None,
    max_normal_windows: int | None = None,
    max_seizure_windows: int | None = None,
    feature_mode: str = FEATURE_MODE_MEAN,
    min_ictal_overlap_ratio: float = 0.0,
    min_contiguous_windows: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extrai matriz X/y de features a partir de um sinal 1D ja carregado."""
    sinal = np.asarray(signal, dtype=float).ravel()
    duration_seconds = sinal.size / sfreq
    specs = gerar_janelas_temporais(
        duration_seconds,
        intervals,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_ictal_overlap_ratio=min_ictal_overlap_ratio,
    )
    specs = _limitar_janelas_por_classe(
        specs,
        max_windows_per_class,
        max_normal_windows=max_normal_windows,
        max_seizure_windows=max_seizure_windows,
        min_contiguous_windows=min_contiguous_windows,
    )

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    metadados: list[dict[str, Any]] = []

    for spec in specs:
        inicio = int(round(spec.start_seconds * sfreq))
        fim = int(round(spec.end_seconds * sfreq))
        janela = sinal[inicio:fim]
        if janela.size < 4:
            continue

        features = extrair_features_por_modo_de_valores(
            janela,
            feature_mode=feature_mode,
            sfreq=sfreq,
        )
        x_rows.append([float(v) for v in features["feature_vector"]])
        y_rows.append(spec.label)
        metadados.append(
            {
                "start_seconds": spec.start_seconds,
                "end_seconds": spec.end_seconds,
                "label": spec.label,
                "context": spec.context,
                "ictal_overlap_ratio": spec.ictal_overlap_ratio,
            }
        )

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), metadados


def _extrair_vetor_por_canal(janela_canais: np.ndarray) -> list[float]:
    vetor: list[float] = []
    for sinal_canal in np.asarray(janela_canais, dtype=float):
        features = extrair_features_de_valores(sinal_canal)
        vetor.extend(float(features[nome]) for nome in FEATURE_NAMES)
    return vetor


def _extrair_vetor_por_canal_modo(
    janela_canais: np.ndarray,
    *,
    sfreq: float,
    feature_mode: str,
) -> list[float]:
    vetor: list[float] = []
    for sinal_canal in np.asarray(janela_canais, dtype=float):
        features = extrair_features_por_modo_de_valores(
            sinal_canal,
            feature_mode=feature_mode,
            sfreq=sfreq,
        )
        vetor.extend(float(v) for v in features["feature_vector"])
    return vetor


def extrair_dataset_janelado_multicanal(
    sinais: np.ndarray,
    sfreq: float,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    max_windows_per_class: int | None = None,
    max_normal_windows: int | None = None,
    max_seizure_windows: int | None = None,
    feature_mode: str = FEATURE_MODE_PER_CHANNEL,
    min_ictal_overlap_ratio: float = 0.0,
    min_contiguous_windows: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """
    Extrai features por canal para cada janela sem colapsar o EEG em media global.

    Shape de entrada esperado: (n_canais, n_amostras).
    Shape de saida: (n_janelas, n_canais * n_features_por_canal).
    """
    matriz = np.asarray(sinais, dtype=float)
    if matriz.ndim != 2:
        raise ValueError("sinais deve ter shape (n_canais, n_amostras).")

    n_canais, n_amostras = matriz.shape
    if n_canais < 1 or n_amostras < 4:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), []

    duration_seconds = n_amostras / sfreq
    specs = gerar_janelas_temporais(
        duration_seconds,
        intervals,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_ictal_overlap_ratio=min_ictal_overlap_ratio,
    )
    specs = _limitar_janelas_por_classe(
        specs,
        max_windows_per_class,
        max_normal_windows=max_normal_windows,
        max_seizure_windows=max_seizure_windows,
        min_contiguous_windows=min_contiguous_windows,
    )

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    metadados: list[dict[str, Any]] = []

    for spec in specs:
        inicio = int(round(spec.start_seconds * sfreq))
        fim = int(round(spec.end_seconds * sfreq))
        janela = matriz[:, inicio:fim]
        if janela.shape[1] < 4:
            continue

        if feature_mode == FEATURE_MODE_PER_CHANNEL:
            x_rows.append(_extrair_vetor_por_canal(janela))
        else:
            x_rows.append(
                _extrair_vetor_por_canal_modo(
                    janela,
                    sfreq=sfreq,
                    feature_mode=feature_mode,
                )
            )
        y_rows.append(spec.label)
        metadados.append(
            {
                "start_seconds": spec.start_seconds,
                "end_seconds": spec.end_seconds,
                "label": spec.label,
                "context": spec.context,
                "ictal_overlap_ratio": spec.ictal_overlap_ratio,
                "n_canais": int(n_canais),
            }
        )

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), metadados


def extrair_dataset_janelado_multicanal_referencia(
    raw: mne.io.BaseRaw,
    intervals: list[SeizureInterval],
    canais_referencia: list[str],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    max_windows_per_class: int | None = None,
    max_normal_windows: int | None = None,
    max_seizure_windows: int | None = None,
    feature_mode: str = FEATURE_MODE_PER_CHANNEL,
    min_ictal_overlap_ratio: float = 0.0,
    min_contiguous_windows: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Extrai features por canal em uma montagem fixa, zerando canais ausentes."""
    if not canais_referencia:
        raise ValueError("canais_referencia nao pode ser vazio em modo por canal.")

    canais_validos_lista = selecionar_canais_eeg_validos(raw)
    canais_disponiveis = list(raw.ch_names)
    indices_validos = [canais_disponiveis.index(canal) for canal in canais_validos_lista]
    dados_validos = raw.get_data(picks=indices_validos)
    sinais_disponiveis = {
        canal: np.asarray(sinal, dtype=float)
        for canal, sinal in zip(canais_validos_lista, dados_validos)
    }
    sinais_referencia: dict[str, np.ndarray] = {}
    canais_derivados: set[str] = set()
    for canal in canais_referencia:
        if canal in sinais_disponiveis:
            sinais_referencia[canal] = sinais_disponiveis[canal]
            continue

        # MNE adds running-number suffixes to duplicate channel names.
        canal_base = re.sub(r"-\d+$", "", canal)
        partes = canal_base.split("-")
        if len(partes) != 2:
            continue
        eletrodo_a, eletrodo_b = partes
        inverso = f"{eletrodo_b}-{eletrodo_a}"
        if inverso in sinais_disponiveis:
            sinais_referencia[canal] = -sinais_disponiveis[inverso]
            canais_derivados.add(canal)
            continue

        prefixo_a = f"{eletrodo_a}-"
        referencias_a = {
            nome[len(prefixo_a):]: sinal
            for nome, sinal in sinais_disponiveis.items()
            if nome.startswith(prefixo_a)
        }
        prefixo_b = f"{eletrodo_b}-"
        referencias_b = {
            nome[len(prefixo_b):]: sinal
            for nome, sinal in sinais_disponiveis.items()
            if nome.startswith(prefixo_b)
        }
        referencias_comuns = sorted(set(referencias_a) & set(referencias_b))
        if referencias_comuns:
            referencia = referencias_comuns[0]
            sinais_referencia[canal] = (
                referencias_a[referencia] - referencias_b[referencia]
            )
            canais_derivados.add(canal)
    sfreq = float(raw.info["sfreq"])
    duration_seconds = raw.n_times / sfreq
    specs = gerar_janelas_temporais(
        duration_seconds,
        intervals,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        min_ictal_overlap_ratio=min_ictal_overlap_ratio,
    )
    specs = _limitar_janelas_por_classe(
        specs,
        max_windows_per_class,
        max_normal_windows=max_normal_windows,
        max_seizure_windows=max_seizure_windows,
        min_contiguous_windows=min_contiguous_windows,
    )

    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    metadados: list[dict[str, Any]] = []
    nomes_features = nomes_features_por_modo(feature_mode)
    for spec in specs:
        inicio = int(round(spec.start_seconds * sfreq))
        fim = int(round(spec.end_seconds * sfreq))
        if fim - inicio < 4:
            continue

        vetor: list[float] = []
        canais_processados: list[str] = []
        canais_omitidos: list[str] = []
        for canal in canais_referencia:
            if canal not in sinais_referencia:
                vetor.extend([0.0] * len(nomes_features))
                canais_omitidos.append(canal)
                continue

            sinal_canal = np.asarray(sinais_referencia[canal][inicio:fim], dtype=float)
            feat = extrair_features_por_modo_de_valores(
                sinal_canal,
                feature_mode=feature_mode,
                sfreq=sfreq,
            )
            vetor.extend(float(v) for v in feat["feature_vector"])
            canais_processados.append(canal)

        x_rows.append(vetor)
        y_rows.append(spec.label)
        metadados.append(
            {
                "start_seconds": spec.start_seconds,
                "end_seconds": spec.end_seconds,
                "label": spec.label,
                "context": spec.context,
                "ictal_overlap_ratio": spec.ictal_overlap_ratio,
                "n_canais": int(len(canais_processados)),
                "canais_omitidos": canais_omitidos,
                "canais_derivados": sorted(canais_derivados),
            }
        )

    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int64), metadados


def extrair_dataset_janelado_edf(
    edf_path: str | Path,
    intervals: list[SeizureInterval],
    *,
    window_seconds: float = 10.0,
    step_seconds: float = 5.0,
    max_windows_per_class: int | None = 40,
    max_normal_windows: int | None = None,
    max_seizure_windows: int | None = None,
    canais_selecionados: list[str] | None = None,
    feature_mode: str = FEATURE_MODE_MEAN,
    min_ictal_overlap_ratio: float = 0.0,
    min_contiguous_windows: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Carrega um EDF, calcula sinal medio dos canais e extrai features por janela."""
    path = Path(edf_path)
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    sfreq = float(raw.info["sfreq"])
    if feature_mode in {
        FEATURE_MODE_MEAN,
        FEATURE_MODE_TIME_FREQUENCY,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE,
        FEATURE_MODE_RAW_SIGNAL,
    }:
        raw.pick(selecionar_canais_eeg_validos(raw, canais_selecionados))
        sinais = raw.get_data()
        signal = np.mean(sinais, axis=0)
        x, y, metadados = extrair_dataset_janelado_de_sinal(
            signal,
            sfreq,
            intervals,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            max_windows_per_class=max_windows_per_class,
            max_normal_windows=max_normal_windows,
            max_seizure_windows=max_seizure_windows,
            feature_mode=feature_mode,
            min_ictal_overlap_ratio=min_ictal_overlap_ratio,
            min_contiguous_windows=min_contiguous_windows,
        )
    elif feature_mode in {
        FEATURE_MODE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_RELATIVE_PER_CHANNEL,
        FEATURE_MODE_TIME_FREQUENCY_MORPHOLOGY_PER_CHANNEL,
    }:
        if canais_selecionados:
            x, y, metadados = extrair_dataset_janelado_multicanal_referencia(
                raw,
                intervals,
                list(canais_selecionados),
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                max_windows_per_class=max_windows_per_class,
                max_normal_windows=max_normal_windows,
                max_seizure_windows=max_seizure_windows,
                feature_mode=feature_mode,
                min_ictal_overlap_ratio=min_ictal_overlap_ratio,
                min_contiguous_windows=min_contiguous_windows,
            )
        else:
            raw.pick(selecionar_canais_eeg_validos(raw))
            sinais = raw.get_data()
            x, y, metadados = extrair_dataset_janelado_multicanal(
                sinais,
                sfreq,
                intervals,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
                max_windows_per_class=max_windows_per_class,
                max_normal_windows=max_normal_windows,
                max_seizure_windows=max_seizure_windows,
                feature_mode=feature_mode,
                min_ictal_overlap_ratio=min_ictal_overlap_ratio,
                min_contiguous_windows=min_contiguous_windows,
            )
    else:
        raise ValueError(f"feature_mode invalido: {feature_mode}")

    for item in metadados:
        item["arquivo"] = path.name
        item["feature_mode"] = feature_mode
        item["canais_processados"] = list(raw.ch_names)
    return x, y, metadados


def avaliar_kfold_features(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Avalia as features com Logistic Regression e k-fold estratificado."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("K-fold requer ao menos duas classes: normal e crise.")

    splits = min(n_splits, int(np.min(counts)))
    if splits < 2:
        raise ValueError("Cada classe precisa ter ao menos duas janelas para k-fold.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for train_idx, test_idx in cv.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(x[train_idx], y[train_idx])
        pred = model.predict(x[test_idx])

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "precision": float(precision_score(y[test_idx], pred, zero_division=0)),
                "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
                "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
            }
        )

    media = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in fold_metrics[0]
    }

    return {
        "n_samples": int(len(y)),
        "n_features": int(x.shape[1]) if x.ndim == 2 else 0,
        "n_splits": int(splits),
        "class_counts": {str(int(cls)): int(count) for cls, count in zip(classes, counts)},
        "folds": fold_metrics,
        "mean": media,
    }


def _class_weights(y: np.ndarray) -> dict[int, float]:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {
        int(cls): float(total / (len(classes) * count))
        for cls, count in zip(classes, counts)
    }


def _tensorizar_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(x.shape[0], x.shape[1], 1)


def avaliar_kfold_cnn_lstm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 3,
    epochs: int = 8,
    batch_size: int = 16,
    random_state: int = 42,
) -> dict[str, Any]:
    """Avalia a arquitetura CNN-LSTM em k-fold estratificado."""
    import tensorflow as tf

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("K-fold CNN-LSTM requer ao menos duas classes.")

    splits = min(n_splits, int(np.min(counts)))
    if splits < 2:
        raise ValueError("Cada classe precisa ter ao menos duas janelas para k-fold.")

    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
    fold_metrics: list[dict[str, float]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_state + fold_idx)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])

        model = criar_modelo_cnn_lstm_hibrido(n_features=x.shape[1])
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            _tensorizar_features(x_train),
            y[train_idx],
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            class_weight=_class_weights(y[train_idx]),
        )

        proba = model.predict(_tensorizar_features(x_test), verbose=0).reshape(-1)
        pred = (proba >= 0.5).astype(np.int64)

        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y[test_idx], pred)),
                "precision": float(precision_score(y[test_idx], pred, zero_division=0)),
                "recall": float(recall_score(y[test_idx], pred, zero_division=0)),
                "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
            }
        )

    media = {
        metric: float(np.mean([fold[metric] for fold in fold_metrics]))
        for metric in fold_metrics[0]
    }

    return {
        "n_samples": int(len(y)),
        "n_features": int(x.shape[1]) if x.ndim == 2 else 0,
        "n_splits": int(splits),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "class_counts": {str(int(cls)): int(count) for cls, count in zip(classes, counts)},
        "folds": fold_metrics,
        "mean": media,
    }


def treinar_cnn_lstm_final(
    x: np.ndarray,
    y: np.ndarray,
    *,
    output_path: str | Path,
    epochs: int = 12,
    batch_size: int = 16,
    random_state: int = 42,
) -> dict[str, str]:
    """Treina CNN-LSTM em todas as janelas e salva modelo Keras + scaler lateral."""
    import tensorflow as tf

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Treino final requer ao menos duas classes.")

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(random_state)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = criar_modelo_cnn_lstm_hibrido(n_features=x.shape[1])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        _tensorizar_features(x_scaled),
        y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        class_weight=_class_weights(y),
    )

    destino = Path(output_path)
    destino.parent.mkdir(parents=True, exist_ok=True)
    model.save(destino)

    scaler_path = destino.with_name(f"{destino.stem}_scaler.pkl")
    with scaler_path.open("wb") as arquivo:
        pickle.dump(scaler, arquivo)

    return {
        "model_path": str(destino.resolve()),
        "scaler_path": str(scaler_path.resolve()),
    }
