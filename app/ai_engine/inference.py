"""
Pipeline de inferência CNN-LSTM híbrida — predição de probabilidade de crise epiléptica.

Porta a lógica de reshape e predict do ml_classifier legado (hybrid CNN-LSTM).
"""

from __future__ import annotations

import asyncio
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Input,
    LSTM,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential

from app.ai_engine.feature_extractor import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)


def criar_modelo_cnn_lstm_hibrido(n_features: int = N_FEATURES) -> tf.keras.Model:
    """
    Arquitetura CNN-LSTM híbrida compatível com o legado PIBITI.

    Entrada: (batch, n_features, 1) — 19 features como sequência 1D para Conv1D.
    Saída: probabilidade sigmoid [0, 1] (classe positiva = crise).
    """
    return Sequential(
        [
            Input(shape=(n_features, 1)),
            Conv1D(32, 3, activation="relu", padding="same"),
            MaxPooling1D(2),
            Conv1D(64, 3, activation="relu", padding="same"),
            MaxPooling1D(2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ],
        name="cnn_lstm_hybrid_eeg",
    )


def criar_modelo_cnn_lstm_dummy(n_features: int = N_FEATURES) -> tf.keras.Model:
    """Modelo leve com pesos aleatórios — apenas para testes e CI."""
    model = Sequential(
        [
            Input(shape=(n_features, 1)),
            Conv1D(8, 3, activation="relu", padding="same"),
            MaxPooling1D(2),
            LSTM(16, return_sequences=False),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),
        ],
        name="cnn_lstm_dummy",
    )
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


class _ModeloCarregado:
    """Recursos necessários para uma inferência."""

    __slots__ = ("model", "scaler")

    def __init__(
        self,
        model: tf.keras.Model,
        scaler: StandardScaler | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler


@lru_cache(maxsize=8)
def _carregar_modelo_cached(modelo_path: str) -> _ModeloCarregado:
    """
    Carrega o modelo Keras (e scaler opcional) uma única vez por caminho.

    Formatos suportados:
        - `.keras` / `.h5` — somente rede Keras
        - `.pkl` — bundle legado {model, scaler, feature_names, ...}
    """
    path = Path(modelo_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with path.open("rb") as arquivo:
            bundle = pickle.load(arquivo)
        model = bundle["model"]
        scaler = bundle.get("scaler")
        return _ModeloCarregado(model=model, scaler=scaler)

    if suffix in {".keras", ".h5"}:
        model = tf.keras.models.load_model(path)
        return _ModeloCarregado(model=model, scaler=None)

    raise ValueError(
        f"Formato de modelo não suportado: {suffix}. Use .keras, .h5 ou .pkl"
    )


def extrair_feature_vector(features: dict[str, Any]) -> np.ndarray:
    """Obtém vetor ordenado (19,) a partir do dicionário de features."""
    if "feature_vector" in features:
        vetor = features["feature_vector"]
    else:
        vetor = [features[nome] for nome in FEATURE_NAMES]

    array = np.asarray(vetor, dtype=np.float32)
    if array.shape != (N_FEATURES,):
        raise ValueError(
            f"feature_vector deve ter {N_FEATURES} elementos, recebeu {array.shape}"
        )
    return array


def preparar_tensor_entrada(
    features: dict[str, Any],
    scaler: StandardScaler | None = None,
) -> np.ndarray:
    """
    Converte features em tensor 3D `(1, 19, 1)` para Conv1D/LSTM.

    Aplica StandardScaler quando disponível (bundle .pkl legado).
    """
    vetor = extrair_feature_vector(features).reshape(1, -1)

    if scaler is not None:
        vetor = scaler.transform(vetor)

    return vetor.reshape(1, vetor.shape[1], 1).astype(np.float32)


def _executar_predicao(features: dict[str, Any], modelo_path: str) -> float:
    """Predição síncrona — executada em thread pelo wrapper assíncrono."""
    recursos = _carregar_modelo_cached(modelo_path)
    tensor = preparar_tensor_entrada(features, recursos.scaler)

    saida = recursos.model.predict(tensor, verbose=0)
    probabilidade = float(np.asarray(saida).reshape(-1)[0])

    return float(max(0.0, min(1.0, probabilidade)))


def obter_vetor_escalonado(features: dict[str, Any], modelo_path: str) -> np.ndarray:
    """Retorna feature_vector 1D no mesmo espaço usado pelo modelo (com scaler se houver)."""
    recursos = _carregar_modelo_cached(modelo_path)
    vetor = extrair_feature_vector(features).reshape(1, -1)
    if recursos.scaler is not None:
        vetor = recursos.scaler.transform(vetor)
    return vetor.flatten().astype(np.float32)


def obter_modelo_keras(modelo_path: str) -> tf.keras.Model:
    """Retorna o modelo Keras em cache para inferência ou SHAP."""
    return _carregar_modelo_cached(modelo_path).model


async def realizar_inferencia(features: dict[str, Any], modelo_path: str) -> float:
    """
    Executa inferência CNN-LSTM de forma assíncrona (não bloqueia o event loop).

    Args:
        features: Dicionário retornado por `extrair_features_edf` (com `feature_vector`).
        modelo_path: Caminho absoluto do modelo (.keras, .h5 ou .pkl).

    Returns:
        Probabilidade da classe positiva (crise epiléptica), entre 0.0 e 1.0.
    """
    return await asyncio.to_thread(_executar_predicao, features, modelo_path)


def limpar_cache_modelos() -> None:
    """Libera modelos em cache (útil em testes e hot-reload)."""
    _carregar_modelo_cached.cache_clear()


class CNNLSTMInferencePipeline:
    """Facade orientada a objetos para o pipeline de inferência."""

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = str(Path(model_path).resolve())

    async def predict(self, features: dict[str, Any]) -> float:
        return await realizar_inferencia(features, self.model_path)

    def predict_sync(self, features: dict[str, Any]) -> float:
        return _executar_predicao(features, self.model_path)
