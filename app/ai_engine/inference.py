"""
Pipeline de inferencia CNN-LSTM hibrida.

Mantem compatibilidade com o modelo legado de 19 features e tambem suporta
modelos treinados com vetor expandido por canal.
"""

from __future__ import annotations

import asyncio
import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Input, LSTM, MaxPooling1D
from tensorflow.keras.models import Sequential

from app.ai_engine.feature_extractor import FEATURE_NAMES

N_FEATURES = len(FEATURE_NAMES)


def criar_modelo_cnn_lstm_hibrido(n_features: int = N_FEATURES) -> tf.keras.Model:
    """Arquitetura CNN-LSTM hibrida usada no projeto."""
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
    """Modelo leve apenas para testes."""
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
    """Recursos necessarios para uma inferencia."""

    __slots__ = ("model", "scaler", "metadata")

    def __init__(
        self,
        model: tf.keras.Model,
        scaler: StandardScaler | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.scaler = scaler
        self.metadata = metadata or {}


def _carregar_metadata_sidecar(path: Path) -> dict[str, Any]:
    metadata_path = path.with_name(f"{path.stem}_metadata.json")
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _carregar_modelo_cached(modelo_path: str) -> _ModeloCarregado:
    """
    Carrega o modelo Keras e scaler opcional uma unica vez por caminho.

    Formatos suportados:
    - .keras / .h5
    - .pkl legado com bundle {model, scaler, metadata}
    """
    path = Path(modelo_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {path}")

    suffix = path.suffix.lower()

    if suffix == ".pkl":
        with path.open("rb") as arquivo:
            bundle = pickle.load(arquivo)
        return _ModeloCarregado(
            model=bundle["model"],
            scaler=bundle.get("scaler"),
            metadata=bundle.get("metadata", {}),
        )

    if suffix in {".keras", ".h5"}:
        model = tf.keras.models.load_model(path)
        scaler_path = path.with_name(f"{path.stem}_scaler.pkl")
        scaler = None
        if scaler_path.exists():
            with scaler_path.open("rb") as arquivo:
                scaler = pickle.load(arquivo)
        return _ModeloCarregado(
            model=model,
            scaler=scaler,
            metadata=_carregar_metadata_sidecar(path),
        )

    raise ValueError(f"Formato de modelo nao suportado: {suffix}. Use .keras, .h5 ou .pkl")


def _n_features_esperadas(recursos: _ModeloCarregado) -> int:
    if recursos.scaler is not None and hasattr(recursos.scaler, "n_features_in_"):
        return int(recursos.scaler.n_features_in_)

    input_shape = recursos.model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if not isinstance(input_shape, tuple) or len(input_shape) < 2 or input_shape[1] is None:
        raise ValueError("Nao foi possivel determinar o numero de features esperado pelo modelo.")
    return int(input_shape[1])


def extrair_feature_vector(
    features: dict[str, Any],
    expected_n_features: int | None = None,
) -> np.ndarray:
    """Obtem vetor ordenado a partir do dicionario de features."""
    if "feature_vector" in features:
        vetor = features["feature_vector"]
    else:
        vetor = [features[nome] for nome in FEATURE_NAMES]

    array = np.asarray(vetor, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"feature_vector deve ser 1D, recebeu {array.shape}")
    if expected_n_features is not None and array.shape != (expected_n_features,):
        raise ValueError(
            f"feature_vector deve ter {expected_n_features} elementos, recebeu {array.shape}"
        )
    return array


def preparar_tensor_entrada(
    features: dict[str, Any],
    scaler: StandardScaler | None = None,
    expected_n_features: int | None = None,
) -> np.ndarray:
    """
    Converte features em tensor 3D `(1, n_features, 1)` para Conv1D/LSTM.

    Aplica StandardScaler quando disponivel.
    """
    vetor = extrair_feature_vector(features, expected_n_features=expected_n_features).reshape(1, -1)
    if scaler is not None:
        vetor = scaler.transform(vetor)
    return vetor.reshape(1, vetor.shape[1], 1).astype(np.float32)


def _executar_predicao(features: dict[str, Any], modelo_path: str) -> float:
    """Predicao sincrona executada em thread pelo wrapper assincrono."""
    recursos = _carregar_modelo_cached(modelo_path)
    tensor = preparar_tensor_entrada(
        features,
        recursos.scaler,
        expected_n_features=_n_features_esperadas(recursos),
    )
    saida = recursos.model.predict(tensor, verbose=0)
    probabilidade = float(np.asarray(saida).reshape(-1)[0])
    return float(max(0.0, min(1.0, probabilidade)))


def obter_vetor_escalonado(features: dict[str, Any], modelo_path: str) -> np.ndarray:
    """Retorna feature_vector 1D no mesmo espaco usado pelo modelo."""
    recursos = _carregar_modelo_cached(modelo_path)
    vetor = extrair_feature_vector(
        features,
        expected_n_features=_n_features_esperadas(recursos),
    ).reshape(1, -1)
    if recursos.scaler is not None:
        vetor = recursos.scaler.transform(vetor)
    return vetor.flatten().astype(np.float32)


def obter_modelo_keras(modelo_path: str) -> tf.keras.Model:
    """Retorna o modelo Keras em cache para inferencia ou SHAP."""
    return _carregar_modelo_cached(modelo_path).model


def obter_metadados_modelo(modelo_path: str) -> dict[str, Any]:
    """Retorna metadados opcionais do modelo carregado."""
    return dict(_carregar_modelo_cached(modelo_path).metadata)


def obter_scaler_modelo(modelo_path: str) -> StandardScaler | None:
    """Retorna o scaler associado ao modelo, quando existir."""
    return _carregar_modelo_cached(modelo_path).scaler


async def realizar_inferencia(features: dict[str, Any], modelo_path: str) -> float:
    """
    Executa inferencia CNN-LSTM de forma assincrona.

    Args:
        features: Dicionario retornado por `extrair_features_edf`.
        modelo_path: Caminho absoluto do modelo.
    """
    return await asyncio.to_thread(_executar_predicao, features, modelo_path)


def limpar_cache_modelos() -> None:
    """Libera modelos em cache."""
    _carregar_modelo_cached.cache_clear()


class CNNLSTMInferencePipeline:
    """Facade OO para o pipeline de inferencia."""

    def __init__(self, model_path: Path | str) -> None:
        self.model_path = str(Path(model_path).resolve())

    async def predict(self, features: dict[str, Any]) -> float:
        return await realizar_inferencia(features, self.model_path)

    def predict_sync(self, features: dict[str, Any]) -> float:
        return _executar_predicao(features, self.model_path)
