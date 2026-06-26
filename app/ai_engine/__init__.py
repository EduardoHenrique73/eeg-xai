"""
Motor de IA — módulo isolado (implementação futura).

Pipeline planejado:
1. Leitura de arquivos .edf via mne-python
2. Extração de features (dinâmica simbólica + entropia de Shannon)
3. Inferência CNN-LSTM híbrida
4. Geração de mapa SHAP (XAI) salvo em disco
"""

from app.ai_engine.feature_extractor import (
    FEATURE_NAMES,
    FeatureExtractor,
    extrair_features_edf,
    extrair_features_de_valores,
)
from app.ai_engine.inference import (
    CNNLSTMInferencePipeline,
    criar_modelo_cnn_lstm_dummy,
    limpar_cache_modelos,
    obter_modelo_keras,
    realizar_inferencia,
)
from app.ai_engine.shap_explainer import SHAPExplainer, gerar_mapa_shap
from app.ai_engine.training import (
    SeizureInterval,
    avaliar_kfold_cnn_lstm,
    avaliar_kfold_features,
    carregar_resumos_chbmit,
    extrair_dataset_janelado_de_sinal,
    gerar_janelas_temporais,
    parse_chbmit_summary,
    treinar_cnn_lstm_final,
)

__all__ = [
    "FeatureExtractor",
    "CNNLSTMInferencePipeline",
    "SHAPExplainer",
    "FEATURE_NAMES",
    "extrair_features_edf",
    "extrair_features_de_valores",
    "realizar_inferencia",
    "criar_modelo_cnn_lstm_dummy",
    "limpar_cache_modelos",
    "gerar_mapa_shap",
    "obter_modelo_keras",
    "SeizureInterval",
    "carregar_resumos_chbmit",
    "parse_chbmit_summary",
    "gerar_janelas_temporais",
    "extrair_dataset_janelado_de_sinal",
    "avaliar_kfold_features",
    "avaliar_kfold_cnn_lstm",
    "treinar_cnn_lstm_final",
]
