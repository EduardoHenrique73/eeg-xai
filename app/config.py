"""Configurações centralizadas da aplicação."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "EEG-XAI"
    app_env: str = "development"
    debug: bool = True

    database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/eeg_xai"
    )

    storage_root: Path = Path(__file__).resolve().parent.parent / "storage"
    edf_storage_path: Path = storage_root / "edf"
    shap_storage_path: Path = storage_root / "shap"
    modelos_path: Path = Path(__file__).resolve().parent.parent / "modelos"
    keras_model_path: Path = modelos_path / "cnn_lstm_hybrid.keras"
    max_edf_duration_seconds: float | None = None

    # Seleciona o fluxo de IA: "legacy" (CNN-LSTM por janela) ou
    # "sequence_cnn_lstm" (modelo global/inter-paciente por sequencias).
    ai_model_type: str = "legacy"
    ai_sequence_model_path: Path = modelos_path / "sequence_cnn_lstm_features.keras"
    # Caminhos opcionais; quando vazios sao derivados de ai_sequence_model_path.
    ai_sequence_scaler_path: Path | None = None
    ai_sequence_metadata_path: Path | None = None
    ai_sequence_calibration_path: Path | None = None
    ai_sequence_default_threshold: float = 0.5
    ai_sequence_default_min_duration_seconds: float = 30.0
    ai_sequence_top_segments_limit: int = 5
    ai_sequence_max_suspicious_coverage: float = 0.7

    ai_sequence_feature_mode: str = "mean"
    ai_sequence_channel_reference_edf: str | None = None
    ai_sequence_window_seconds: float = 4.0
    ai_sequence_step_seconds: float = 2.0
    ai_sequence_length: int = 8
    ai_sequence_stride: int = 2
    ai_sequence_max_normal_windows_per_file: int = 180
    ai_sequence_max_seizure_windows_per_file: int = 40
    ai_sequence_epochs: int = 4
    ai_sequence_batch_size: int = 16
    ai_sequence_calibration_patients_per_fold: int = 4
    ai_sequence_thresholds: str = "0.5,0.6,0.7,0.8,0.85,0.9,0.95"
    ai_sequence_durations: str = "10,20,30,45,60,90,120,180"
    ai_sequence_disk_cache_enabled: bool = True
    ai_sequence_cache_path: Path = storage_root / "feature_cache" / "sequence"

    jwt_secret_key: str = "troque-esta-chave-em-producao-eeg-xai"
    jwt_expire_minutes: int = 60 * 8

    @field_validator("max_edf_duration_seconds", mode="before")
    @classmethod
    def parse_max_edf_duration(cls, value: object) -> object:
        """Vazio ou 'none' = analisar o arquivo .edf inteiro."""
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "none", "null", "full", "all"}:
                return None
        return value

    @field_validator("ai_sequence_channel_reference_edf", mode="before")
    @classmethod
    def parse_optional_string(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @field_validator(
        "ai_sequence_scaler_path",
        "ai_sequence_metadata_path",
        "ai_sequence_calibration_path",
        mode="before",
    )
    @classmethod
    def parse_optional_path(cls, value: object) -> object:
        """Vazio/none = derivar do caminho do modelo em tempo de execucao."""
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
            return None
        return value

    @field_validator("ai_model_type", mode="before")
    @classmethod
    def parse_model_type(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "legacy", "cnn_lstm", "hybrid"}:
                return "legacy"
            if normalized in {"sequence", "sequence_cnn_lstm", "sequencial"}:
                return "sequence_cnn_lstm"
            raise ValueError(
                "AI_MODEL_TYPE deve ser 'legacy' ou 'sequence_cnn_lstm'."
            )
        return value

    @property
    def usar_modelo_sequencial(self) -> bool:
        return self.ai_model_type == "sequence_cnn_lstm"

    @field_validator("ai_sequence_feature_mode", mode="before")
    @classmethod
    def parse_feature_mode(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized not in {"mean", "per_channel", "time_frequency", "time_frequency_per_channel", "raw_signal"}:
                raise ValueError(
                    "AI_SEQUENCE_FEATURE_MODE deve ser 'mean', 'per_channel', "
                    "'time_frequency', 'time_frequency_per_channel' ou 'raw_signal'."
                )
            return normalized
        return value

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_flag(cls, value: object) -> object:
        """
        Aceita valores comuns de ambiente para evitar conflito com variaveis
        globais genericas como DEBUG=release.
        """
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"debug", "development", "dev"}:
                return True
        return value

    @property
    def is_development(self) -> bool:
        return self.app_env.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
