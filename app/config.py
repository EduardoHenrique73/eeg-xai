"""Configurações centralizadas da aplicação."""

from functools import lru_cache
from pathlib import Path

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
    max_edf_duration_seconds: float | None = 120.0

    jwt_secret_key: str = "troque-esta-chave-em-producao-eeg-xai"
    jwt_expire_minutes: int = 60 * 8

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    return Settings()
