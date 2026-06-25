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
