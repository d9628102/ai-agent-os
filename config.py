"""pydantic-settings 設定管理。"""

from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class HermesSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    anthropic_api_key: str = Field(..., validation_alias="ANTHROPIC_API_KEY")
    hermes_postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/hermes_db"
    hermes_api_key: Optional[str] = None
    hermes_max_steps: int = 8
    hermes_env: str = "development"

    @field_validator("hermes_max_steps")
    @classmethod
    def max_steps_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("HERMES_MAX_STEPS 必須為正整數")
        return v

    @property
    def require_api_key(self) -> bool:
        if self.hermes_env == "development":
            return False
        return bool(self.hermes_api_key)


_settings: Optional[HermesSettings] = None


def get_settings() -> HermesSettings:
    global _settings
    if _settings is None:
        _settings = HermesSettings()
    return _settings
