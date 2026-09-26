from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """

    APP_NAME: str = "my-fastAPI-boilerplate"
    APP_VERSION: str = "0.1.0"

    # MongoDB configuration
    MONGO_URI: str = Field(...)  # MongoDB URI
    DB_NAME: str = Field(...)  # Database name
    
    # JWT / Auth configuration
    JWT_SECRET: str = Field(...)
    JWT_ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)  # Token expiration time (minutes)

    # Email server
    MAIL_SERVER: Optional[str] = Field(default=None)
    MAIL_PORT: Optional[int] = Field(default=None)
    MAIL_USE_TLS: Optional[bool] = Field(default=None)
    MAIL_USERNAME: Optional[str] = Field(default=None)
    MAIL_PASSWORD: Optional[str] = Field(default=None)
    MAIL_DEFAULT_SENDER: Optional[str] = Field(default=None)


    model_config = SettingsConfigDict(
        env_file=".env",
        extra='ignore'
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings, loading from environment variables if not already cached
    """
    return Settings()

    