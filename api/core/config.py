import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

    # Database
    DATABASE_URL: str = "sqlite:///./inference_platform.db"

    # Redis Cache & Queue
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_FALLBACK_MEMORY: bool = True

    # Security & Auth
    AUTH_ENABLED: bool = True
    DEFAULT_MASTER_KEY: str = "ai_live_master_key_999999"
    FREE_TIER_RPM: int = 100
    PREMIUM_TIER_RPM: int = 10000

    # Dynamic Batching
    MAX_BATCH_SIZE: int = 16
    MAX_WAIT_MS: int = 20

    # Worker & Storage
    WORKER_DEVICE: str = "cpu"
    WORKER_ID: str = "worker-node-1"
    MODEL_STORAGE_DIR: str = "./model_artifacts"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
