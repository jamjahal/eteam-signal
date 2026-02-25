from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    """
    Application configuration via Environment Variables.
    """
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # Project Info
    PROJECT_NAME: str = "SEC Alpha-Sentinel"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    
    # Paths
    DATA_DIR: str = "./data"
    
    # Vector DB (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "sec_filings"
    QDRANT_API_KEY: Optional[str] = None
    
    # LLM (Anthropic)
    ANTHROPIC_API_KEY: str = ""
    LLM_MODEL: str = "claude-sonnet-4-20250514"
    
    # SEC
    SEC_USER_AGENT: str = "SEC-Alpha-Sentinel jameshall@example.com"

    # PostgreSQL / TimescaleDB
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sec_alpha"
    POSTGRES_USER: str = "sec_alpha"
    POSTGRES_PASSWORD: str = ""
    POSTGRES_POOL_MIN: int = 2
    POSTGRES_POOL_MAX: int = 10

    # Insider Trading Detection
    INSIDER_ANOMALY_THRESHOLD: float = 0.6
    INSIDER_LOOKBACK_DAYS: int = 730
    INSIDER_CLUSTER_WINDOW_DAYS: int = 14
    INSIDER_INGEST_RATE_LIMIT: int = 8
    INSIDER_ATOM_POLL_INTERVAL_MARKET: int = 300
    INSIDER_ATOM_POLL_INTERVAL_OFF: int = 1800
    INSIDER_BATCH_INTERVAL_MINUTES: int = 60
    INSIDER_BATCH_OVERLAP_HOURS: int = 2

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
