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
    LLM_MODEL: str = "claude-3-5-sonnet-20240620"
    
    # SEC
    SEC_USER_AGENT: str = "SEC-Alpha-Sentinel jameshall@example.com"

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
