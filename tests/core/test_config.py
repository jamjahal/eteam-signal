import os
import pytest
from src.core.config import Settings, get_settings

def test_settings_load_defaults():
    settings = Settings()
    assert settings.PROJECT_NAME == "SEC Alpha-Sentinel"
    assert settings.QDRANT_HOST == "localhost"

def test_settings_load_env(monkeypatch):
    monkeypatch.setenv("QDRANT_HOST", "test-host")
    monkeypatch.setenv("QDRANT_PORT", "9999")
    
    # We need to bypass lru_cache for this test or manually instantiate Settings
    settings = Settings() 
    assert settings.QDRANT_HOST == "test-host"
    assert settings.QDRANT_PORT == 9999

def test_get_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
