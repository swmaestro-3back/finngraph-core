"""
Type-checked settings loaded from .env by pydantic-settings
Import this settings instance instead of directly reaching for os.getenv
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    # Load values from .env
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore any variable not declared below
    )

    GEMINI_MODEL: str
    GOOGLE_API_KEY: str

    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str
    
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

settings = Settings()

# The LangSmith SDK reads os.environ only, never constructor arguments, so copy the values
# across once at import time. Module caching keeps this to a single run per process.
if settings.LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT