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

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str

    langsmith_tracing: bool
    langsmith_endpoint: str
    langsmith_api_key: str
    langsmith_project: str

    naver_client_id: str = ""
    naver_client_secret: str = ""

    bedrock_region: str = ""
    bedrock_chat_model: str = ""
    bedrock_request_timeout: int = 300
    aws_bearer_token_bedrock: str = ""

settings = Settings()

# boto3 reads the Bedrock API key from os.environ only, never constructor arguments,
# so copy it across once at import time. Leaving it unset falls back to the standard
# AWS credential chain (profile, SSO, IAM role).
if settings.aws_bearer_token_bedrock:
    os.environ.setdefault("AWS_BEARER_TOKEN_BEDROCK", settings.aws_bearer_token_bedrock)

# The LangSmith SDK reads os.environ only, never constructor arguments, so copy the values
# across once at import time. Module caching keeps this to a single run per process.
if settings.langsmith_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project