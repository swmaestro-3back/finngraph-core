import os

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    pydantic-settings로 .env를 읽어 타입 검증된 설정 객체를 만든다.
    환경변수가 필요한 코드는 os.getenv가 아니라 이 settings 인스턴스를 import해서 쓴다.
    """

    # .env 파일을 읽어오기 위한 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # .env에 정의되지 않은 환경변수는 무시하고 진행
        extra="ignore"
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

# LangSmith SDK는 생성자 인자가 아니라 os.environ만 읽으므로, settings를 import하는
# 시점에 한 번만 값을 os.environ으로 옮겨준다. (파이썬 모듈 캐싱 덕분에 프로세스당 1회만 실행됨)
if settings.LANGSMITH_TRACING:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT