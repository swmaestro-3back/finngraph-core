from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    python-settings를 사용하여 환경변수 관리
    main.py에서 import를 통해 앱 실행 시 읽고 시작하게 한다
    """

    # .env 파일을 읽어오기 위한 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # .env에 정의되지 않은 환경변수는 무시하고 진행
        extra="ignore"
    )

    LLM_PROVIDER: str
    GOOGLE_API_KEY: str
    OPENAI_API_KEY: str

    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str
    
    LANGSMITH_TRACING: bool
    LANGSMITH_ENDPOINT: str
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str

settings = Settings()