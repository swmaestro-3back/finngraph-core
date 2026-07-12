"""
환경변수 접근 규칙

- 이 프로젝트의 모든 환경변수는 반드시 settings.X 형태로만 읽는다. os.getenv나
  os.environ을 코드 어디에서도 직접 사용하지 않는다.
- .env에 값을 추가해도 아래 Settings 클래스에 필드로 선언하지 않으면 아무도 그 값을
  읽을 수 없다. pydantic-settings의 env_file 설정은 .env를 읽어 settings 객체의
  필드를 채울 뿐, os.environ 자체를 변경하지는 않기 때문이다. 즉 settings.X는 항상
  동작하지만 os.getenv("X")는 이 값을 절대 보지 못한다.
- 유일한 예외는 LangSmith SDK다. 이 라이브러리는 설정값을 생성자 인자로 받지 않고
  자기 내부에서 os.environ만 직접 읽도록 만들어져 있어서, settings 값을 넘겨줄 방법이
  없다. 그래서 이 파일 하단에서 settings 값을 os.environ으로 복사해주는 브릿징 코드를
  둔다. os.environ 사용은 이런 서드파티 제약이 있을 때만 예외적으로 허용하고, 그 외
  자체 코드에서는 절대 사용하지 않는다.
"""

import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

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

    LLM_PROVIDER: str

    OPENAI_MODEL: str
    GEMINI_MODEL: str

    OPENAI_API_KEY: SecretStr
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