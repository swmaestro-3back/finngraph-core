import os

from langchain_core.language_models import BaseChatModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from core.config import settings

def get_llm() -> BaseChatModel:
    provider = settings.LLM_PROVIDER.lower()

    # 이거 다 settings 참조해서 불러오게끔 바꿔야함
    if provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=0)
    else:
        raise ValueError(f"지원하지 않는 LLM 프로바이더: {provider!r}. 'openai' 또는 'gemini'를 사용하세요.")
