from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from app.core.config import settings

def get_llm() -> BaseChatModel:
    provider = settings.LLM_PROVIDER.lower()

    # 이거 다 settings 참조해서 불러오게끔 바꿔야함
    if provider == "openai":
        model = settings.OPENAI_MODEL
        return ChatOpenAI(model=model, temperature=0, api_key=settings.OPENAI_API_KEY)
    elif provider == "gemini":
        model = settings.GEMINI_MODEL
        return ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=settings.GOOGLE_API_KEY)
    else:
        raise ValueError(f"지원하지 않는 LLM 프로바이더: {provider!r}. 'openai' 또는 'gemini'를 사용하세요.")
