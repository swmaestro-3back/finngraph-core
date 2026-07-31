from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings

llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_MODEL,
    temperature=0,
    google_api_key=settings.GOOGLE_API_KEY
)