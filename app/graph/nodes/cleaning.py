import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.llm import get_llm

_SYSTEM = """\
You are an expert in Korean news text preprocessing and data cleaning.
Your task is to remove all background noise, metadata, and non-essential text from the provided Korean news article, extracting ONLY the core body text for text analysis.

Strictly remove the following elements from the text:
1. Reporter Information: Names, email addresses, and contact info (e.g., "홍길동 기자 hong@news.com", "기자명").
2. Technical Noise: HTML tags, broken characters, and unformatted special characters.
3. Promotional Content: Advertisements, links to related articles, SNS share prompts, and newsletter subscription requests.
5. Copyright Notices: Media copyright claims and redistribution bans (e.g., "Copyrights ⓒ...", "무단전재 및 재배포 금지").
6. Image Captions: Descriptions or credits for photos and illustrations (e.g., "[사진=OOO]").

Output Guidelines:
- Return ONLY the cleaned core body text of the article.
- Do not include any introductory phrases, explanations, or conversational responses.
"""

_HUMAN = "{text}"

_prompt : ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", _HUMAN),
])

class Cleaner:
    def __init__(self):
        llm = get_llm()
        self._chain = _prompt | llm | StrOutputParser()

    def clean(self, text: str) -> str:
        cleaned = self._chain.invoke({"text": text})
        # LLM이 프롬프트만으로 개행/탭을 항상 지우진 않으므로, 남은 공백류(\n, \t, \r 등
        # 연속 공백 포함)를 스페이스 하나로 정규화해 단어가 붙지 않게 한다.
        return re.sub(r"\s+", " ", cleaned).strip()
