"""유닛 테스트는 실제 LLM·DB를 호출하지 않지만, app.core.config가 import 시점에
Settings()를 만들기 때문에 환경변수가 없으면 모듈 import 자체가 실패한다.
테스트 모듈이 import되기 전에 더미 값을 채워둔다. 이미 설정된 값은 덮어쓰지 않는다.
"""

import os

_DUMMY_ENV = {
    "GEMINI_MODEL": "gemini-test",
    "GOOGLE_API_KEY": "test-key",
    "NEO4J_URI": "bolt://localhost:7687",
    "NEO4J_USERNAME": "neo4j",
    "NEO4J_PASSWORD": "password",
    "NEO4J_DATABASE": "neo4j",
    "LANGSMITH_TRACING": "false",
    "LANGSMITH_ENDPOINT": "https://example.invalid",
    "LANGSMITH_API_KEY": "test-key",
    "LANGSMITH_PROJECT": "test",
}

for _key, _value in _DUMMY_ENV.items():
    os.environ.setdefault(_key, _value)
