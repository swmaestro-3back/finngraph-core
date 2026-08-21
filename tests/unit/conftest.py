"""
Fill in dummy environment variables before any test module is imported

Unit tests never touch a real LLM or database, but app.core.config builds Settings() at
import time, so a missing variable breaks the import itself. Existing values are left alone.
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
