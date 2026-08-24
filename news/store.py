"""news.json / clusters.json 읽기·쓰기."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from news.client import NewsItem

NEWS_PATH = Path("news/news.json")
CLUSTERS_PATH = Path("news/clusters.json")


def save_news(items: list[NewsItem], path: Path = NEWS_PATH) -> Path:
    payload = {"count": len(items), "items": [item.to_dict() for item in items]}
    _write_json(path, payload)
    return path


def load_news(path: Path = NEWS_PATH) -> list[NewsItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [NewsItem.from_dict(raw) for raw in payload.get("items", [])]


def save_clusters(payload: dict[str, Any], path: Path = CLUSTERS_PATH) -> Path:
    _write_json(path, payload)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
