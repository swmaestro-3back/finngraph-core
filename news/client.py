"""
Naver 검색 API(뉴스) 클라이언트.

API가 돌려주는 항목은 title / originallink / link / description / pubDate 뿐이고
본문은 포함되지 않는다. description(200자 안팎 요약)이 본문 대용이다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

import httpx

from app.core.config import settings

API_URL = "https://openapi.naver.com/v1/search/news.json"

# 검색 API 한 번에 받을 수 있는 최대 건수, 그리고 start 파라미터의 상한.
MAX_DISPLAY = 100
MAX_START = 1000


@dataclass(slots=True)
class NewsItem:
    """검색 결과 한 건. HTML 태그가 제거된 상태로 보관한다."""

    title: str
    link: str
    original_link: str
    description: str
    pub_date: str
    # 본문 전문. 지금은 항상 비어 있고, 나중에 본문 크롤러가 채울 자리다.
    body: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "NewsItem":
        return cls(
            title=raw.get("title", ""),
            link=raw.get("link", ""),
            original_link=raw.get("original_link", ""),
            description=raw.get("description", ""),
            pub_date=raw.get("pub_date", ""),
            body=raw.get("body", ""),
        )


class NaverNewsClient:
    """검색 API를 감싼 얇은 래퍼. 페이징과 헤더 주입만 책임진다."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._client_id = client_id or settings.naver_client_id
        self._client_secret = client_secret or settings.naver_client_secret
        if not self._client_id or not self._client_secret:
            raise RuntimeError(
                "NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 가 .env 에 설정되어 있지 않습니다."
            )
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "X-Naver-Client-Id": self._client_id,
            "X-Naver-Client-Secret": self._client_secret,
        }

    def search(
        self,
        query: str,
        total: int = MAX_DISPLAY,
        sort: str = "sim",
    ) -> list[NewsItem]:
        """`total` 건이 모일 때까지 100건씩 나눠 호출한다."""
        from news.preprocess import strip_html

        collected: list[NewsItem] = []
        seen_links: set[str] = set()

        for start in _start_offsets(total):
            display = min(MAX_DISPLAY, total - len(collected))
            if display <= 0:
                break

            payload = self._request(query, display=display, start=start, sort=sort)
            items = payload.get("items", [])
            if not items:
                break

            for raw in items:
                # link 는 네이버 뉴스 주소, 없으면 언론사 원문 주소로 중복을 판단한다.
                key = raw.get("link") or raw.get("originallink", "")
                if key in seen_links:
                    continue
                seen_links.add(key)
                collected.append(
                    NewsItem(
                        title=strip_html(raw.get("title", "")),
                        link=raw.get("link", ""),
                        original_link=raw.get("originallink", ""),
                        description=strip_html(raw.get("description", "")),
                        pub_date=raw.get("pubDate", ""),
                    )
                )

            # 남은 결과가 요청한 것보다 적으면 더 부를 페이지가 없다.
            if len(items) < display:
                break
            time.sleep(0.1)  # 초당 호출 제한 회피

        return collected[:total]

    def _request(self, query: str, *, display: int, start: int, sort: str) -> dict[str, Any]:
        response = httpx.get(
            API_URL,
            params={"query": query, "display": display, "start": start, "sort": sort},
            headers=self._headers,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()


def _start_offsets(total: int) -> Iterator[int]:
    """1, 101, 201 ... 형태의 start 값을 상한까지 흘려보낸다."""
    start = 1
    while start <= MAX_START and start <= total:
        yield start
        start += MAX_DISPLAY
