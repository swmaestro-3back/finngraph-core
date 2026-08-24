"""
수집 → 저장 → 클러스터링 파이프라인.

    uv run python -m news.pipeline --query "특징주,공급" --display 100
    uv run python -m news.pipeline --from-file          # 저장된 news.json 재사용
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from news.client import NaverNewsClient, NewsItem
from news.cluster import DEFAULT_THRESHOLD, build_clusters
from news.preprocess import document_terms
from news.store import CLUSTERS_PATH, NEWS_PATH, load_news, save_clusters, save_news
from news.vectorize import build_tfidf

DEFAULT_QUERY = "특징주,공급"


def fetch(query: str = DEFAULT_QUERY, display: int = 100, sort: str = "sim") -> list[NewsItem]:
    return NaverNewsClient().search(query, total=display, sort=sort)


def cluster(
    items: list[NewsItem],
    threshold: float = DEFAULT_THRESHOLD,
    description_weight: float = 0.4,
) -> dict[str, Any]:
    """제목 중심 TF-IDF 로 군집을 만들고 직렬화 가능한 리포트를 돌려준다."""
    documents = [
        document_terms(item.title, item.description, description_weight) for item in items
    ]
    tfidf = build_tfidf(documents)
    similarity = tfidf.cosine_similarity()
    clusters = build_clusters(similarity, threshold)

    return {
        "threshold": threshold,
        "description_weight": description_weight,
        "document_count": len(items),
        "cluster_count": len(clusters),
        "singleton_count": sum(1 for c in clusters if len(c.members) == 1),
        "clusters": [
            {
                "id": cluster_id,
                "size": len(c.members),
                "cohesion": round(c.cohesion, 4),
                "keywords": tfidf.top_terms(c.members),
                "representative": items[c.representative].title,
                "items": [
                    {
                        "index": i,
                        "title": items[i].title,
                        "link": items[i].link,
                        "pub_date": items[i].pub_date,
                    }
                    for i in c.members
                ],
            }
            for cluster_id, c in enumerate(clusters)
        ],
    }


def run(
    query: str = DEFAULT_QUERY,
    display: int = 100,
    sort: str = "sim",
    threshold: float = DEFAULT_THRESHOLD,
    description_weight: float = 0.4,
    news_path: Path = NEWS_PATH,
    clusters_path: Path = CLUSTERS_PATH,
    from_file: bool = False,
) -> dict[str, Any]:
    if from_file:
        items = load_news(news_path)
    else:
        items = fetch(query, display, sort)
        save_news(items, news_path)

    report = cluster(items, threshold, description_weight)
    save_clusters(report, clusters_path)
    return report


def _print_summary(report: dict[str, Any]) -> None:
    print(
        f"문서 {report['document_count']}건 → 군집 {report['cluster_count']}개 "
        f"(단독 기사 {report['singleton_count']}개, threshold={report['threshold']})"
    )
    for entry in report["clusters"]:
        if entry["size"] < 2:
            continue
        keywords = ", ".join(entry["keywords"])
        print(f"\n[{entry['id']}] {entry['size']}건  응집도 {entry['cohesion']}  ({keywords})")
        for item in entry["items"]:
            print(f"    - {item['title']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="네이버 뉴스 수집 및 제목 기반 클러스터링")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--display", type=int, default=100, help="수집할 기사 수")
    parser.add_argument("--sort", default="sim", choices=["sim", "date"])
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--description-weight", type=float, default=0.4)
    parser.add_argument("--news-path", type=Path, default=NEWS_PATH)
    parser.add_argument("--clusters-path", type=Path, default=CLUSTERS_PATH)
    parser.add_argument(
        "--from-file",
        action="store_true",
        help="API를 호출하지 않고 저장된 news.json 으로 클러스터링만 다시 수행",
    )
    args = parser.parse_args()

    report = run(
        query=args.query,
        display=args.display,
        sort=args.sort,
        threshold=args.threshold,
        description_weight=args.description_weight,
        news_path=args.news_path,
        clusters_path=args.clusters_path,
        from_file=args.from_file,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()
