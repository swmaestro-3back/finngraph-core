from typing import TypedDict

from app.graph.models import (
    Entity,
    RelationFrame,
    Triplet,
)

class GraphState(TypedDict, total=False):
    news_id: str
    article: str
    entities: list[Entity]
    relations: list[RelationFrame]
    triplets: list[Triplet]
    triplet_stats: dict