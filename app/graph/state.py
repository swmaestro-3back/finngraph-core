from typing import TypedDict

from app.graph.models import (
    Entity,
    SRLFrame,
    Triple,
)

class GraphState(TypedDict, total=False):
    article_id: str
    article: str
    sentences: list[str] # Kiwipiepy로 분리된 문장 텍스트 리스트
    ner: list[Entity]
    srl: list[SRLFrame]
    triples: list[Triple]
    fpdf_stats: dict