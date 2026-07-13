from typing import TypedDict

from kiwipiepy import Sentence

from app.graph.models import (
    Entity,
    SRLFrame,
    Triple,
)

class GraphState(TypedDict, total=False):
    article_id: str
    article: str
    sentences: list[Sentence] # Kiwipipey를 통해 분리된 Sentence(문장 + 형태소 분석 정보)
    ner_chunks: list[list[Entity]]  # NER 노드의 배치추론 결과로 한번에 채워지는 필드
    verb_lemmas: list[str]
    ner: list[Entity]
    srl: list[SRLFrame]
    triples: list[Triple]
    fpdf_stats: dict