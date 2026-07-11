import operator
from typing import Annotated, TypedDict

from kiwipiepy import Sentence

from app.graph.models import (
    Entity,
    SRLFrame,
    Triple,
)

class GraphState(TypedDict, total=False):
    article_id: str
    article: str
    # kiwi 기준으로 분리된 문장 목록 (형태소 토큰 포함). NER 청킹(문장 표면형)과
    # morphology(문장별 tokens)가 이 결과를 공유해 형태소 분석이 중복 수행되지 않는다.
    sentences: list[Sentence]
    chunk_text: str
    # 청크별 NER 노드가 병렬로 쓰는 채널이라 리듀서(operator.add)로 리스트를 이어붙인다
    ner_chunks: Annotated[list[list[Entity]], operator.add]
    verb_lemmas: list[str]
    ner: list[Entity]
    srl: list[SRLFrame]
    triples: list[Triple]
    fpdf_stats: dict