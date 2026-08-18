from typing import TypedDict

from app.graph.models import (
    CandidateFrame,
    Entity,
    RelationFrame,
    Triplet,
)


class GraphState(TypedDict, total=False):
    news_id: str
    article: str
    entities: list[Entity]
    # RelationExtractor의 출력. 아직 evidence/polarity/tense가 붙기 전이다.
    candidates: list[CandidateFrame]
    # FrameAnnotator까지 통과한 최종 프레임.
    relations: list[RelationFrame]
    # FrameAnnotator에서 드롭된 프레임 수. 드롭 정책을 택한 이상 손실률이 관측 가능해야 한다.
    annotation_stats: dict
    triplets: list[Triplet]
    triplet_stats: dict
