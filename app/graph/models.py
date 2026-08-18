from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field

EntityLabel = Literal[
    "COMPANY", "GOVERNMENT", "COUNTRY",
    "COMMODITY", "PRODUCT",
]

# 관계의 시제·양태 분류. UI 표시명은 각각 확정 / 예정 / 관측.
TenseLabel = Literal[
    "past_or_present_fact",
    "future_or_planned",
    "modal_possibility",
]

# 관계의 극성 분류. UI 표시명은 각각 성립 / 부인 / 종료.
# 기존 is_negated(bool)를 대체한다. bool과 병존시키면 두 값이 어긋난 상태가 생길 수 있다.
Polarity = Literal[
    # 관계가 성립한다고 서술
    "affirmed",
    # 관계 성립 자체를 부정·부인
    "denied",
    # 성립했던 관계가 해지·취소·중단
    "terminated",
]


# NER
class Entity(BaseModel):
    text: str = Field(description="개체명 표면형")
    label: EntityLabel = Field(description="개체명 태그")


# LLM이 NER 결과와 무관하게 subject/object/item에 라벨을 지어낼 수 있으므로
# 할루시네이션 방지용 모델. 이후 Label을 매핑해서 반환한다.
class RawRelation(BaseModel):
    # 필드 선언 순서가 곧 추론 순서다(structured output은 선언 순서대로 값을 채운다).
    # "근거 문장을 먼저 찾고 → 절로 재구성하고 → 그 다음에 술어를 고른다"를 프레임마다 강제한다.
    source_sentence: str = Field(
        description=(
            "The sentence from the original text that expresses this relationship, "
            "copied VERBATIM — character-for-character, no paraphrasing, no omission. "
            "This is the evidence for the frame."
        )
    )
    clause: str = Field(
        description=(
            "The source_sentence rewritten as a short standalone clause that directly matches "
            "the chosen predicate's direction: resolve pronouns to their referent entity, "
            "restate counterparty phrasing in the predicate's direction "
            "(e.g. 'A buys X from B' becomes 'B supplies X to A'), "
            "and keep only the single relationship this frame expresses."
        )
    )
    predicate: str = Field(description="Must be strictly selected from the registered_predicates list.")
    subject: str = Field(description="Must exactly match an entity from the provided NER results.")
    object: str = Field(description="Must exactly match an entity from the provided NER results.")
    item: Optional[str] = Field(
        default=None,
        description=(
            "Must exactly match an entity from the provided NER results."
            "Leave null if the predicate has no item argument, or if the text does not name a specific item and the item argument is marked [optional]."
        ),
    )


class RawRelationList(BaseModel):
    frames: List[RawRelation] = Field(
        description=(
            "List of extracted frames, each containing source_sentence, clause, subject, predicate, object, and (if applicable) item. "
            "One frame per relationship; a single sentence may yield multiple frames, each repeating the same source_sentence."
        )
    )


# FrameAnnotator가 프레임 하나에 붙이는 주석. frame_index와 subject/predicate/object는
# 정합 검증용 에코이고, evidence를 먼저 쓴 뒤 그 근거를 보고 polarity/tense를 판정하도록
# 선언 순서를 고정한다.
class RawAnnotation(BaseModel):
    frame_index: int = Field(description="The 0-based index of the frame being annotated, copied from the input list.")
    subject: str = Field(description="Copy the subject of the input frame back EXACTLY as given.")
    predicate: str = Field(description="Copy the predicate of the input frame back EXACTLY as given.")
    object: str = Field(description="Copy the object of the input frame back EXACTLY as given.")
    evidence: str = Field(
        description=(
            "A single self-contained Korean sentence that a reader can understand on its own: "
            "resolve pronouns to entity names, restore the omitted subject, and pull in the time, "
            "condition, and attribution stated elsewhere in the article. Never add facts not in the text."
        )
    )
    polarity: Polarity = Field(
        description="Polarity of the relationship: affirmed, denied, or terminated."
    )
    tense: TenseLabel = Field(
        description="Temporal/modal status of the relationship: past_or_present_fact, future_or_planned, or modal_possibility."
    )


class RawAnnotationList(BaseModel):
    annotations: List[RawAnnotation] = Field(
        description="One annotation per input frame, in the same order as the input list."
    )


# RelationExtractor의 출력. 아직 의미 라벨이 붙기 전의 관계 후보다.
class CandidateFrame(BaseModel):
    subject: Entity = Field(description="Must exactly match an entity from the provided NER results.")
    object: Entity = Field(description="Must exactly match an entity from the provided NER results.")
    # PREDICATE_DICT_NARY에서 3번째 argument(item)를 등록한 술어(공급하다 등)에만 채워지는
    # 선택적 필드. 해당 술어라도 본문에 품목이 명시되지 않으면 None으로 남는다.
    item: Entity | None = Field(
        default=None,
        description="Only set for predicates with a third 'item' argument in PREDICATE_DICT_NARY.",
    )
    predicate: str = Field(
        description="Must be strictly selected from the registered_predicates list."
    )
    # 원문 대조 검증(공백 정규화 후 substring 매칭)을 통과한 프레임만 생성되므로 항상 채워진다.
    source_sentence: str = Field(
        description="The verbatim sentence from the source text that expresses this relationship.",
    )
    # 술어 방향에 정합한 기계용 최소 절. 사람이 읽는 근거는 evidence이며 역할이 다르다.
    # clause를 제거하면 "A가 B로부터 구매 → B가 A에게 공급" 방향 정합 추론이 사라져
    # 술어 오선택이 늘어나므로 유지한다.
    clause: str = Field(
        description="The relationship restated as a single clause matching the predicate's direction.",
    )


# FrameAnnotator까지 통과한 최종 관계 프레임.
class RelationFrame(CandidateFrame):
    # 사람이 한 줄만 읽고 이해할 수 있도록 맥락을 복원한 근거. 생성문이므로 원문 대조가
    # 불가능하고, 대신 subject/object/item 표면형 포함 여부와 길이로 검증한다.
    evidence: str = Field(
        description="A self-contained Korean sentence describing the relationship with restored context.",
    )
    polarity: Polarity = Field(description="affirmed / denied / terminated")
    tense: TenseLabel = Field(
        description="past_or_present_fact / future_or_planned / modal_possibility"
    )


# Triplet (확정 삼중항)
class Triplet(BaseModel):
    subject: Entity = Field(description="주체")
    predicate: str = Field(description="술어 원형")
    object: Entity = Field(description="객체")
    item: Entity | None = Field(default=None, description="품목 (3rd Argument가 필수적인 술어에서만 채워짐)")
    source_sentence: str = Field(description="관계의 근거가 된 원문 문장 (provenance)")
    evidence: str = Field(description="맥락이 복원된 자립적 근거 문장 (UI 노출용)")
    polarity: Polarity = Field(description="affirmed / denied / terminated")
    tense: TenseLabel = Field(description="past_or_present_fact / future_or_planned / modal_possibility")
