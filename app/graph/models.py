from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityLabel = Literal[
    "COMPANY", "GOVERNMENT", "COUNTRY",
    "COMMODITY", "PRODUCT",
]

# 확정된 사실만 최종 트리플로 남기기 위한 시제·양태 분류.
# modal_possibility(추측·가능성 표현)만 FPDF에서 탈락시키고, future_or_planned는
# "~할 계획/예정/방침/전망이다"처럼 확정된 사실이므로 past_or_present_fact와 함께 통과시킨다.
TenseLabel = Literal[
    "past_or_present_fact", "future_or_planned", "modal_possibility",
]

# NER
class Entity(BaseModel):
    text: str = Field(description="개체명 표면형")
    label: EntityLabel = Field(description="개체명 태그")

# SRL
class SRLFrame(BaseModel):
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
    is_negated: bool = Field(
        description="True if the relationship involves a negative expression, otherwise False."
    )
    tense: TenseLabel = Field(
        description="Temporal/modal status of the relationship: past_or_present_fact, future_or_planned, or modal_possibility."
    )

# FPDF
class Triple(BaseModel):
    subject: str = Field(description="주체")
    subject_type: str = Field(description="주체 개체명 타입")
    predicate: str = Field(description="술어 원형")
    object: str = Field(description="객체")
    object_type: str = Field(description="객체 개체명 타입")
    item: str | None = Field(default=None, description="품목 (3-argument 술어에서만 채워짐)")
    item_type: str | None = Field(default=None, description="품목 개체명 타입")