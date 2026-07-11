from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EntityLabel = Literal[
    "COMPANY", "GOVERNMENT", "COUNTRY",
    "COMMODITY", "PRODUCT",
]

# 확정된 과거/현재 사실만 최종 트리플로 남기기 위한 시제·양태 분류.
# past_or_present_fact 이외(future_or_planned, modal_possibility)는 FPDF에서 탈락시킨다.
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