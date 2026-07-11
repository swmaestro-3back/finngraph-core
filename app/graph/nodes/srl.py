import json
from pathlib import Path
from typing import List

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.graph.models import (
    Entity,
    SRLFrame,
    TenseLabel,
)

# LLM이 NER 결과와 무관하게 subject/object에 라벨을 지어낼 수 있으므로
# 할루시네이션을 방지용 모델 생성
# 추후 Label 매핑해서 반환
class _RawSRLFrame(BaseModel):
    predicate: str = Field(description="Must be strictly selected from the registered_predicates list.")
    is_negated: bool = Field(description="True if the relationship involves a negative expression, otherwise False.")
    tense: TenseLabel = Field(
        description="Temporal/modal status of the relationship: past_or_present_fact, future_or_planned, or modal_possibility."
    )
    subject: str = Field(description="Must exactly match an entity from the provided NER results.")
    object: str = Field(description="Must exactly match an entity from the provided NER results.")

class _RawSRLOutput(BaseModel):
    frames: List[_RawSRLFrame] = Field(description="List of extracted triplets, each containing subject, predicate, and object.")

_PREDICATE_DICT_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "dictionaries" / "predicate_dict.json"
)
with open(_PREDICATE_DICT_PATH, encoding="utf-8") as _f:
    _REGISTERED_PREDICATES: set[str] = set(json.load(_f).keys())

_SYSTEM = """\
[Role]
You are an expert Information Extraction system specialized in Korean economic and financial news.
Your sole task is to extract multilateral business relationships among entities such as 'COMPANY, GOVERNMENT, COUNTRY, and MATERIAL/PRODUCT' in the form of pure, structured triplets: (subject, predicate, object).

[Core Principles]
1. Strict Triplet Structure: Every extracted frame must contain exactly three elements: "subject", "predicate", and "object". Never extract any additional arguments, modifiers, temporal information (time/date), monetary amounts, or percentage shares.
2. Arguments Must Match NER Entities: The values of "subject" and "object" MUST exactly match the surface forms provided in the "NER Results (entities)" list below. Never use terms (e.g., general nouns, abstract concepts, or unrecognized words) that are not present in the given "entities" list. If an entity is not available in the list to fill either the subject or the object, do not extract that frame. Do not invent any values.
3. Predicate Constraint: The "predicate" MUST be chosen strictly from the "registered_predicates" list. Do not create new verbs or relationship names. Prioritize matching verbs from the "verb_lemmas" (verbs actually appearing in the text).
4. Multiple Triplet Separation (Mandatory): If a single sentence contains multiple facts or relationships, never combine them into a single frame. Extract them as independent frames and include all of them in the "frames" list.
5. Negation Handling (is_negated): Set "is_negated" to true if the relationship involves a negative expression (e.g., "does not", "fails to", "rejects", "denied"). Otherwise, set it to false.
6. Tense/Modality Classification (tense): Classify the temporal/modal status of the clause containing the predicate into exactly one of:
   - "past_or_present_fact": an already-completed past event, or an ongoing/habitual present-tense fact (including implied relationships presented as presently existing, e.g. "코델코는 칠레에 위치한 광산기업이다").
   - "future_or_planned": a future, planned, or scheduled action (e.g., "~할 계획이다", "~할 예정이다", "~할 방침이다", "~할 전망이다").
   - "modal_possibility": a speculative, hypothetical, or ability/possibility expression (e.g., "~할 수도 있다", "~할 수 있다", "~것으로 보인다", "~가능성이 있다").
   Always extract the frame regardless of its tense/modality classification — do not omit future/modal relationships. Downstream filtering relies on this field to keep only confirmed facts.

[Inferring Grounding and Contextual Mapping]
If the contextual meaning clearly maps to a registered predicate (e.g., "establishing a joint venture" maps to "cooperate_with"), you may use that registered predicate even if the exact verb lemma is missing.

[Inferring Implied Relationships]
A relationship between two COMPANY/GOVERNMENT/COUNTRY entities does not need to be stated by an explicit verb connecting them directly.
If the surrounding context clearly implies that a business relationship already exists or is presupposed between two such entities infer that implied relationship and map it to the closest matching registered predicate, even though no verb directly links the two entities in the sentence.

[Argument Type]
- subject: The active agent (COMPANY, GOVERNMENT, or COUNTRY).
- object: The target entity. (COMPANY, GOVERNMENT, COUNTRY, COMMODITY, PRODUCT)
"""

# Few-shot 예시: 아래 human/ai 메시지 쌍으로 대화 turn을 재현해 실제 입출력 형식을 시연한다.
# output은 값으로 주입되므로(템플릿 리터럴이 아님) JSON의 중괄호를 이스케이프할 필요가 없다.
_EXAMPLES = [
    {
        "text": "삼성전자는 테슬라, 엔비디아에 이어 새 대형 고객사를 확보하기 위해 칠레의 코델코로부터 리튬을 대량 구매했다.",
        "entities": "삼성전자(COMPANY), 테슬라(COMPANY), 엔비디아(COMPANY), 칠레(COUNTRY), 코델코(COMPANY), 리튬(COMMODITY)",
        "output": json.dumps(
            {
                "frames": [
                    {"predicate": "공급하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "삼성전자", "object": "테슬라"},
                    {"predicate": "공급하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "삼성전자", "object": "엔비디아"},
                    {"predicate": "구매하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "삼성전자", "object": "리튬"},
                    {"predicate": "생산하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "코델코", "object": "리튬"},
                    {"predicate": "위치하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "코델코", "object": "칠레"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        "text": (
            "구글은 페이스북을 인수할 계획이라고 밝혔다. 앞서 구글은 유튜브를 인수했다. "
            "한편 테슬라는 리비안을 인수할 수도 있다는 관측이 나온다."
        ),
        "entities": "구글(COMPANY), 페이스북(COMPANY), 유튜브(COMPANY), 테슬라(COMPANY), 리비안(COMPANY)",
        "output": json.dumps(
            {
                "frames": [
                    {"predicate": "인수하다", "is_negated": False, "tense": "future_or_planned", "subject": "구글", "object": "페이스북"},
                    {"predicate": "인수하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "구글", "object": "유튜브"},
                    {"predicate": "인수하다", "is_negated": False, "tense": "modal_possibility", "subject": "테슬라", "object": "리비안"},
                ]
            },
            ensure_ascii=False,
        ),
    },
]

_example_prompt = ChatPromptTemplate.from_messages([
    ("human", "**Text**:\n{text}\n\n**NER results (entities)**:\n{entities}"),
    ("ai", "{output}"),
])

_few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=_example_prompt,
    examples=_EXAMPLES,
)

_HUMAN = """\
**Text**:
{text}

**Registered predicate list**:
{registered_predicates}

**Verb lemma list**:
{verb_lemmas}

**NER results (entities)**:
{entities}
"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    _few_shot_prompt,
    ("human", _HUMAN),
])


class SRL:
    def __init__(self):
        llm = get_llm()
        self._chain = _prompt | llm.with_structured_output(_RawSRLOutput)

    def label(
        self,
        text: str,
        entities: list[Entity],
        verb_lemmas: list[str],
    ) -> list[SRLFrame]:
        # verb_lemmas는 문서 전체에서 등장한 동사 원형의 중복 제거 리스트 (프롬프트 참고용
        # 힌트일 뿐, LLM이 문장에 실제 등장한 동사를 우선 사용하도록 유도하는 강제 조건은 아님)
        verb_lemmas_str = ", ".join(verb_lemmas) if verb_lemmas else "없음"

        registered_predicates_str = ", ".join(sorted(_REGISTERED_PREDICATES))

        entity_lines = [
            f"- {e.text} ({e.label})" for e in entities
        ]
        entities_str = "\n".join(entity_lines) if entity_lines else "없음"

        # 그라운딩 검증용 조회 테이블: 이후 subject/object 텍스트가 실제 NER 결과에
        # 존재하는지, 존재한다면 라벨이 무엇인지 확인하는 유일한 소스가 된다
        entity_label_by_text = {e.text.strip(): e.label for e in entities}

        result = self._chain.invoke({
            "text": text,
            "registered_predicates": registered_predicates_str,
            "verb_lemmas": verb_lemmas_str,
            "entities": entities_str,
        })

        frames: list[SRLFrame] = []
        for raw_frame in result.frames:
            # 술어 사전 미등록 술어는 즉시 제외 (LLM이 프롬프트 지시를 어기고 새 술어를 지어냈을 가능성에 대한 가드레일)
            if raw_frame.predicate not in _REGISTERED_PREDICATES:
                continue

            # 그라운딩 체크: subject/object 텍스트가 NER 결과에 없으면(None) 할루시네이션으로
            # 간주하고 프레임 전체를 버린다. 라벨은 LLM이 아니라 항상 NER 결과에서만 가져온다.
            subject_label = entity_label_by_text.get(raw_frame.subject.strip())
            object_label = entity_label_by_text.get(raw_frame.object.strip())
            if subject_label is None or object_label is None:
                continue

            frames.append(
                SRLFrame(
                    predicate=raw_frame.predicate,
                    is_negated=raw_frame.is_negated,
                    tense=raw_frame.tense,
                    subject=Entity(text=raw_frame.subject.strip(), label=subject_label),
                    object=Entity(text=raw_frame.object.strip(), label=object_label),
                )
            )

        return frames
