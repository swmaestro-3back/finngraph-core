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
    # frames보다 먼저 선언: structured output은 필드 선언 순서대로 채워지므로, LLM이 frames를
    # 채우기 전에 절 분해를 먼저 수행하도록 강제한다 (병렬 목적어 분리, 종속절 인식, 대용어 해소를
    # 유도하는 chain-of-thought 필드). 이후 로직에서는 사용하지 않고 버린다.
    clauses: List[str] = Field(
        description=(
            "Before extracting frames, decompose the text into independent simple clauses. "
            "Split coordinate objects/subjects joined by '와/과/및/그리고' into separate clauses "
            "(one per entity). Split subordinate/causal/coordinate sentence connections "
            "(e.g., '~하기 때문에', '~하여', '~하면서') into separate clauses, one per underlying "
            "event. Resolve pronouns and referential expressions (e.g., '이를', '그것') to the "
            "entity they refer to. Each clause should be a short standalone sentence with an "
            "explicit subject and object."
        )
    )
    frames: List[_RawSRLFrame] = Field(description="List of extracted triplets, each containing subject, predicate, and object.")

_PREDICATE_DICT_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "dictionaries" / "predicate_dict.json"
)
with open(_PREDICATE_DICT_PATH, encoding="utf-8") as _f:
    _PREDICATE_DICT: dict = json.load(_f)
_REGISTERED_PREDICATES: set[str] = set(_PREDICATE_DICT.keys())

# 술어 이름만 나열하지 않고 description/subject_role/object_role을 함께 조립해 프롬프트에
# 제공한다. 타입 제약(subject/object)만으로는 방향을 구분할 수 없는 근접 동의어나 특수 비즈니스
# 용어(예: 공급하다 vs 구매하다, 낙찰받다 vs 입찰하다)를 LLM이 헷갈리지 않도록 돕는 역할 설명이다.
# 문서 자체가 요청당 변하지 않으므로 모듈 로드 시 한 번만 조립해둔다.
_REGISTERED_PREDICATES_STR = "\n".join(
    f"- {predicate}: {entry['description']} "
    f"(subject={entry['subject_role']}, object={entry['object_role']})"
    for predicate, entry in sorted(_PREDICATE_DICT.items())
)

_SYSTEM = """\
[Role]
You are an expert Information Extraction system specialized in Korean economic and financial news.
Your sole task is to extract multilateral business relationships among entities such as 'COMPANY, GOVERNMENT, COUNTRY, and MATERIAL/PRODUCT' in the form of pure, structured triplets: (subject, predicate, object).

[Core Principles]
0. Clause Decomposition First (clauses field): Before extracting frames, always fill the "clauses" field by decomposing the text into independent simple clauses. Coordinate arguments joined by '와/과/및/그리고' (e.g., "양극재와 음극재") must be split into one clause per entity, never treated as a single combined argument. Subordinate/causal clauses (e.g., "~하기 때문에", "~하여") that themselves describe a relationship must be split out as their own clause, not discarded as background context. Resolve pronouns (e.g., "이를") to their referent entity before extracting frames from that clause.
1. Strict Triplet Structure: Every extracted frame must contain exactly three elements: "subject", "predicate", and "object". Never extract any additional arguments, modifiers, temporal information (time/date), monetary amounts, or percentage shares.
2. Arguments Must Match NER Entities: The values of "subject" and "object" MUST exactly match the surface forms provided in the "NER Results (entities)" list below. Never use terms (e.g., general nouns, abstract concepts, or unrecognized words) that are not present in the given "entities" list. If an entity is not available in the list to fill either the subject or the object, do not extract that frame. Do not invent any values.
3. Predicate Constraint: The "predicate" MUST be chosen strictly from the "registered_predicates" list. Do not create new verbs or relationship names.
4. Multiple Triplet Separation (Mandatory): If a single sentence contains multiple facts or relationships — including one frame per clause produced in step 0 — never combine them into a single frame. Extract them as independent frames and include all of them in the "frames" list.
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
                "clauses": [
                    "삼성전자는 테슬라에 공급한다.",
                    "삼성전자는 엔비디아에 공급한다.",
                    "삼성전자는 코델코로부터 리튬을 구매했다.",
                    "코델코는 리튬을 생산한다.",
                    "코델코는 칠레에 위치한다.",
                ],
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
                "clauses": [
                    "구글은 페이스북을 인수할 계획이다.",
                    "구글은 유튜브를 인수했다.",
                    "테슬라는 리비안을 인수할 수도 있다.",
                ],
                "frames": [
                    {"predicate": "인수하다", "is_negated": False, "tense": "future_or_planned", "subject": "구글", "object": "페이스북"},
                    {"predicate": "인수하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "구글", "object": "유튜브"},
                    {"predicate": "인수하다", "is_negated": False, "tense": "modal_possibility", "subject": "테슬라", "object": "리비안"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        # 병렬 구조: 와/과로 묶인 목적어를 하나로 뭉치지 않고 개별 프레임으로 분리 (docs/korean_triplet_extraction.md 3번)
        "text": "에코프로비엠은 올 하반기부터 삼성SDI에 배터리 핵심 소재인 양극재와 음극재를 동시에 공급한다.",
        "entities": "에코프로비엠(COMPANY), 삼성SDI(COMPANY), 양극재(COMMODITY), 음극재(COMMODITY)",
        "output": json.dumps(
            {
                "clauses": [
                    "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다.",
                    "에코프로비엠은 올 하반기부터 삼성SDI에 음극재를 공급한다.",
                ],
                "frames": [
                    {"predicate": "공급하다", "is_negated": False, "tense": "future_or_planned", "subject": "에코프로비엠", "object": "삼성SDI"},
                    {"predicate": "공급하다", "is_negated": False, "tense": "future_or_planned", "subject": "에코프로비엠", "object": "양극재"},
                    {"predicate": "공급하다", "is_negated": False, "tense": "future_or_planned", "subject": "에코프로비엠", "object": "음극재"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        # 종속절(원인절) 내부에 숨은 관계 추출 + 대용어("이를") 해소 (docs/korean_triplet_extraction.md 4번)
        "text": "TSMC가 3나노 최첨단 반도체를 대량 생산하기 때문에, 애플이 이를 전량 수입하여 차세대 아이폰을 시장에 출시할 수 있었다.",
        "entities": "TSMC(COMPANY), 애플(COMPANY), 3나노 반도체(COMMODITY), 아이폰(PRODUCT)",
        "output": json.dumps(
            {
                "clauses": [
                    "TSMC가 3나노 반도체를 생산한다.",
                    "애플이 TSMC로부터 3나노 반도체를 수입한다.",
                ],
                "frames": [
                    {"predicate": "생산하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "TSMC", "object": "3나노 반도체"},
                    {"predicate": "공급하다", "is_negated": False, "tense": "past_or_present_fact", "subject": "TSMC", "object": "애플"},
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
    ) -> list[SRLFrame]:
        entity_lines = [
            f"- {e.text} ({e.label})" for e in entities
        ]
        entities_str = "\n".join(entity_lines) if entity_lines else "없음"

        # 그라운딩 검증용 조회 테이블: 이후 subject/object 텍스트가 실제 NER 결과에
        # 존재하는지, 존재한다면 라벨이 무엇인지 확인하는 유일한 소스가 된다
        entity_label_by_text = {e.text.strip(): e.label for e in entities}

        result = self._chain.invoke({
            "text": text,
            "registered_predicates": _REGISTERED_PREDICATES_STR,
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
