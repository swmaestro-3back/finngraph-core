import json
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.graph.models import (
    Entity,
    SRLFrame,
    TenseLabel,
)
from app.graph.ontology.predicate_dict_nary import PREDICATE_DICT_NARY

# LLM이 NER 결과와 무관하게 subject/object/item에 라벨을 지어낼 수 있으므로
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
    item: Optional[str] = Field(
        default=None,
        description=(
            "Must exactly match an entity from the provided NER results."
            "Leave null if the predicate has no item argument, or if the text does not name a specific item and the item argument is marked [optional]."
        ),
    )

class _RawSRLOutput(BaseModel):
    # frames보다 먼저 선언: structured output은 필드 선언 순서대로 채워지므로, LLM이 frames를
    # 채우기 전에 절 분해를 먼저 수행하도록 강제한다 (병렬 목적어 분리, 종속절 인식, 대용어 해소를
    # 유도하는 chain-of-thought 필드). 이후 로직에서는 사용하지 않고 버린다.
    clauses: List[str] = Field(
        description=(
            "Before extracting frames, decompose the text into independent simple clauses. "
            "Split coordinate objects/subjects joined by '와/과/및/그리고' into separate clauses (one per entity)."
            "Split subordinate/causal/coordinate sentence connections (e.g., '~하기 때문에', '~하여', '~하면서') into separate clauses, one per underlying event."
            "Resolve pronouns and referential expressions (e.g., '이를', '그것') to the entity they refer to."
            "Rephrase each clause, if needed, into the perspective that most directly matches a registered relationship (e.g. rephrase 'A buys X from B' as 'B supplies X to A') so the relationship it expresses is explicit."
            "Each clause should be a short standalone sentence with an explicit subject and object."
        )
    )
    frames: List[_RawSRLFrame] = Field(description="List of extracted frames, each containing subject, predicate, object, and (if applicable) item.")

_PREDICATE_DICT: dict = PREDICATE_DICT_NARY
_REGISTERED_PREDICATES: set[str] = set(_PREDICATE_DICT.keys())

# 술어 이름만 나열하지 않고 description과 argument(역할)별 설명을 함께 조립해 프롬프트에
# 제공한다. 타입 제약만으로는 방향을 구분할 수 없는 근접 동의어(예: ACQUIRES vs DIVESTS_FROM,
# WINS_CONTRACT_FROM vs PARTNERS_WITH)를 LLM이 헷갈리지 않도록 돕는 역할 설명이다.
# argument가 3개인 술어는 세 번째(item) 역할을 subject/object와 별도로 표시하고, dictionary에
# 기록된 required 값을 그대로 노출해 "item이 없으면 프레임 자체를 버려야 하는지(mandatory)"
# 아니면 "item 없이도 추출해도 되는지(optional)"를 LLM이 predicate별로 구분하게 한다.
# 문서 자체가 요청당 변하지 않으므로 모듈 로드 시 한 번만 조립해둔다.
def _format_predicate_entry(predicate: str, entry: dict) -> str:
    arg_names = list(entry["arguments"].keys())
    subject_role, object_role = arg_names[0], arg_names[1]
    subject_desc = entry["arguments"][subject_role]["description"]
    object_desc = entry["arguments"][object_role]["description"]
    line = f"- {predicate}: {entry['description']} (subject={subject_desc}, object={object_desc}"
    if len(arg_names) > 2:
        item_role = arg_names[2]
        item_entry = entry["arguments"][item_role]
        item_tag = "mandatory" if item_entry["required"] else "optional"
        line += f", item={item_entry['description']} [{item_tag}]"
    line += ")"
    return line

_REGISTERED_PREDICATES_STR = "\n".join(
    _format_predicate_entry(predicate, entry)
    for predicate, entry in sorted(_PREDICATE_DICT.items())
)

_SYSTEM = """\
### [Role]
You are an expert Information Extraction system specialized in Korean economic and financial news.
Your sole task is to extract multilateral business relationships among entities such as 'COMPANY, GOVERNMENT, COUNTRY, and MATERIAL/PRODUCT' in the form of structured frames: (subject, predicate, object), optionally with a fourth 'item' argument.

### [Core Principles]
0. Clause Decomposition First (clauses field)
    - Before extracting frames, always fill the "clauses" field by decomposing the text into independent simple clauses.
    - Coordinate arguments joined by '와/과/및/그리고' (e.g., "양극재와 음극재") must be split into one clause per entity, never treated as a single combined argument.
    - Subordinate/causal clauses (e.g., "~하기 때문에", "~하여") that themselves describe a relationship must be split out as their own clause, not discarded as background context.
    - Resolve pronouns (e.g., "이를") to their referent entity before extracting frames from that clause.
1. Semantic Relation Matching: Match each clause against a predicate's description and argument roles, not against a specific verb. A predicate applies whenever the relationship it describes holds between two grounded entities, however the text phrases it — synonym, passive form, nominalized phrase, or indirect wording all count equally.
   - When a clause is phrased from the counterparty's point of view, restate it in the matching predicate's direction before extracting. e.g. "A가 B로부터 X를 구매했다" restates as "B가 A에게 X를 공급한다" → extract SUPPLIES_TO(subject=B, object=A, item=X).
   - Extract implied relationships between two COMPANY/GOVERNMENT/COUNTRY entities whenever the surrounding context clearly presupposes them, even without a single verb connecting the two directly — e.g., two companies jointly founding a venture presupposes PARTNERS_WITH between them.
   - Extract a frame only when the relationship is actually stated or clearly implied. Entities that merely co-occur in a sentence with no relational content between them do not form a frame.
2. Strict Frame Structure: Every extracted frame must contain "subject", "predicate", and "object". Some predicates in the registered_predicates list additionally declare an "item" argument (shown as "item=... [mandatory]" or "item=... [optional]"). For those predicates, also fill "item" within the same frame (do not split it into a separate frame) whenever the text names a specific product/commodity. If a predicate's item argument is marked [mandatory], do not extract that frame at all unless a specific item entity is named in the text. Never extract any other additional arguments, modifiers, temporal information (time/date), monetary amounts, or percentage shares.
3. Arguments Must Match NER Entities: The values of "subject", "object", and "item" MUST exactly match the surface forms provided in the "NER Results (entities)" list below. Never use terms (e.g., general nouns, abstract concepts, or unrecognized words) that are not present in the given "entities" list. If an entity is not available in the list to fill subject or object, do not extract that frame. Do not invent any values.
4. Predicate Constraint: The "predicate" MUST be one of the exact strings in the "registered_predicates" list. Do not invent new relationship names.
5. Multiple Triplet Separation (Mandatory): If a single sentence contains multiple facts or relationships — including one frame per clause produced in step 0 — never combine them into a single frame. Extract them as independent frames and include all of them in the "frames" list. If a predicate has an "item" argument and the text lists multiple coordinate items (e.g. "양극재와 음극재를 공급한다"), extract one frame per item, each repeating the same subject and object.
6. Negation Handling (is_negated): Set "is_negated" to true if the relationship involves a negative expression (e.g., "does not", "fails to", "rejects", "denied"). Otherwise, set it to false.
7. Tense/Modality Classification (tense): Classify the temporal/modal status of the clause containing the predicate into exactly one of:
   - "past_or_present_fact": an already-completed past event, or an ongoing/habitual present-tense fact (including implied relationships presented as presently existing, e.g. "코델코는 칠레에 위치한 광산기업이다").
   - "future_or_planned": a future, planned, or scheduled action (e.g., "~할 계획이다", "~할 예정이다", "~할 방침이다", "~할 전망이다").
   - "modal_possibility": a speculative, hypothetical, or ability/possibility expression (e.g., "~할 수도 있다", "~할 수 있다", "~것으로 보인다", "~가능성이 있다").
   Always extract the frame regardless of its tense/modality classification — do not omit future/modal relationships. Downstream filtering relies on this field to keep only confirmed facts.

[Argument Type]
- subject: The active agent (COMPANY, GOVERNMENT, or COUNTRY).
- object: The target entity. (COMPANY, GOVERNMENT, COUNTRY, COMMODITY, PRODUCT)
- item: Only for predicates that declare an "item" role (see registered_predicates list) — the specific product/commodity/material (COMMODITY, PRODUCT) involved. Fill it whenever the text names one. If the predicate's item is [mandatory] and no item is named, do not extract that frame. If [optional], extract the frame anyway and leave item null.
"""

# Few-shot 예시: 아래 human/ai 메시지 쌍으로 대화 turn을 재현해 실제 입출력 형식을 시연한다.
# output은 값으로 주입되므로(템플릿 리터럴이 아님) JSON의 중괄호를 이스케이프할 필요가 없다.
_EXAMPLES = [
    {
        # 명시적 관계 동사("공급하다") 없이 "구매했다"만 등장 - 반대 방향(공급자 관점)으로
        # 재해석해서 SUPPLIES_TO를 추출해야 한다 (원칙 1의 방향 추론 예시).
        "text": "삼성전자는 칠레의 코델코로부터 리튬을 대량 구매했다.",
        "entities": "삼성전자(COMPANY), 칠레(COUNTRY), 코델코(COMPANY), 리튬(COMMODITY)",
        "output": json.dumps(
            {
                "clauses": [
                    "코델코가 삼성전자에 리튬을 공급한다.",
                    "코델코는 리튬을 생산한다.",
                    "코델코는 칠레에 위치한다.",
                ],
                "frames": [
                    {"predicate": "SUPPLIES_TO", "is_negated": False, "tense": "past_or_present_fact", "subject": "코델코", "object": "삼성전자", "item": "리튬"},
                    {"predicate": "PRODUCES", "is_negated": False, "tense": "past_or_present_fact", "subject": "코델코", "object": "리튬", "item": None},
                    {"predicate": "LOCATED_IN", "is_negated": False, "tense": "past_or_present_fact", "subject": "코델코", "object": "칠레", "item": None},
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
                    # ACQUIRES는 item argument가 없는 술어이므로 항상 item: null
                    {"predicate": "ACQUIRES", "is_negated": False, "tense": "future_or_planned", "subject": "구글", "object": "페이스북", "item": None},
                    {"predicate": "ACQUIRES", "is_negated": False, "tense": "past_or_present_fact", "subject": "구글", "object": "유튜브", "item": None},
                    {"predicate": "ACQUIRES", "is_negated": False, "tense": "modal_possibility", "subject": "테슬라", "object": "리비안", "item": None},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        # 병렬 목적어(item): 와/과로 묶인 품목을 하나로 뭉치지 않고 item별로 프레임을 분리하되,
        # 수령자(object)는 각 프레임에 동일하게 반복한다. SUPPLIES_TO의 item은 [mandatory]이므로
        # item이 없는 프레임은 애초에 만들지 않는다.
        "text": "에코프로비엠은 올 하반기부터 삼성SDI에 배터리 핵심 소재인 양극재와 음극재를 동시에 공급한다.",
        "entities": "에코프로비엠(COMPANY), 삼성SDI(COMPANY), 양극재(COMMODITY), 음극재(COMMODITY)",
        "output": json.dumps(
            {
                "clauses": [
                    "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다.",
                    "에코프로비엠은 올 하반기부터 삼성SDI에 음극재를 공급한다.",
                ],
                "frames": [
                    {"predicate": "SUPPLIES_TO", "is_negated": False, "tense": "future_or_planned", "subject": "에코프로비엠", "object": "삼성SDI", "item": "양극재"},
                    {"predicate": "SUPPLIES_TO", "is_negated": False, "tense": "future_or_planned", "subject": "에코프로비엠", "object": "삼성SDI", "item": "음극재"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        # 종속절(원인절) 내부에 숨은 관계 추출 + 대용어("이를") 해소. IMPORTS_FROM은 item argument를
        # 가지므로 "누가(importer)-누구로부터(supplying_party)-무엇을(item)"을 프레임 하나로 포착한다.
        "text": "TSMC가 3나노 최첨단 반도체를 대량 생산하기 때문에, 애플이 이를 전량 수입하여 차세대 아이폰을 시장에 출시할 수 있었다.",
        "entities": "TSMC(COMPANY), 애플(COMPANY), 3나노 반도체(COMMODITY), 아이폰(PRODUCT)",
        "output": json.dumps(
            {
                "clauses": [
                    "TSMC가 3나노 반도체를 생산한다.",
                    "애플이 TSMC로부터 3나노 반도체를 수입한다.",
                ],
                "frames": [
                    {"predicate": "PRODUCES", "is_negated": False, "tense": "past_or_present_fact", "subject": "TSMC", "object": "3나노 반도체", "item": None},
                    {"predicate": "IMPORTS_FROM", "is_negated": False, "tense": "past_or_present_fact", "subject": "애플", "object": "TSMC", "item": "3나노 반도체"},
                ]
            },
            ensure_ascii=False,
        ),
    },
    {
        # "협력"/"제휴"/"수주"/"낙찰" 같은 단어가 전혀 없어도, 맥락이 그 관계를 함의하면 추출해야 한다 (원칙 1).
        "text": "현대차와 LG에너지솔루션은 미국에 배터리 공장을 짓기 위한 합작법인을 설립하기로 했다.",
        "entities": "현대차(COMPANY), LG에너지솔루션(COMPANY), 미국(COUNTRY), 배터리(COMMODITY)",
        "output": json.dumps(
            {
                "clauses": [
                    "현대차는 LG에너지솔루션과 협력한다."
                ],
                "frames": [
                    {"predicate": "PARTNERS_WITH", "is_negated": False, "tense": "future_or_planned", "subject": "현대차", "object": "LG에너지솔루션", "item": None},
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

        # 그라운딩 검증용 조회 테이블: 이후 subject/object/item 텍스트가 실제 NER 결과에
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

            # item이 할루시네이션이어도 프레임 전체를 버리지 않고 item만 누락시킨다
            # (subject/object로 이루어진 관계 자체는 여전히 유효하기 때문). item이 mandatory인
            # 술어인데 결과적으로 None이 되는 경우, 현재 FPDF는 이를 별도로 걸러내지 않고
            # item=None인 채로 통과시킨다 — required 강제는 아직 프롬프트 레벨(위 [Core
            # Principles] 2번)에만 있고 FPDF에는 구현돼 있지 않다.
            item_entity: Entity | None = None
            if raw_frame.item is not None:
                item_label = entity_label_by_text.get(raw_frame.item.strip())
                if item_label is not None:
                    item_entity = Entity(text=raw_frame.item.strip(), label=item_label)

            frames.append(
                SRLFrame(
                    predicate=raw_frame.predicate,
                    is_negated=raw_frame.is_negated,
                    tense=raw_frame.tense,
                    subject=Entity(text=raw_frame.subject.strip(), label=subject_label),
                    object=Entity(text=raw_frame.object.strip(), label=object_label),
                    item=item_entity,
                )
            )

        return frames
