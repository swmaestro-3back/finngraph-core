from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.graph.models import (
    CandidateFrame,
    Entity,
    RawRelation,
    RawRelationList,
)
from app.graph.ontology.predicate_dict import PREDICATE_DICT
from app.graph.prompts.relation_extraction import PROMPT
from app.graph.utils.text import normalize_whitespace

# LLM이 프롬프트 지시를 어기고 미등록 술어를 지어냈을 때를 걸러내기 위한 검증용 술어 집합.
_REGISTERED_PREDICATES: set[str] = set(PREDICATE_DICT.keys())


def build_candidate_frames(
    raw_frames: list[RawRelation],
    entities: list[Entity],
    text: str,
) -> list[CandidateFrame]:
    """LLM 원시 출력을 검증해 관계 후보 프레임으로 조립한다.

    LLM 호출과 분리된 순수 함수라 API 키 없이 단위 테스트할 수 있다.
    """

    # 그라운딩 검증용 조회 테이블: subject/object/item 텍스트가 실제 NER 결과에
    # 존재하는지, 존재한다면 라벨이 무엇인지 확인하는 유일한 소스가 된다.
    entity_label_by_text = {e.text.strip(): e.label for e in entities}
    normalized_text = normalize_whitespace(text)

    frames: list[CandidateFrame] = []
    for raw_frame in raw_frames:
        # 술어 사전 미등록 술어는 즉시 제외 (LLM이 새 술어를 지어냈을 가능성에 대한 가드레일)
        if raw_frame.predicate not in _REGISTERED_PREDICATES:
            continue

        # subject/object가 NER 결과에 없으면 할루시네이션으로 간주하고 프레임 전체를 버린다.
        # 라벨은 LLM이 아니라 항상 NER 결과에서만 가져온다.
        subject_label = entity_label_by_text.get(raw_frame.subject.strip())
        object_label = entity_label_by_text.get(raw_frame.object.strip())
        if subject_label is None or object_label is None:
            continue

        # item이 할루시네이션이어도 프레임 전체를 버리지 않고 item만 누락시킨다
        # (subject/object로 이루어진 관계 자체는 여전히 유효하기 때문).
        item_entity: Entity | None = None
        if raw_frame.item is not None:
            item_label = entity_label_by_text.get(raw_frame.item.strip())
            if item_label is not None:
                item_entity = Entity(text=raw_frame.item.strip(), label=item_label)

        # 근거 문장 검증: 공백 정규화 후 원문에 substring으로 존재해야 통과.
        # 실패한 프레임은 근거 없는 관계로 간주하고 전체를 버린다.
        source_sentence = raw_frame.source_sentence.strip()
        if not source_sentence or normalize_whitespace(source_sentence) not in normalized_text:
            continue

        clause = raw_frame.clause.strip()

        frames.append(
            CandidateFrame(
                predicate=raw_frame.predicate,
                subject=Entity(text=raw_frame.subject.strip(), label=subject_label),
                object=Entity(text=raw_frame.object.strip(), label=object_label),
                item=item_entity,
                source_sentence=source_sentence,
                clause=clause,
            )
        )

    # 중복 제거: LLM이 동일한 관계를 여러 프레임으로 반복 추출할 수 있으므로 모든 필드가
    # 완전히 동일한 프레임은 하나만 남긴다. 최초 등장 순서는 보존한다.
    seen: set[str] = set()
    deduped: list[CandidateFrame] = []
    for frame in frames:
        key = frame.model_dump_json()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(frame)

    return deduped


class RelationExtractor:
    def __init__(self):
        self._model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self._chain = PROMPT | self._model.with_structured_output(
            schema=RawRelationList,
            method="json_schema",
        )

    async def extract(
        self,
        text: str,
        entities: list[Entity],
    ) -> list[CandidateFrame]:
        entity_lines = [f"- {e.text} ({e.label})" for e in entities]
        entities_str = "\n".join(entity_lines) if entity_lines else "없음"

        # 술어 사전·few-shot은 PROMPT의 고정 prefix(system 지시)에 이미 들어있으므로,
        # 요청마다 채우는 변수는 text/entities만이다.
        invoke_input = {
            "text": text,
            "entities": entities_str,
        }

        # LLM 호출은 네트워크 I/O라 ainvoke로 await 해서 이벤트 루프를 막지 않는다.
        result = await self._chain.ainvoke(invoke_input)
        return build_candidate_frames(result.frames, entities, text)
