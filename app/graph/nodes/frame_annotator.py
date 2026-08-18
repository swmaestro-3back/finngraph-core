from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.graph.models import (
    CandidateFrame,
    RawAnnotation,
    RawAnnotationList,
    RelationFrame,
)
from app.graph.prompts.frame_annotation import PROMPT
from app.graph.utils.text import normalize_whitespace

# evidence 길이 상한. 이를 넘으면 문단을 통째로 복사한 것으로 보고 드롭한다.
_MAX_EVIDENCE_LENGTH = 200


def format_candidates(candidates: list[CandidateFrame]) -> str:
    """
    Render candidate frames as the numbered text block sent to the LLM

    The [n] prefix is the frame_index that the LLM must echo back
    """

    lines: list[str] = []
    for index, candidate in enumerate(candidates):
        item_text = candidate.item.text if candidate.item is not None else "없음"
        lines.append(
            f"[{index}] subject={candidate.subject.text} | predicate={candidate.predicate} "
            f"| object={candidate.object.text} | item={item_text}"
        )
        lines.append(f"    source_sentence: {candidate.source_sentence}")
        lines.append(f"    clause: {candidate.clause}")
    return "\n".join(lines)


def _is_grounded(evidence: str, candidate: CandidateFrame) -> bool:
    """
    Check whether an evidence sentence can stand as this frame's grounding

    Evidence is generated, so it cannot be matched against the source verbatim. Instead
    require the subject/object/item surface forms to appear in it, within a length cap.
    """

    if not evidence or len(evidence) > _MAX_EVIDENCE_LENGTH:
        return False

    required = [candidate.subject.text, candidate.object.text]
    if candidate.item is not None:
        required.append(candidate.item.text)

    normalized_evidence = normalize_whitespace(evidence)
    return all(normalize_whitespace(term) in normalized_evidence for term in required)


def merge_annotations(
    candidates: list[CandidateFrame],
    annotations: list[RawAnnotation],
) -> tuple[list[RelationFrame], dict[str, int]]:
    """
    Merge annotations into their frames, dropping any frame that fails to line up

    Filling a default instead would turn a denial into a fact, which costs far more
    than losing one relation.
    """

    # frame_index로 매칭하므로 LLM이 순서를 바꿔 돌려줘도 안전하다.
    # 같은 인덱스가 중복되면 첫 번째만 채택한다.
    annotation_by_index: dict[int, RawAnnotation] = {}
    for annotation in annotations:
        if annotation.frame_index in annotation_by_index:
            continue
        annotation_by_index[annotation.frame_index] = annotation

    frames: list[RelationFrame] = []
    dropped_mismatch = 0
    dropped_grounding = 0

    for index, candidate in enumerate(candidates):
        annotation = annotation_by_index.get(index)

        # 주석이 아예 없거나(누락), 범위 밖 인덱스만 돌아온 경우
        if annotation is None:
            dropped_mismatch += 1
            continue

        # 에코 대조: LLM이 인덱스를 어긋나게 매긴 경우를 잡는다
        if (
            annotation.subject.strip() != candidate.subject.text
            or annotation.predicate.strip() != candidate.predicate
            or annotation.object.strip() != candidate.object.text
        ):
            dropped_mismatch += 1
            continue

        evidence = annotation.evidence.strip()
        if not _is_grounded(evidence, candidate):
            dropped_grounding += 1
            continue

        frames.append(
            RelationFrame(
                subject=candidate.subject,
                object=candidate.object,
                item=candidate.item,
                predicate=candidate.predicate,
                source_sentence=candidate.source_sentence,
                clause=candidate.clause,
                evidence=evidence,
                polarity=annotation.polarity,
                tense=annotation.tense,
            )
        )

    stats = {
        "dropped_annotation_mismatch": dropped_mismatch,
        "dropped_evidence_grounding": dropped_grounding,
    }
    return frames, stats


class FrameAnnotator:
    def __init__(self):
        self._model = ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self._chain = PROMPT | self._model.with_structured_output(
            schema=RawAnnotationList,
            method="json_schema",
        )

    async def annotate(
        self,
        text: str,
        candidates: list[CandidateFrame],
    ) -> tuple[list[RelationFrame], dict[str, int]]:
        """
        Annotate every candidate frame in one batched LLM call
        """
        # 프레임이 없으면 LLM을 호출하지 않는다 (빈 목록에 대한 낭비 호출 방지).
        if not candidates:
            return [], {"dropped_annotation_mismatch": 0, "dropped_evidence_grounding": 0}

        invoke_input = {
            "text": text,
            "frames": format_candidates(candidates),
        }

        result = await self._chain.ainvoke(invoke_input)
        return merge_annotations(candidates, result.annotations)
