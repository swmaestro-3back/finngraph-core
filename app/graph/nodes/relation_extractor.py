from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.graph.models import (
    CandidateFrame,
    Entity,
    RawRelation,
    RawRelationList,
)
from app.graph.ontology.predicate_dict import REGISTERED_PREDICATES
from app.graph.prompts.relation_extraction import PROMPT
from app.graph.utils.text import normalize_whitespace


def build_candidate_frames(
    raw_frames: list[RawRelation],
    entities: list[Entity],
    text: str,
) -> list[CandidateFrame]:
    """
    Validate raw LLM output and assemble grounded relation candidates

    Pure function, kept apart from the LLM call so it can be unit tested without an API key
    """

    # Lookup table for grounding: the only source of truth for whether a subject/object/item
    # string exists in the NER output, and which label it carries.
    entity_label_by_text = {e.text.strip(): e.label for e in entities}
    normalized_text = normalize_whitespace(text)

    frames: list[CandidateFrame] = []
    for raw_frame in raw_frames:
        # Guardrail for predicate hallucination
        if raw_frame.predicate not in REGISTERED_PREDICATES:
            continue

        # Discard if subject or object is not in extracted entities
        subject_label = entity_label_by_text.get(raw_frame.subject.strip())
        object_label = entity_label_by_text.get(raw_frame.object.strip())
        if subject_label is None or object_label is None:
            continue

        # Discard item if not in extracted entities
        item_entity: Entity | None = None
        if raw_frame.item is not None:
            item_label = entity_label_by_text.get(raw_frame.item.strip())
            if item_label is not None:
                item_entity = Entity(text=raw_frame.item.strip(), label=item_label)

        # Discard if source_sentence is not in the article
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

    # Deduplicate frames
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
        """
        Extract relation candidates from the article
        """
        entity_lines = [f"- {e.text} ({e.label})" for e in entities]
        entities_str = "\n".join(entity_lines) if entity_lines else "없음"

        # The predicate dictionary and few-shot examples already live in PROMPT's fixed system
        # prefix, so text and entities are the only per-request variables.
        invoke_input = {
            "text": text,
            "entities": entities_str,
        }

        # The call is network I/O, so await it rather than blocking the event loop
        result = await self._chain.ainvoke(invoke_input)
        return build_candidate_frames(result.frames, entities, text)
