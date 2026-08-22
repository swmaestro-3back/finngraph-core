"""Attach a taxonomy category to each free-text product mention.

Three stages, cheapest first:
  1. alias lookup against the taxonomy (no LLM)
  2. one batched LLM call for whatever is left
  3. write the resolved pairs back as learned aliases, so stage 1 catches them next run

Items nothing resolves are logged for human review and left with category=None. The supply
relation itself still reaches the graph; only the category rollup is missing.
"""

from __future__ import annotations

import json
from pathlib import Path

from flashtext import KeywordProcessor
from langchain_aws import ChatBedrockConverse

from app.core.config import settings
from app.graph.models import ProductRef, RawItemCategoryList, RelationFrame
from app.graph.ontology.product_taxonomy import (
    all_aliases,
    append_learned_aliases,
    load_taxonomy,
)
from app.graph.prompts.product_linking import PROMPT

UNCLASSIFIED_LOG = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "review" / "unclassified_items.jsonl"
)


def build_alias_processor(categories: list[dict]) -> KeywordProcessor:
    """Map every alias (seed and learned) to its category id.

    Lookup only — unlike EntityExtractor's canonicalizer this never rewrites text, so the
    substring hazards that constrained the old product gazetteer do not apply.
    """
    processor = KeywordProcessor(case_sensitive=True)
    for category in categories:
        for alias in all_aliases(category):
            processor.add_keyword(alias, category["id"])
    return processor


def match_category(processor: KeywordProcessor, item_text: str) -> str | None:
    """Return the category of the longest alias found inside the item text.

    Ties are broken by whichever match ends furthest right. Korean noun compounds put the
    head noun last, so in "단결정 NCM 양극재" the trailing "양극재" names what the item IS,
    while the leading "NCM" only qualifies it. Without this rule the winner would depend on
    the accident of which alias appears first in the string.
    """
    spans = processor.extract_keywords(item_text, span_info=True)
    if not spans:
        return None
    category_id, _start, _end = max(spans, key=lambda span: (span[2] - span[1], span[2]))
    return category_id


def pending_items(
    frames: list[RelationFrame],
    processor: KeywordProcessor,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Split distinct item texts into alias hits and everything needing the LLM."""
    resolved: dict[str, str] = {}
    unresolved: list[tuple[str, str]] = []
    seen: set[str] = set()

    for frame in frames:
        if frame.item is None or frame.item.text in seen:
            continue
        seen.add(frame.item.text)
        category_id = match_category(processor, frame.item.text)
        if category_id is None:
            unresolved.append((frame.item.text, frame.source_sentence))
        else:
            resolved[frame.item.text] = category_id

    return resolved, unresolved


def apply_categories(
    frames: list[RelationFrame],
    mapping: dict[str, str],
) -> list[RelationFrame]:
    """Return frames whose items carry their category. Frames are not mutated in place."""
    linked: list[RelationFrame] = []
    for frame in frames:
        if frame.item is None:
            linked.append(frame)
            continue
        category_id = mapping.get(frame.item.text)
        linked.append(
            frame.model_copy(
                update={"item": ProductRef(text=frame.item.text, category=category_id)}
            )
        )
    return linked


def format_categories(categories: list[dict]) -> str:
    return "\n".join(
        f"- {category['id']} ({category['kind']}): {category['definition']}"
        for category in categories
    )


def format_items(items: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    for item_text, source_sentence in items:
        lines.append(f"- item_text: {item_text}")
        lines.append(f"  source_sentence: {source_sentence}")
    return "\n".join(lines)


def log_unclassified(items: list[tuple[str, str]], news_id: str) -> None:
    """Append review-queue rows. Kept out of Neo4j: a node per unknown string would rebuild
    exactly the node explosion this redesign removes."""
    if not items:
        return
    UNCLASSIFIED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with UNCLASSIFIED_LOG.open("a", encoding="utf-8") as handle:
        for item_text, source_sentence in items:
            handle.write(
                json.dumps(
                    {
                        "item_text": item_text,
                        "source_sentence": source_sentence,
                        "news_id": news_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class ProductLinker:
    def __init__(self):
        self._categories = load_taxonomy()
        self._processor = build_alias_processor(self._categories)
        self._category_ids = {category["id"] for category in self._categories}
        self._model = ChatBedrockConverse(
            model=settings.bedrock_chat_model,
            region_name=settings.bedrock_region,
            temperature=0,
        )
        self._chain = PROMPT | self._model.with_structured_output(
            schema=RawItemCategoryList,
            method="json_schema",
        )

    async def link(
        self,
        frames: list[RelationFrame],
        news_id: str,
    ) -> tuple[list[RelationFrame], dict[str, int]]:
        resolved, unresolved = pending_items(frames, self._processor)
        stats = {
            "linked_by_alias": len(resolved),
            "linked_by_llm": 0,
            "unclassified": 0,
        }

        if unresolved:
            result = await self._chain.ainvoke(
                {
                    "categories": format_categories(self._categories),
                    "items": format_items(unresolved),
                }
            )
            learned: dict[str, str] = {}
            for entry in result.items:
                # Guardrail: the category list is closed, so an unknown id is a hallucination.
                if entry.category_id in self._category_ids:
                    learned[entry.item_text] = entry.category_id

            resolved.update(learned)
            append_learned_aliases(learned)
            stats["linked_by_llm"] = len(learned)

            still_unknown = [pair for pair in unresolved if pair[0] not in resolved]
            log_unclassified(still_unknown, news_id)
            stats["unclassified"] = len(still_unknown)

        return apply_categories(frames, resolved), stats
