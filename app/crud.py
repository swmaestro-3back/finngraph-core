# Neo4j CRUD
from __future__ import annotations

from collections import defaultdict

from app.core.db import neo4j_database
from app.graph.models import Triplet
from app.graph.ontology.predicate_dict import PREDICATE_DICT

# Maximum provenance entries kept per edge; older ones are evicted FIFO
_MAX_PROVENANCE = 10

# Maximum distinct raw item phrases kept per edge; older ones are evicted FIFO. Frequently
# reported pairs would otherwise grow this array without bound.
_MAX_ITEM_TEXTS = 20


def build_edge_rows(triplets: list[Triplet]) -> dict[str, list[dict]]:
    """
    Group triplets into one row list per predicate

    Items are NOT split into their own node. A three-argument relation reified through a shared
    product node loses the pairing between supplier and recipient: two supply relations passing
    through the same product node imply a path that neither of them states. The item therefore
    travels as a property of the single (subject)-[predicate]->(object) edge.

    Rows identical in every field are dropped, so one article restating the same fact does not
    inflate the provenance arrays.
    """

    grouped: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple] = set()

    for triplet in triplets:
        # The predicate is whitelisted upstream, but it is interpolated straight into the
        # relationship type, so check it once more here.
        if triplet.predicate not in PREDICATE_DICT:
            continue

        item_text = triplet.item.text if triplet.item is not None else None
        category = triplet.item.category if triplet.item is not None else None
        row = {
            "subject_name": triplet.subject.text,
            "object_name": triplet.object.text,
            "evidence": triplet.evidence,
            "polarity": triplet.polarity,
            "tense": triplet.tense,
            "item_text": item_text,
            "category": category,
        }

        key = (triplet.predicate, *row.values())
        if key in seen:
            continue
        seen.add(key)
        grouped[triplet.predicate].append(row)

    return dict(grouped)


async def upsert_triplets(news_id: str, triplets: list[Triplet]) -> None:
    """
    Write extracted triplets to Neo4j

    1. Every triplet is one (subject)-[predicate]->(object) edge between Company nodes.
    2. The same triplet reported again under the same news_id is ignored.
    3. Each edge keeps at most _MAX_PROVENANCE provenance entries (news_ids, evidences,
       polarities, tenses, mentioned_ats), evicting the oldest first. The five arrays are
       written together, so entry i of each describes the same mention.
    4. SUPPLIES_TO edges also accumulate item_texts (raw phrases, capped and FIFO-evicted) and
       categories (taxonomy ids, deduplicated and uncapped).
    5. Every edge tracks first_mentioned_at, last_mentioned_at and mention_count.
    """

    # The query's is_dup guard only sees news_ids as they were when the query started (the
    # planner inserts an Eager between WITH and SET, evaluating is_dup for every row up front),
    # so duplicates within a batch are filtered by build_edge_rows instead.
    for predicate, rows in build_edge_rows(triplets).items():
        await neo4j_database.execute(
            f"""
            UNWIND $rows AS row
            MERGE (s:Company {{name: row.subject_name}})
            MERGE (o:Company {{name: row.object_name}})
            MERGE (s)-[r:{predicate}]->(o)
            WITH r, row,
                 $news_id IN coalesce(r.news_ids, []) AS is_dup,
                 size(coalesce(r.news_ids, [])) >= $max_provenance AS at_cap
            SET r.first_mentioned_at = coalesce(r.first_mentioned_at, date()),
                r.last_mentioned_at = CASE WHEN is_dup THEN coalesce(r.last_mentioned_at, date())
                                           ELSE date() END,
                r.mention_count = coalesce(r.mention_count, 0)
                    + (CASE WHEN is_dup THEN 0 ELSE 1 END),
                r.news_ids = CASE
                    WHEN is_dup THEN r.news_ids
                    WHEN at_cap THEN r.news_ids[1..] + $news_id
                    ELSE coalesce(r.news_ids, []) + $news_id END,
                r.evidences = CASE
                    WHEN is_dup THEN r.evidences
                    WHEN at_cap THEN r.evidences[1..] + row.evidence
                    ELSE coalesce(r.evidences, []) + row.evidence END,
                r.polarities = CASE
                    WHEN is_dup THEN r.polarities
                    WHEN at_cap THEN r.polarities[1..] + row.polarity
                    ELSE coalesce(r.polarities, []) + row.polarity END,
                r.tenses = CASE
                    WHEN is_dup THEN r.tenses
                    WHEN at_cap THEN r.tenses[1..] + row.tense
                    ELSE coalesce(r.tenses, []) + row.tense END,
                r.mentioned_ats = CASE
                    WHEN is_dup THEN r.mentioned_ats
                    WHEN at_cap THEN r.mentioned_ats[1..] + date()
                    ELSE coalesce(r.mentioned_ats, []) + date() END,
                r.item_texts = CASE
                    WHEN row.item_text IS NULL THEN coalesce(r.item_texts, [])
                    WHEN row.item_text IN coalesce(r.item_texts, []) THEN r.item_texts
                    WHEN size(coalesce(r.item_texts, [])) >= $max_item_texts
                        THEN r.item_texts[1..] + row.item_text
                    ELSE coalesce(r.item_texts, []) + row.item_text END,
                r.categories = CASE
                    WHEN row.category IS NULL THEN coalesce(r.categories, [])
                    WHEN row.category IN coalesce(r.categories, []) THEN r.categories
                    ELSE coalesce(r.categories, []) + row.category END
            """,
            {
                "rows": rows,
                "news_id": news_id,
                "max_provenance": _MAX_PROVENANCE,
                "max_item_texts": _MAX_ITEM_TEXTS,
            },
        )
