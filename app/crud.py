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


def _predicate_has_item_slot(predicate: str) -> bool:
    """Return whether the predicate declares a third 'item' argument in PREDICATE_DICT.

    Derived from the ontology rather than hardcoded to SUPPLIES_TO, the same way
    relation_extractor._product_item_slot derives it: adding a predicate with an item
    argument later needs no change here.
    """
    entry = PREDICATE_DICT.get(predicate)
    if entry is None:
        return False
    arguments = list(entry["arguments"].values())
    return len(arguments) >= 3 and arguments[2]["types"] == ["PRODUCT"]


def build_edge_rows(triplets: list[Triplet]) -> dict[str, list[dict]]:
    """
    Group triplets into one row per (subject, predicate, object), keyed by predicate

    Items are NOT split into their own node. A three-argument relation reified through a shared
    product node loses the pairing between supplier and recipient: two supply relations passing
    through the same product node imply a path that neither of them states. The item therefore
    travels as a property of the single (subject)-[predicate]->(object) edge.

    An edge gets at most one row per call even when the article named several items for it.
    Neo4j evaluates the is_dup guard for every row of an UNWIND before any SET runs, so two rows
    touching the same relationship would both see is_dup=false and append the same news_id
    twice. Merging here keeps provenance at one entry per article; the first frame to reach an
    edge owns that entry, and the items collected along the way ride along as lists.
    """

    grouped: dict[str, list[dict]] = defaultdict(list)
    row_by_edge: dict[tuple[str, str, str], dict] = {}

    for triplet in triplets:
        # The predicate is whitelisted upstream, but it is interpolated straight into the
        # relationship type, so check it once more here.
        if triplet.predicate not in PREDICATE_DICT:
            continue

        edge_key = (triplet.predicate, triplet.subject.text, triplet.object.text)
        row = row_by_edge.get(edge_key)
        if row is None:
            row = {
                "subject_name": triplet.subject.text,
                "object_name": triplet.object.text,
                "evidence": triplet.evidence,
                "polarity": triplet.polarity,
                "tense": triplet.tense,
                "item_texts": [],
                "categories": [],
            }
            row_by_edge[edge_key] = row
            grouped[triplet.predicate].append(row)

        if triplet.item is None:
            continue
        if triplet.item.text not in row["item_texts"]:
            row["item_texts"].append(triplet.item.text)
        category = triplet.item.category
        if category is not None and category not in row["categories"]:
            row["categories"].append(category)

    return dict(grouped)


async def upsert_triplets(news_id: str, triplets: list[Triplet]) -> None:
    """
    Write extracted triplets to Neo4j

    1. Every triplet is one (subject)-[predicate]->(object) edge between Company nodes. One
       article contributes at most one provenance entry per edge, even if it names several
       items for that edge.
    2. The same triplet reported again under the same news_id is ignored.
    3. Each edge keeps at most _MAX_PROVENANCE provenance entries (news_ids, evidences,
       polarities, tenses, mentioned_ats), evicting the oldest first. The five arrays are
       written together, so entry i of each describes the same mention.
    4. SUPPLIES_TO edges also accumulate item_texts (raw phrases, value-deduped, capped and
       FIFO-evicted) and categories (taxonomy ids, value-deduped, uncapped).
    5. Every edge tracks first_mentioned_at, last_mentioned_at and mention_count.
    """

    # The query's is_dup guard only sees news_ids as they were when the query started (the
    # planner inserts an Eager between WITH and SET, evaluating is_dup for every row up front),
    # so duplicates within a batch are filtered by build_edge_rows instead.
    for predicate, rows in build_edge_rows(triplets).items():
        # item_texts/categories belong only on predicates that declare a product item
        # (SUPPLIES_TO today). Emitting them for INVESTS_IN/ACQUIRES would just write []
        # onto every such edge, since those triplets never carry an item.
        item_clause = (
            """,
                r.item_texts = reduce(
                    acc = coalesce(r.item_texts, []), t IN row.item_texts |
                    CASE
                        WHEN t IN acc THEN acc
                        WHEN size(acc) >= $max_item_texts THEN acc[1..] + t
                        ELSE acc + t END),
                r.categories = reduce(
                    acc = coalesce(r.categories, []), c IN row.categories |
                    CASE WHEN c IN acc THEN acc ELSE acc + c END)"""
            if _predicate_has_item_slot(predicate)
            else ""
        )
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
                    ELSE coalesce(r.mentioned_ats, []) + date() END{item_clause}
            """,
            {
                "rows": rows,
                "news_id": news_id,
                "max_provenance": _MAX_PROVENANCE,
                "max_item_texts": _MAX_ITEM_TEXTS,
            },
        )
