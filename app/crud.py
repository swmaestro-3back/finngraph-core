# Neo4j CRUD
from __future__ import annotations

from collections import defaultdict
from typing import get_args

from app.core.db import neo4j_database
from app.graph.models import EntityLabel, Triplet
from app.graph.ontology.predicate_dict import PREDICATE_DICT

# Map NER tags to Neo4j labels (COMPANY -> Company, COUNTRY -> Country)
# (COMPANY → Company, COUNTRY → Country)
_TYPE_TO_LABEL: dict[EntityLabel, str] = {
    label: "".join(part.capitalize() for part in label.split("_"))
    for label in get_args(EntityLabel)
}

# Predicates whose triplet is split into two edges through the item node
_ITEM_DECOMPOSITION: dict[str, tuple[str, str]] = {
    "SUPPLIES_TO": ("SUPPLIES", "SUPPLIED_TO"),
    "EXPORTS_TO": ("EXPORTS", "EXPORTED_TO"),
}

# Maximum provenance entries kept per edge; older ones are evicted FIFO
_MAX_PROVENANCE = 10


def _edge_specs(triplet: Triplet) -> list[tuple[str, str, str, str, str]]:
    """
    Turn one triplet into (subject_label, subject_name, rel, object_label, object_name) edges

    Without an item this is a single (subject)-[predicate]->(object) edge. With an item,
    _ITEM_DECOMPOSITION splits it into (subject-item) and (item-object), so the item is the
    object of the first edge and the subject of the second.
    """

    # Convert NER labels to Neo4j labels
    subject_label = _TYPE_TO_LABEL[triplet.subject.label]
    object_label = _TYPE_TO_LABEL[triplet.object.label]

    if triplet.item is None:
        # The predicate is already whitelisted upstream, but it is interpolated straight into
        # the relationship type, so check it once more here.
        if triplet.predicate not in PREDICATE_DICT:
            return []
        return [(subject_label, triplet.subject.text, triplet.predicate, object_label, triplet.object.text)]

    decomposition = _ITEM_DECOMPOSITION.get(triplet.predicate)
    # With an item but no decomposition rule, drop the item and store a binary edge
    if decomposition is None:
        if triplet.predicate not in PREDICATE_DICT:
            return []
        return [(subject_label, triplet.subject.text, triplet.predicate, object_label, triplet.object.text)]

    item_label = _TYPE_TO_LABEL[triplet.item.label]
    rel_subject_item, rel_item_object = decomposition
    return [
        (subject_label, triplet.subject.text, rel_subject_item, item_label, triplet.item.text),
        (item_label, triplet.item.text, rel_item_object, object_label, triplet.object.text),
    ]


async def upsert_triplets(news_id: str, triplets: list[Triplet]) -> None:
    """
    Write extracted triplets to Neo4j

    1. Predicates carrying an item argument are split into two edges.
    2. The same triplet reported again under the same news_id is ignored.
    3. Each edge keeps at most _MAX_PROVENANCE entries, evicting the oldest first.
    4. Every edge tracks first_mentioned_at, last_mentioned_at and mention_count.
    """

    # Group edges sharing a (subject_label, rel, object_label) signature so each group needs a
    # single UNWIND + MERGE. When several sentences in one article converge on the same edge,
    # only the first is kept as its evidence.
    # The query's is_dup guard only sees news_ids as they were when the query started (the
    # planner inserts an Eager between WITH and SET, evaluating is_dup for every row up front),
    # so duplicates within a batch have to be filtered here instead.
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    seen_edges: set[tuple[str, str, str, str, str]] = set()
    for triplet in triplets:
        for subject_label, subject_name, rel, object_label, object_name in _edge_specs(triplet):
            edge_key = (subject_label, subject_name, rel, object_label, object_name)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            grouped[(subject_label, rel, object_label)].append(
                {
                    "subject_name": subject_name,
                    "object_name": object_name,
                    "source_sentence": triplet.source_sentence,
                }
            )

    for (subject_label, rel, object_label), rows in grouped.items():
        await neo4j_database.execute(
            f"""
            UNWIND $rows AS row
            MERGE (s:{subject_label} {{name: row.subject_name}})
            MERGE (o:{object_label} {{name: row.object_name}})
            MERGE (s)-[r:{rel}]->(o)
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
                r.source_sentences = CASE
                    WHEN is_dup THEN r.source_sentences
                    WHEN at_cap THEN r.source_sentences[1..] + row.source_sentence
                    ELSE coalesce(r.source_sentences, []) + row.source_sentence END,
                r.mentioned_ats = CASE
                    WHEN is_dup THEN r.mentioned_ats
                    WHEN at_cap THEN r.mentioned_ats[1..] + date()
                    ELSE coalesce(r.mentioned_ats, []) + date() END
            """,
            {"rows": rows, "news_id": news_id, "max_provenance": _MAX_PROVENANCE},
        )
