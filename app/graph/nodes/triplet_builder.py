from typing import get_args

from app.graph.models import Entity, Polarity, RelationFrame, Tense, Triplet
from app.graph.ontology.predicate_dict import PREDICATE_DICT


class TripletBuilder:
    def __init__(self):
        self._predicate_dict: dict = PREDICATE_DICT

    def build(self, relation_frames: list[RelationFrame]) -> list[Triplet]:
        """
        Validate frames against the ontology, convert them to triplets and drop duplicates
        """
        triples: list[Triplet] = []
        seen: set[str] = set()
        for frame in relation_frames:
            triple = self._to_triplet(frame)
            if triple is None:
                continue
            key = triple.model_dump_json()
            if key in seen:
                continue
            seen.add(key)
            triples.append(triple)
        return triples

    def _to_triplet(self, frame: RelationFrame) -> Triplet | None:
        """
        Convert one frame to a triplet, or None if the ontology rejects it
        """
        # Polarity is deliberately not filtered here. A denied or terminated relation keeps its
        # edge and is distinguished by its label, so the UI can show the full history.

        # Reject unregistered predicates
        entry = self._predicate_dict.get(frame.predicate)
        if entry is None:
            return None

        # Arguments in PREDICATE_DICT_NARY are ordered by role (subject, object, optional item)
        arg_names = list(entry["arguments"].keys())
        agent_key, counterparty_key = arg_names[0], arg_names[1]
        item_key = arg_names[2] if len(arg_names) > 2 else None

        # Validate subject entity type (empty type list means unrestricted)
        agent_types = entry["arguments"][agent_key]["types"]
        if agent_types and frame.subject.label not in agent_types:
            return None

        # Validate object entity type
        counterparty_types = entry["arguments"][counterparty_key]["types"]
        if counterparty_types and frame.object.label not in counterparty_types:
            return None

        # Item is optional: clear it on type mismatch instead of dropping the entire frame
        item: Entity | None = None
        if item_key is not None and frame.item is not None:
            item_types = entry["arguments"][item_key]["types"]
            if not item_types or frame.item.label in item_types:
                item = frame.item

        return Triplet(
            subject=frame.subject,
            predicate=frame.predicate,
            object=frame.object,
            item=item,
            source_sentence=frame.source_sentence,
            evidence=frame.evidence,
            polarity=frame.polarity,
            tense=frame.tense,
        )

    def stats(self, relation_frames: list[RelationFrame]) -> dict:
        """
        Report per-stage counts for debugging and evaluation
        """
        total = len(relation_frames)
        not_in_dict = sum(
            1 for f in relation_frames if f.predicate not in self._predicate_dict
        )

        polarity_counts: dict[str, int] = {polarity: 0 for polarity in get_args(Polarity)}
        tense_counts: dict[str, int] = {tense: 0 for tense in get_args(Tense)}
        for frame in relation_frames:
            polarity_counts[frame.polarity] += 1
            tense_counts[frame.tense] += 1

        passed = len(self.build(relation_frames))

        return {
            "total_frames": total,
            "filtered_not_in_dict": not_in_dict,
            "polarity_counts": polarity_counts,
            "tense_counts": tense_counts,
            "passed": passed,
        }
