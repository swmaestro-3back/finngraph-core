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
        # polarity(affirmed/denied/terminated)는 여기서 필터링하지 않는다.
        # 부정형이라고 간선을 지우지 않고, 라벨로 구분해 UI에서 함께 보여주는 것이 제품 방향이다.

        # 조건 1: 술어 사전 미등록 술어 제거
        entry = self._predicate_dict.get(frame.predicate)
        if entry is None:
            return None

        # PREDICATE_DICT_NARY는 argument를 subject/object(/item) 순서로 등록해두므로,
        # dict 순서를 그대로 역할 순서로 사용한다.
        arg_names = list(entry["arguments"].keys())
        agent_key, counterparty_key = arg_names[0], arg_names[1]
        item_key = arg_names[2] if len(arg_names) > 2 else None

        # 조건 2: subject(행위자) 개체명 타입이 술어의 subject argument 타입 목록에 없으면 제거.
        # 목록이 빈 리스트면 타입 제약이 없는 술어이므로 통과시킨다.
        agent_types = entry["arguments"][agent_key]["types"]
        if agent_types and frame.subject.label not in agent_types:
            return None

        # 조건 3: object(피행위자) 개체명 타입 검증 (위와 동일한 규칙)
        counterparty_types = entry["arguments"][counterparty_key]["types"]
        if counterparty_types and frame.object.label not in counterparty_types:
            return None

        # item은 optional argument이므로, 술어에 item argument가 없거나 타입이 맞지 않으면
        # 프레임 전체를 버리지 않고 item만 비워서 통과시킨다.
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

        # 한 번도 등장하지 않은 라벨도 0으로 남도록 모든 값을 미리 채워둔다.
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
