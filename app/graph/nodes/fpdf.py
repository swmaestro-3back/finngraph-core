from app.graph.models import SRLFrame, Triple
from app.graph.ontology.predicate_dict import PREDICATE_DICT

def _normalize_label(label: str | None) -> str | None:
    """LLM이 None 대신 문자열 "null"/"none"을 반환하는 경우를 정규화한다."""
    if label is None or label.lower() in ("null", "none", ""):
        return None
    return label

class FPDF:
    def __init__(self):
        self._predicate_dict: dict = PREDICATE_DICT

    def filter(self, srl_output: list[SRLFrame]) -> list[Triple]:
        triples: list[Triple] = []
        for frame in srl_output:
            triple = self._process_frame(frame)
            if triple is not None:
                triples.append(triple)
        return triples

    def _process_frame(self, frame: SRLFrame) -> Triple | None:
        # 조건 1: 부정 표현 프레임 제거
        if frame.is_negated:
            return None

        # 조건 2: 술어 사전 미등록 술어 제거
        entry = self._predicate_dict.get(frame.predicate)
        if entry is None:
            return None

        # 조건 3: 추측·가능성 표현(modal_possibility) 프레임 제거.
        # future_or_planned는 "~할 계획/예정/방침/전망이다"처럼 확정된 사실이므로 통과시키고,
        # "~할 수도 있다"류의 modal_possibility만 확정되지 않은 것으로 간주해 걸러낸다.
        if frame.tense == "modal_possibility":
            return None

        # PREDICATE_DICT_NARY는 argument를 subject/object(/item) 순서로 등록해두므로,
        # dict 순서를 그대로 역할 순서로 사용한다 (첫 번째=subject, 두 번째=object, 세 번째=item).
        arg_names = list(entry["arguments"].keys())
        agent_key, counterparty_key = arg_names[0], arg_names[1]
        item_key = arg_names[2] if len(arg_names) > 2 else None

        subject_label = _normalize_label(frame.subject.label)
        object_label = _normalize_label(frame.object.label)

        # 조건 4: subject(행위자) 개체명 타입이 술어의 subject argument 타입 목록에 없으면 제거.
        # 목록이 빈 리스트면 타입 제약이 없는 술어이므로 통과시킨다.
        agent_types = entry["arguments"][agent_key]["types"]
        if agent_types and subject_label is not None and subject_label not in agent_types:
            return None

        # 조건 4: object(피행위자) 개체명 타입이 술어의 object argument 타입 목록에 없으면 제거 (위와 동일한 규칙)
        counterparty_types = entry["arguments"][counterparty_key]["types"]
        if counterparty_types and object_label is not None and object_label not in counterparty_types:
            return None

        # item은 optional argument이므로, 술어에 item argument가 없거나 타입이 맞지 않으면
        # (subject/object와 달리) 프레임 전체를 버리지 않고 item만 비워서 통과시킨다.
        item_text: str | None = None
        item_label: str | None = None
        if item_key is not None and frame.item is not None:
            candidate_label = _normalize_label(frame.item.label)
            item_types = entry["arguments"][item_key]["types"]
            if candidate_label is not None and (not item_types or candidate_label in item_types):
                item_text = frame.item.text
                item_label = candidate_label

        return Triple(
            subject=frame.subject.text,
            subject_type=subject_label or "",
            predicate=frame.predicate,
            object=frame.object.text,
            object_type=object_label or "",
            item=item_text,
            item_type=item_label,
        )

    def stats(self, srl_output: list[SRLFrame]) -> dict:
        """FPDF 필터링 통계를 반환한다 (디버깅·평가용)."""
        total = len(srl_output)
        negated = sum(1 for f in srl_output if f.is_negated)
        not_in_dict = sum(
            1
            for f in srl_output
            if not f.is_negated and f.predicate not in self._predicate_dict
        )
        not_confirmed_fact = sum(
            1
            for f in srl_output
            if not f.is_negated
            and f.predicate in self._predicate_dict
            and f.tense == "modal_possibility"
        )
        passed = len(self.filter(srl_output))

        return {
            "total_frames": total,
            "filtered_negated": negated,
            "filtered_not_in_dict": not_in_dict,
            "filtered_not_confirmed_fact": not_confirmed_fact,
            # 조건 4(개체명 타입 불일치)로 걸러진 개수는 따로 세지 않고, filter()가 실제로
            # 통과시킨 개수를 뺀 나머지로 역산한다 (_process_frame의 로직을 중복 구현하지 않기 위함)
            "filtered_other": total - negated - not_in_dict - not_confirmed_fact - passed,
            "passed": passed,
        }
