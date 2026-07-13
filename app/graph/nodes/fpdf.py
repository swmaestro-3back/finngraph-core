import json
from pathlib import Path

from app.graph.models import SRLFrame, Triple

def _normalize_label(label: str | None) -> str | None:
    """LLM이 None 대신 문자열 "null"/"none"을 반환하는 경우를 정규화한다."""
    if label is None or label.lower() in ("null", "none", ""):
        return None
    return label


DICT_PATH = (
    Path(__file__).parent.parent.parent.parent / "data" / "dictionaries" / "predicate_dict.json"
)

class FPDF:
    def __init__(self):
        with open(DICT_PATH, encoding="utf-8") as f:
            self._predicate_dict: dict = json.load(f)

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

        # 조건 3: 확정된 과거/현재 사실이 아닌 프레임 제거 (미래·계획·추측·가능성 표현)
        if frame.tense != "past_or_present_fact":
            return None

        subject_label = _normalize_label(frame.subject.label)
        object_label = _normalize_label(frame.object.label)

        # 조건 4: subject(행위자) 개체명 타입이 술어의 subject 목록에 없으면 제거.
        # 목록이 빈 리스트면 타입 제약이 없는 술어이므로 통과시킨다.
        agent_types = entry.get("subject", [])
        if agent_types and subject_label is not None and subject_label not in agent_types:
            return None

        # 조건 4: object(피행위자) 개체명 타입이 술어의 object 목록에 없으면 제거 (위와 동일한 규칙)
        theme_types = entry.get("object", [])
        if theme_types and object_label is not None and object_label not in theme_types:
            return None

        return Triple(
            subject=frame.subject.text,
            subject_type=subject_label or "",
            predicate=frame.predicate,
            object=frame.object.text,
            object_type=object_label or "",
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
            and f.tense != "past_or_present_fact"
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
