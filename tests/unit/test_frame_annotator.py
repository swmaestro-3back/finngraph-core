from app.graph.models import CandidateFrame, Entity, RawAnnotation
from app.graph.nodes.frame_annotator import format_candidates, merge_annotations

_CANDIDATE = CandidateFrame(
    subject=Entity(text="에코프로비엠", label="COMPANY"),
    object=Entity(text="삼성SDI", label="COMPANY"),
    item=Entity(text="양극재", label="COMMODITY"),
    predicate="SUPPLIES_TO",
    source_sentence="이 회사는 올 하반기부터 삼성SDI에 양극재를 공급한다.",
    clause="에코프로비엠은 삼성SDI에 양극재를 공급한다.",
)

_EVIDENCE = "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다."


def _annotation(**overrides) -> RawAnnotation:
    base = {
        "frame_index": 0,
        "subject": "에코프로비엠",
        "predicate": "SUPPLIES_TO",
        "object": "삼성SDI",
        "evidence": _EVIDENCE,
        "polarity": "affirmed",
        "tense": "future_or_planned",
    }
    base.update(overrides)
    return RawAnnotation(**base)


def test_valid_annotation_is_merged():
    frames, stats = merge_annotations([_CANDIDATE], [_annotation()])

    assert len(frames) == 1
    frame = frames[0]
    assert frame.evidence == _EVIDENCE
    assert frame.polarity == "affirmed"
    assert frame.tense == "future_or_planned"
    # CandidateFrame의 필드는 그대로 보존된다
    assert frame.predicate == "SUPPLIES_TO"
    assert frame.source_sentence == _CANDIDATE.source_sentence
    assert frame.clause == _CANDIDATE.clause
    assert stats == {"dropped_annotation_mismatch": 0, "dropped_evidence_grounding": 0}


def test_echo_mismatch_is_dropped():
    frames, stats = merge_annotations([_CANDIDATE], [_annotation(object="LG에너지솔루션")])

    assert frames == []
    assert stats["dropped_annotation_mismatch"] == 1
    assert stats["dropped_evidence_grounding"] == 0


def test_missing_annotation_is_dropped():
    frames, stats = merge_annotations([_CANDIDATE], [])

    assert frames == []
    assert stats["dropped_annotation_mismatch"] == 1


def test_out_of_range_index_does_not_annotate_any_frame():
    frames, stats = merge_annotations([_CANDIDATE], [_annotation(frame_index=7)])

    assert frames == []
    assert stats["dropped_annotation_mismatch"] == 1


def test_evidence_missing_subject_is_dropped():
    frames, stats = merge_annotations(
        [_CANDIDATE], [_annotation(evidence="삼성SDI에 양극재를 공급하기로 했다.")]
    )

    assert frames == []
    assert stats["dropped_evidence_grounding"] == 1
    assert stats["dropped_annotation_mismatch"] == 0


def test_evidence_missing_item_is_dropped():
    frames, stats = merge_annotations(
        [_CANDIDATE], [_annotation(evidence="에코프로비엠은 삼성SDI에 소재를 공급한다.")]
    )

    assert frames == []
    assert stats["dropped_evidence_grounding"] == 1


def test_empty_evidence_is_dropped():
    frames, stats = merge_annotations([_CANDIDATE], [_annotation(evidence="   ")])

    assert frames == []
    assert stats["dropped_evidence_grounding"] == 1


def test_too_long_evidence_is_dropped():
    long_evidence = _EVIDENCE + "그리고 " * 100
    frames, stats = merge_annotations([_CANDIDATE], [_annotation(evidence=long_evidence)])

    assert frames == []
    assert stats["dropped_evidence_grounding"] == 1


def test_evidence_grounding_ignores_whitespace():
    frames, _ = merge_annotations(
        [_CANDIDATE],
        [_annotation(evidence="에코프로비엠 은 올 하반기부터 삼성 SDI 에 양극재 를 공급한다.")],
    )

    assert len(frames) == 1


def test_annotations_are_matched_by_index_not_order():
    second = CandidateFrame(
        subject=Entity(text="에코프로비엠", label="COMPANY"),
        object=Entity(text="포스코케미칼", label="COMPANY"),
        item=None,
        predicate="PARTNERS_WITH",
        source_sentence="에코프로비엠은 지난달 포스코케미칼과의 합작법인을 청산했다.",
        clause="에코프로비엠은 포스코케미칼과 협력한다.",
    )
    second_annotation = _annotation(
        frame_index=1,
        object="포스코케미칼",
        predicate="PARTNERS_WITH",
        evidence="에코프로비엠은 지난달 포스코케미칼과의 합작법인을 청산했다.",
        polarity="terminated",
        tense="past_or_present_fact",
    )

    # 순서를 뒤집어 넣어도 frame_index로 매칭되어야 한다
    frames, stats = merge_annotations(
        [_CANDIDATE, second], [second_annotation, _annotation()]
    )

    assert len(frames) == 2
    assert frames[0].polarity == "affirmed"
    assert frames[1].polarity == "terminated"
    assert stats["dropped_annotation_mismatch"] == 0


def test_format_candidates_includes_index_and_fields():
    formatted = format_candidates([_CANDIDATE])

    assert "[0]" in formatted
    assert "에코프로비엠" in formatted
    assert "SUPPLIES_TO" in formatted
    assert "삼성SDI" in formatted
    assert "양극재" in formatted
    assert _CANDIDATE.source_sentence in formatted
    assert _CANDIDATE.clause in formatted
