from app.graph.models import Entity, RelationFrame
from app.graph.nodes.triplet_builder import TripletBuilder


def _frame(**overrides) -> RelationFrame:
    base = {
        "subject": Entity(text="에코프로비엠", label="COMPANY"),
        "object": Entity(text="삼성SDI", label="COMPANY"),
        "item": Entity(text="양극재", label="COMMODITY"),
        "predicate": "SUPPLIES_TO",
        "source_sentence": "에코프로비엠은 삼성SDI에 양극재를 공급한다.",
        "clause": "에코프로비엠은 삼성SDI에 양극재를 공급한다.",
        "evidence": "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다.",
        "polarity": "affirmed",
        "tense": "future_or_planned",
    }
    base.update(overrides)
    return RelationFrame(**base)


def test_denied_frame_is_no_longer_dropped():
    builder = TripletBuilder()

    triplets = builder.filter([_frame(polarity="denied")])

    assert len(triplets) == 1
    assert triplets[0].polarity == "denied"


def test_terminated_frame_is_no_longer_dropped():
    builder = TripletBuilder()

    triplets = builder.filter([_frame(polarity="terminated")])

    assert len(triplets) == 1
    assert triplets[0].polarity == "terminated"


def test_triplet_carries_evidence_and_tense():
    builder = TripletBuilder()

    triplet = builder.filter([_frame()])[0]

    assert triplet.evidence == "에코프로비엠은 올 하반기부터 삼성SDI에 양극재를 공급한다."
    assert triplet.tense == "future_or_planned"
    assert triplet.source_sentence == "에코프로비엠은 삼성SDI에 양극재를 공급한다."


def test_unregistered_predicate_is_still_dropped():
    builder = TripletBuilder()

    assert builder.filter([_frame(predicate="NOT_A_PREDICATE")]) == []


def test_subject_type_violation_is_still_dropped():
    builder = TripletBuilder()

    # SUPPLIES_TO의 supplier는 COMPANY만 허용된다
    frame = _frame(subject=Entity(text="미국", label="COUNTRY"))

    assert builder.filter([frame]) == []


def test_frames_differing_only_in_polarity_are_both_kept():
    builder = TripletBuilder()

    triplets = builder.filter([_frame(polarity="affirmed"), _frame(polarity="terminated")])

    assert len(triplets) == 2


def test_stats_counts_polarity_and_tense():
    builder = TripletBuilder()

    stats = builder.stats(
        [
            _frame(polarity="affirmed", tense="future_or_planned"),
            _frame(polarity="denied", tense="past_or_present_fact"),
            _frame(polarity="terminated", tense="past_or_present_fact"),
        ]
    )

    assert stats["total_frames"] == 3
    assert stats["polarity_counts"] == {"affirmed": 1, "denied": 1, "terminated": 1}
    assert stats["tense_counts"] == {
        "past_or_present_fact": 2,
        "future_or_planned": 1,
        "modal_possibility": 0,
    }
    assert stats["passed"] == 3
    assert "filtered_negated" not in stats
