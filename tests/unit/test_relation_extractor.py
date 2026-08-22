from app.graph.models import Entity, ProductRef, RawRelation
from app.graph.nodes.relation_extractor import build_candidate_frames

_TEXT = "에코프로비엠은 올 하반기부터 삼성SDI에 하이니켈 양극재를 공급한다."
_ENTITIES = [
    Entity(text="에코프로비엠"),
    Entity(text="삼성SDI"),
]


def _raw(**overrides) -> RawRelation:
    base = {
        "source_sentence": _TEXT,
        "clause": "에코프로비엠은 삼성SDI에 하이니켈 양극재를 공급한다.",
        "predicate": "SUPPLIES_TO",
        "subject": "에코프로비엠",
        "object": "삼성SDI",
        "item": "하이니켈 양극재",
    }
    base.update(overrides)
    return RawRelation(**base)


def test_valid_frame_is_built_with_clause_preserved():
    frames = build_candidate_frames([_raw()], _ENTITIES, _TEXT)

    assert len(frames) == 1
    frame = frames[0]
    assert frame.subject == Entity(text="에코프로비엠")
    assert frame.object == Entity(text="삼성SDI")
    assert frame.predicate == "SUPPLIES_TO"
    assert frame.source_sentence == _TEXT
    assert frame.clause == "에코프로비엠은 삼성SDI에 하이니켈 양극재를 공급한다."


def test_item_absent_from_any_gazetteer_is_still_extracted():
    # The whole point of the redesign: no dictionary entry is required for the item.
    frames = build_candidate_frames([_raw()], _ENTITIES, _TEXT)

    assert frames[0].item == ProductRef(text="하이니켈 양극재", category=None)


def test_item_not_present_in_the_article_drops_the_frame():
    # SUPPLIES_TO declares item as required, so a hallucinated item kills the frame.
    frames = build_candidate_frames([_raw(item="전고체 배터리")], _ENTITIES, _TEXT)

    assert frames == []


def test_missing_required_item_drops_the_frame():
    frames = build_candidate_frames([_raw(item=None)], _ENTITIES, _TEXT)

    assert frames == []


def test_item_matches_ignoring_whitespace():
    frames = build_candidate_frames([_raw(item="하이니켈  양극재")], _ENTITIES, _TEXT)

    assert len(frames) == 1
    assert frames[0].item == ProductRef(text="하이니켈  양극재")


def test_item_is_ignored_for_predicates_without_a_product_slot():
    frames = build_candidate_frames(
        [
            _raw(
                predicate="ACQUIRES",
                item="하이니켈 양극재",
                clause="에코프로비엠은 삼성SDI를 인수한다.",
            )
        ],
        _ENTITIES,
        _TEXT,
    )

    assert len(frames) == 1
    assert frames[0].item is None


def test_unregistered_predicate_is_dropped():
    frames = build_candidate_frames([_raw(predicate="NOT_A_PREDICATE")], _ENTITIES, _TEXT)
    assert frames == []


def test_deleted_predicate_is_dropped():
    frames = build_candidate_frames([_raw(predicate="PARTNERS_WITH")], _ENTITIES, _TEXT)
    assert frames == []


def test_ungrounded_subject_is_dropped():
    frames = build_candidate_frames([_raw(subject="없는회사")], _ENTITIES, _TEXT)
    assert frames == []


def test_source_sentence_not_in_text_is_dropped():
    frames = build_candidate_frames(
        [_raw(source_sentence="이 문장은 원문에 없다.")], _ENTITIES, _TEXT
    )
    assert frames == []


def test_source_sentence_matches_ignoring_whitespace():
    frames = build_candidate_frames(
        [_raw(source_sentence="에코프로비엠은  올 하반기부터 삼성SDI에 하이니켈 양극재를 공급한다.")],
        _ENTITIES,
        _TEXT,
    )
    assert len(frames) == 1


def test_identical_frames_are_deduped():
    frames = build_candidate_frames([_raw(), _raw()], _ENTITIES, _TEXT)
    assert len(frames) == 1
