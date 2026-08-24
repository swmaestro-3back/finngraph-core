from app.crud import _predicate_has_item_slot, build_edge_rows
from app.graph.models import Entity, Triplet


def _triplet(**overrides) -> Triplet:
    base = {
        "subject": Entity(text="에코프로비엠"),
        "predicate": "SUPPLIES_TO",
        "object": Entity(text="삼성SDI"),
        "item": "하이니켈 양극재",
        "source_sentence": "에코프로비엠은 삼성SDI에 하이니켈 양극재를 공급한다.",
        "evidence": "에코프로비엠은 올 하반기부터 삼성SDI에 하이니켈 양극재를 공급한다.",
        "polarity": "affirmed",
        "tense": "future_or_planned",
    }
    base.update(overrides)
    return Triplet(**base)


def test_supply_triplet_becomes_one_direct_edge():
    rows = build_edge_rows([_triplet()])

    assert list(rows.keys()) == ["SUPPLIES_TO"]
    assert len(rows["SUPPLIES_TO"]) == 1


def test_row_carries_item_text():
    row = build_edge_rows([_triplet()])["SUPPLIES_TO"][0]

    assert row["subject_name"] == "에코프로비엠"
    assert row["object_name"] == "삼성SDI"
    assert row["item_texts"] == ["하이니켈 양극재"]


def test_itemless_predicate_produces_empty_item_lists():
    triplet = _triplet(predicate="ACQUIRES", item=None)

    row = build_edge_rows([triplet])["ACQUIRES"][0]

    assert row["item_texts"] == []


def test_unregistered_predicate_is_dropped():
    assert build_edge_rows([_triplet(predicate="PARTNERS_WITH")]) == {}


def test_same_pair_with_different_items_collapses_to_one_row():
    rows = build_edge_rows(
        [
            _triplet(),
            _triplet(item="단결정 양극재"),
        ]
    )

    assert len(rows["SUPPLIES_TO"]) == 1
    assert rows["SUPPLIES_TO"][0]["item_texts"] == ["하이니켈 양극재", "단결정 양극재"]


def test_identical_rows_are_deduplicated():
    rows = build_edge_rows([_triplet(), _triplet()])

    assert len(rows["SUPPLIES_TO"]) == 1
    assert rows["SUPPLIES_TO"][0]["item_texts"] == ["하이니켈 양극재"]


def test_predicate_has_item_slot_is_derived_from_predicate_dict():
    """SUPPLIES_TO declares a third PRODUCT argument; INVESTS_IN and ACQUIRES do not. The
    item_texts SET clause in upsert_triplets is only emitted for predicates where this
    returns True, so INVESTS_IN/ACQUIRES edges no longer get [] written onto them."""
    assert _predicate_has_item_slot("SUPPLIES_TO") is True
    assert _predicate_has_item_slot("INVESTS_IN") is False
    assert _predicate_has_item_slot("ACQUIRES") is False
    assert _predicate_has_item_slot("NOT_A_REAL_PREDICATE") is False


def test_first_frame_owns_the_provenance_of_a_merged_edge():
    rows = build_edge_rows(
        [
            _triplet(evidence="첫 번째 근거 문장"),
            _triplet(
                item="단결정 양극재",
                evidence="두 번째 근거 문장",
            ),
        ]
    )

    assert rows["SUPPLIES_TO"][0]["evidence"] == "첫 번째 근거 문장"
