from app.crud import build_edge_rows
from app.graph.models import Entity, ProductRef, Triplet


def _triplet(**overrides) -> Triplet:
    base = {
        "subject": Entity(text="에코프로비엠"),
        "predicate": "SUPPLIES_TO",
        "object": Entity(text="삼성SDI"),
        "item": ProductRef(text="하이니켈 양극재", category="양극재"),
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


def test_row_carries_item_text_and_category():
    row = build_edge_rows([_triplet()])["SUPPLIES_TO"][0]

    assert row["subject_name"] == "에코프로비엠"
    assert row["object_name"] == "삼성SDI"
    assert row["item_text"] == "하이니켈 양극재"
    assert row["category"] == "양극재"


def test_unclassified_item_keeps_the_edge_without_a_category():
    triplet = _triplet(item=ProductRef(text="2층 전동차 개조작업", category=None))

    row = build_edge_rows([triplet])["SUPPLIES_TO"][0]

    assert row["item_text"] == "2층 전동차 개조작업"
    assert row["category"] is None


def test_itemless_predicate_produces_null_item_fields():
    triplet = _triplet(predicate="ACQUIRES", item=None)

    row = build_edge_rows([triplet])["ACQUIRES"][0]

    assert row["item_text"] is None
    assert row["category"] is None


def test_unregistered_predicate_is_dropped():
    assert build_edge_rows([_triplet(predicate="PARTNERS_WITH")]) == {}


def test_same_pair_with_different_items_yields_two_rows():
    rows = build_edge_rows(
        [
            _triplet(),
            _triplet(item=ProductRef(text="단결정 양극재", category="양극재")),
        ]
    )

    assert len(rows["SUPPLIES_TO"]) == 2


def test_identical_rows_are_deduplicated():
    rows = build_edge_rows([_triplet(), _triplet()])

    assert len(rows["SUPPLIES_TO"]) == 1
