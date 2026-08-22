from app.graph.models import Entity, ProductRef, RelationFrame
from app.graph.nodes.product_linker import (
    apply_categories,
    build_alias_processor,
    match_category,
    pending_items,
)

_CATEGORIES = [
    {
        "id": "양극재",
        "domain": "2차전지",
        "kind": "제품",
        "definition": "배터리 양극 활물질과 그 전구체.",
        "aliases": ["양극재", "하이니켈 양극재"],
        "learned_aliases": [],
    },
    {
        "id": "반도체 장비",
        "domain": "반도체",
        "kind": "제품",
        "definition": "반도체 제조에 쓰이는 설비.",
        "aliases": ["증착 장비"],
        "learned_aliases": ["차세대 증착 장비"],
    },
    {
        "id": "배터리 셀",
        "domain": "2차전지",
        "kind": "제품",
        "definition": "완성된 배터리 셀.",
        "aliases": ["NCM"],
        "learned_aliases": [],
    },
]


def _frame(item_text: str | None) -> RelationFrame:
    return RelationFrame(
        subject=Entity(text="에코프로비엠"),
        object=Entity(text="삼성SDI"),
        item=None if item_text is None else ProductRef(text=item_text),
        predicate="SUPPLIES_TO",
        source_sentence="에코프로비엠은 삼성SDI에 공급한다.",
        clause="에코프로비엠은 삼성SDI에 공급한다.",
        evidence="에코프로비엠은 삼성SDI에 공급한다.",
        polarity="affirmed",
        tense="future_or_planned",
    )


def test_exact_alias_resolves_to_its_category():
    processor = build_alias_processor(_CATEGORIES)

    assert match_category(processor, "양극재") == "양극재"


def test_longest_alias_wins():
    processor = build_alias_processor(_CATEGORIES)

    # Both "양극재" and "하이니켈 양극재" are aliases; the longer span must win.
    assert match_category(processor, "하이니켈 양극재") == "양극재"


def test_equal_length_ties_go_to_the_rightmost_match():
    processor = build_alias_processor(_CATEGORIES)

    # "NCM" (배터리 셀) and "양극재" (양극재) are both 3-character aliases found in this
    # phrase, so their spans tie on length. The qualifier "NCM" appears first, but the head
    # noun "양극재" appears last and names what the item actually is; it must win the tie.
    assert match_category(processor, "단결정 NCM 양극재") == "양극재"


def test_learned_alias_is_a_cache_hit():
    processor = build_alias_processor(_CATEGORIES)

    assert match_category(processor, "차세대 증착 장비") == "반도체 장비"


def test_unknown_item_returns_none():
    processor = build_alias_processor(_CATEGORIES)

    assert match_category(processor, "2층 전동차 개조작업") is None


def test_pending_items_splits_resolved_from_unresolved():
    processor = build_alias_processor(_CATEGORIES)
    frames = [_frame("양극재"), _frame("2층 전동차 개조작업"), _frame(None)]

    resolved, unresolved = pending_items(frames, processor)

    assert resolved == {"양극재": "양극재"}
    assert [text for text, _ in unresolved] == ["2층 전동차 개조작업"]


def test_pending_items_deduplicates_repeated_item_text():
    processor = build_alias_processor(_CATEGORIES)
    frames = [_frame("2층 전동차 개조작업"), _frame("2층 전동차 개조작업")]

    _, unresolved = pending_items(frames, processor)

    assert len(unresolved) == 1


def test_apply_categories_fills_matching_frames_only():
    frames = [_frame("양극재"), _frame("2층 전동차 개조작업")]

    linked = apply_categories(frames, {"양극재": "양극재"})

    assert linked[0].item.category == "양극재"
    assert linked[1].item.category is None


def test_apply_categories_leaves_itemless_frames_alone():
    linked = apply_categories([_frame(None)], {"양극재": "양극재"})

    assert linked[0].item is None
