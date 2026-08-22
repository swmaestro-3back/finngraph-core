import json

from app.graph.prompts.relation_extraction import _EXAMPLES, _SYSTEM_MESSAGE

_DELETED = [
    "DIVESTS_FROM",
    "PARTNERS_WITH",
    "EXPORTS_TO",
    "LOCATED_IN",
    "PRODUCES",
    "COMPETES_WITH",
    "DEVELOPS",
    "SANCTIONS",
]


def test_system_message_lists_only_live_predicates():
    for predicate in _DELETED:
        assert predicate not in _SYSTEM_MESSAGE
    for predicate in ("SUPPLIES_TO", "INVESTS_IN", "ACQUIRES"):
        assert predicate in _SYSTEM_MESSAGE


def test_examples_use_only_live_predicates():
    for example in _EXAMPLES:
        for frame in json.loads(example["output"])["frames"]:
            assert frame["predicate"] in {"SUPPLIES_TO", "INVESTS_IN", "ACQUIRES"}


def test_an_example_demonstrates_an_item_outside_any_gazetteer():
    """The redesign only works if the model copies unseen product spans verbatim."""
    items = {
        frame["item"]
        for example in _EXAMPLES
        for frame in json.loads(example["output"])["frames"]
        if frame["item"] is not None
    }
    assert "차량용 5나노 AI 칩" in items


def test_examples_never_put_an_item_in_the_entity_list():
    """Entity lists carry companies only; items must not look gazetteer-anchored."""
    for example in _EXAMPLES:
        frames = json.loads(example["output"])["frames"]
        for frame in frames:
            if frame["item"] is not None:
                assert frame["item"] not in example["entities"]
