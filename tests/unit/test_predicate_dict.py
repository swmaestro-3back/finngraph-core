from app.graph.ontology.predicate_dict import PREDICATE_DICT, REGISTERED_PREDICATES


def test_only_three_predicates_are_registered():
    assert REGISTERED_PREDICATES == {"SUPPLIES_TO", "INVESTS_IN", "ACQUIRES"}


def test_anchor_slots_accept_company_only():
    for predicate, entry in PREDICATE_DICT.items():
        arg_names = list(entry["arguments"].keys())
        for role in arg_names[:2]:
            assert entry["arguments"][role]["types"] == ["COMPANY"], (
                f"{predicate}.{role} must be a COMPANY anchor slot"
            )


def test_supplies_to_is_the_only_predicate_with_a_product_item():
    with_item = {
        predicate
        for predicate, entry in PREDICATE_DICT.items()
        if len(entry["arguments"]) > 2
    }
    assert with_item == {"SUPPLIES_TO"}

    item_arg = list(PREDICATE_DICT["SUPPLIES_TO"]["arguments"].values())[2]
    assert item_arg["types"] == ["PRODUCT"]
    assert item_arg["required"] is True
