import json
from pathlib import Path

from app.graph.ontology.product_taxonomy import (
    TAXONOMY_PATH,
    UNSAFE_ALIASES,
    all_aliases,
    append_learned_aliases,
    load_taxonomy,
)

_LEGACY = Path("data/seed/legacy_product_dict.json")
_MAPPING = Path("data/seed/canonical_to_category.json")

_REQUIRED_FIELDS = {"id", "domain", "kind", "definition", "aliases", "learned_aliases"}


def test_every_category_has_all_required_fields():
    for category in load_taxonomy():
        assert _REQUIRED_FIELDS <= set(category), category.get("id")
        assert category["definition"].strip(), category["id"]


def test_category_ids_are_unique():
    ids = [category["id"] for category in load_taxonomy()]
    assert len(ids) == len(set(ids))


def test_an_alias_belongs_to_exactly_one_category():
    owner_by_alias: dict[str, str] = {}
    for category in load_taxonomy():
        for alias in all_aliases(category):
            assert alias not in owner_by_alias, (
                f"{alias!r} claimed by both {owner_by_alias.get(alias)} and {category['id']}"
            )
            owner_by_alias[alias] = category["id"]


def test_every_legacy_canonical_is_mapped_to_a_live_category():
    legacy = json.loads(_LEGACY.read_text(encoding="utf-8"))
    mapping = json.loads(_MAPPING.read_text(encoding="utf-8"))
    category_ids = {category["id"] for category in load_taxonomy()}

    canonicals = set(legacy["product"]) | set(legacy["commodity"])
    unmapped = canonicals - set(mapping)
    assert not unmapped, f"unmapped canonicals: {sorted(unmapped)}"

    unknown_targets = set(mapping.values()) - category_ids
    assert not unknown_targets, f"mapping points at unknown categories: {sorted(unknown_targets)}"


def test_every_legacy_surface_form_is_reachable_as_an_alias():
    """The dictionaries were deleted; their surface forms must survive as aliases.

    Forms in UNSAFE_ALIASES are exempt: they were dropped on purpose because they match as
    substrings inside unrelated words, not lost by accident.
    """
    legacy = json.loads(_LEGACY.read_text(encoding="utf-8"))
    known = {alias for category in load_taxonomy() for alias in all_aliases(category)}

    missing = []
    for section in ("product", "commodity"):
        for canonical, surfaces in legacy[section].items():
            for surface in [canonical, *surfaces]:
                if surface not in known and surface not in UNSAFE_ALIASES:
                    missing.append(surface)
    assert not missing, f"{len(missing)} surface forms lost, e.g. {missing[:10]}"


def test_taxonomy_path_points_at_the_seed_file():
    assert TAXONOMY_PATH == Path("data/seed/product_taxonomy.json").resolve()


def test_unsafe_aliases_are_never_written_back(tmp_path):
    """A deny-listed item_text must never be appended to learned_aliases, even if the LLM
    classifies it: otherwise it would be re-appended on every subsequent run forever, since
    the deny-list also hides it from the dedup check via all_aliases()."""
    category_id = "농축산물"
    payload = {
        "categories": [
            {
                "id": category_id,
                "domain": "농축산",
                "kind": "제품",
                "definition": "곡물 등 농축산 원자재.",
                "aliases": ["대두박"],
                "learned_aliases": [],
            }
        ]
    }
    target = tmp_path / "product_taxonomy.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    unsafe_alias = next(iter(UNSAFE_ALIASES))
    append_learned_aliases(
        {unsafe_alias: category_id, "카놀라유": category_id},
        path=target,
    )

    written = json.loads(target.read_text(encoding="utf-8"))
    learned = written["categories"][0]["learned_aliases"]
    assert learned == ["카놀라유"]
    assert unsafe_alias not in learned
