"""Category dictionary access.

data/seed/product_taxonomy.json is the single store for the dictionary: ``aliases`` are
hand-written seeds, ``learned_aliases`` are what ProductLinker resolved at runtime. Keeping
both in one file means "why did this item land in this category" is answered in one place.
"""

from __future__ import annotations

import json
from pathlib import Path

TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "seed" / "product_taxonomy.json"
)


def load_taxonomy(path: Path | None = None) -> list[dict]:
    """Return the category list, newest contents read from disk each call."""
    target = path or TAXONOMY_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload["categories"]


def all_aliases(category: dict) -> list[str]:
    """Seed aliases plus everything ProductLinker has learned for this category."""
    return [*category["aliases"], *category["learned_aliases"]]


def append_learned_aliases(mapping: dict[str, str], path: Path | None = None) -> None:
    """Record {item_text: category_id} pairs so the next run resolves them without an LLM.

    Unknown category ids and aliases already present are skipped rather than raising: this
    runs inside the pipeline and must never take the run down.
    """
    target = path or TAXONOMY_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    by_id = {category["id"]: category for category in payload["categories"]}
    known = {
        alias
        for category in payload["categories"]
        for alias in all_aliases(category)
    }

    changed = False
    for item_text, category_id in mapping.items():
        category = by_id.get(category_id)
        if category is None or item_text in known:
            continue
        category["learned_aliases"].append(item_text)
        known.add(item_text)
        changed = True

    if changed:
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
