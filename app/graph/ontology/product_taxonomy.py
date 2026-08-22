"""Category dictionary access.

data/seed/product_taxonomy.json is the single store for the dictionary: ``aliases`` are
hand-written seeds, ``learned_aliases`` are what ProductLinker resolved at runtime. Keeping
both in one file means "why did this item land in this category" is answered in one place.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TAXONOMY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "seed" / "product_taxonomy.json"
)

# Surface forms inherited from the legacy gazetteers that are unsafe as substring lookups.
# flashtext's non-word-boundary set is ASCII-only, so a Korean alias matches inside longer
# words: "밀" fires on 정밀, "요소" on the everyday word for "element", "놀라" is debris left
# over from 카놀라. An alias hit skips the LLM stage, so a false hit is unrecoverable while a
# miss is not — the LLM still sees the source sentence. Anything ambiguous belongs here.
UNSAFE_ALIASES: frozenset[str] = frozenset({
    "밀", "쌀", "황", "놀라", "요소", "주석", "고무", "대두", "경유",
})


def load_taxonomy(path: Path | None = None) -> list[dict]:
    """Return the category list, newest contents read from disk each call."""
    target = path or TAXONOMY_PATH
    payload = json.loads(target.read_text(encoding="utf-8"))
    return payload["categories"]


def all_aliases(category: dict) -> list[str]:
    """Seed aliases plus everything ProductLinker has learned, minus the unsafe ones."""
    return [
        alias
        for alias in (*category["aliases"], *category["learned_aliases"])
        if alias not in UNSAFE_ALIASES
    ]


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
        # Write to a sibling temp file and os.replace() onto the target so the swap is atomic.
        # This file is hand-authored and cannot be regenerated, and this write runs once per
        # article in a batch job: a process interrupted mid-write of the real file would leave
        # truncated JSON behind with no way to recover it.
        tmp_path = target.with_name(f"{target.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(tmp_path, target)
