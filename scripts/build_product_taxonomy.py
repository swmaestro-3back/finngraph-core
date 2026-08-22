"""One-shot: fill product_taxonomy.json aliases from the legacy gazetteer snapshot.

Reads data/seed/canonical_to_category.json (hand-authored) and copies every canonical and
its surface forms into the aliases list of the category it maps to.

Run: uv run python scripts/build_product_taxonomy.py
"""

import json
from pathlib import Path

SEED = Path("data/seed")
legacy = json.loads((SEED / "legacy_product_dict.json").read_text(encoding="utf-8"))
mapping = json.loads((SEED / "canonical_to_category.json").read_text(encoding="utf-8"))
taxonomy_path = SEED / "product_taxonomy.json"
payload = json.loads(taxonomy_path.read_text(encoding="utf-8"))

by_id = {category["id"]: category for category in payload["categories"]}
seen: dict[str, str] = {}
dropped: list[tuple[str, str, str]] = []

for section in ("product", "commodity"):
    for canonical, surfaces in legacy[section].items():
        category_id = mapping[canonical]
        category = by_id[category_id]
        for surface in [canonical, *surfaces]:
            owner = seen.get(surface)
            if owner == category_id:
                continue
            if owner is not None:
                # An alias must belong to exactly one category; report instead of guessing.
                dropped.append((surface, owner, category_id))
                continue
            category["aliases"].append(surface)
            seen[surface] = category_id

taxonomy_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"aliases written: {len(seen)}")
for surface, owner, other in dropped:
    print(f"  CONFLICT {surface!r}: kept in {owner}, also mapped to {other}")
