"""One-shot: dump the product/commodity gazetteers to JSON before they are deleted.

The dump is the raw material for data/seed/product_taxonomy.json (Task 5) and the
reference set for the coverage test that proves no canonical was silently dropped.

Run: uv run python scripts/snapshot_legacy_dicts.py
"""

import json
from pathlib import Path

from app.graph.ontology.gazetteers.commodity_dict import COMMODITY_DICT
from app.graph.ontology.gazetteers.product_dict import PRODUCT_DICT

OUT = Path("data/seed/legacy_product_dict.json")

payload = {
    "description": (
        "Snapshot of product_dict.py and commodity_dict.py taken before deletion. "
        "Every canonical here must appear as an alias of exactly one category in "
        "product_taxonomy.json."
    ),
    "product": PRODUCT_DICT,
    "commodity": COMMODITY_DICT,
}

OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"{OUT}: {len(PRODUCT_DICT)} product + {len(COMMODITY_DICT)} commodity canonicals")
