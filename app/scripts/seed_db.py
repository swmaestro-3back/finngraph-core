"""
Load data/seed/{country,krx,us}.json into Neo4j following docs/neo4j_schema.md

seed() is the entry point; app.main calls it when the database is empty.
"""

import json
from pathlib import Path

from app.core.db import neo4j_database

SEED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "seed"

def _load(filename: str) -> list[dict]:
    with open(SEED_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


# Neo4j has no primary keys, and the internal ids it assigns to nodes and relationships are
# not ours to choose. A unique constraint is how a field is made to act like a primary key,
# and it creates the backing B-tree index for free.
async def create_constraints() -> None:
    # COMPANY TICKER
    await neo4j_database.execute(
        """
        CREATE CONSTRAINT company_ticker_unique IF NOT EXISTS
        FOR (c:Company) REQUIRE c.ticker IS UNIQUE
        """
    )

    # COUNTRY ISO_NUM
    await neo4j_database.execute(
        """
        CREATE CONSTRAINT country_iso_num_unique IF NOT EXISTS
        FOR (c:Country) REQUIRE c.iso_num IS UNIQUE
        """
    )
    print("Constraints created")


async def seed_countries() -> None:
    rows = [
        {
            "name": row["country_nm"],
            "iso_alp2": row["country_iso_alp2"],
            "iso_num": row["iso_num"],
        }
        for row in _load("country.json")
        if row["iso_num"].strip()
    ]

    await neo4j_database.execute(
        """
        UNWIND $rows AS row
        MERGE (c:Country {iso_num: row.iso_num})
        SET c.name = row.name, c.iso_alp2 = row.iso_alp2
        """,
        {"rows": rows},
    )
    print(f"Country: {len(rows)} nodes merged")


async def seed_krx_companies() -> None:
    # KOSDAQ GLOBAL is folded into the KOSDAQ label
    rows_by_label: dict[str, list[dict]] = {"KOSPI": [], "KOSDAQ": []}
    for row in _load("krx.json"):
        label = "KOSPI" if row["market"] == "KOSPI" else "KOSDAQ"
        rows_by_label[label].append({"ticker": row["ticker"], "name": row["name"]})

    for label, rows in rows_by_label.items():
        await neo4j_database.execute(
            f"""
            UNWIND $rows AS row
            MERGE (c:Company:{label} {{ticker: row.ticker}})
            SET c.name = row.name
            """,
            {"rows": rows},
        )
        print(f"Company:{label}: {len(rows)} nodes merged")


async def seed_us_companies() -> None:
    rows_by_label: dict[str, list[dict]] = {"NYSE": [], "NASDAQ": []}
    for row in _load("us.json"):
        rows_by_label[row["market"]].append(
            {
                "ticker": row["ticker"],
                "name": row["name"],
                "kr_name": row["kr_name"],
                "sp500": row["sp500"] == "true",
            }
        )

    for label, rows in rows_by_label.items():
        await neo4j_database.execute(
            f"""
            UNWIND $rows AS row
            MERGE (c:Company:{label} {{ticker: row.ticker}})
            SET c.name = row.name, c.kr_name = row.kr_name, c.sp500 = row.sp500
            """,
            {"rows": rows},
        )
        print(f"Company:{label}: {len(rows)} nodes merged")


async def seed():
    await neo4j_database.init_driver()

    # Skip if the database has already been seeded
    records = await neo4j_database.execute("MATCH (n) RETURN count(n) AS c")
    node_count = records[0]["c"]
    if node_count > 0:
        print(f"Skip seeding: {node_count} nodes already exist")
        return
    
    # Constraints must exist before the seed rows are inserted
    await create_constraints()

    await seed_countries()
    await seed_krx_companies()
    await seed_us_companies()