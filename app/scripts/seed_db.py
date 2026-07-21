"""data/seed의 country.json, krx.json, us.json을 읽어 docs/neo4j_schema.md 스키마에 맞게 Neo4j에 적재한다.

실행: python -m app.scripts.seed_db
"""

import asyncio
import json
from pathlib import Path

from app.core.db import neo4j_database

SEED_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "seed"


def _load(filename: str) -> list[dict]:
    with open(SEED_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


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
        MERGE (c:Country {iso_alp2: row.iso_alp2})
        SET c.name = row.name, c.iso_num = row.iso_num
        """,
        {"rows": rows},
    )
    print(f"Country: {len(rows)} nodes merged")


async def seed_krx_companies() -> None:
    # KOSDAQ GLOBAL은 KOSDAQ 라벨로 합쳐서 저장한다.
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


async def main() -> None:
    await neo4j_database.init_driver()
    await seed_countries()
    await seed_krx_companies()
    await seed_us_companies()


if __name__ == "__main__":
    asyncio.run(main())
