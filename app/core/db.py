from typing import LiteralString, Optional

from neo4j import AsyncGraphDatabase, Record

from app.core.config import settings

# Driver
# Interface between Software and Database

# BoltDriver
# It addresses a single database machine. This may be a standalone server or could be a specific member of a cluster.
# Connections established by a BoltDriver are always made to the exact host and port detailed in the URI.

# Neo4jDriver
# The routing behaviour works in tandem with Neo4j’s Causal Clustering feature by directing read and write behaviour to appropriate cluster members.

class Neo4jDatabase:
    # Constructor
    def __init__(self):
        self._driver = None

    # URI에 맞는 Driver 생성 (BoltDriver 또는 Neo4jDriver)
    # DB Connection을 생성하기 위한 인증 정보 및 Connection Pool 관리
    async def init_driver(self):
        try:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
        except Exception as e:
            print(f"Error occurred while initializing Neo4j driver: {e}")

    # execute_query가 get_session으로 쿼리날리는 것보다 더 고수준 api
    # session.close를 알아서 해준다
    async def execute(self, query: LiteralString, parameters: Optional[dict] = None) -> list[Record]:
        if not self._driver:
            raise RuntimeError("Neo4j Driver is not initialized. Call init_driver first.")
        
        records, _, _ = await self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=settings.NEO4J_DATABASE
        )

        # 원래 records, summary, keys 이렇게 3개 주는데 지금은 쿼리 결과인 records만 사용하니 나머지는 버리는 용으로 _ 표기
        return records

neo4j_database = Neo4jDatabase()