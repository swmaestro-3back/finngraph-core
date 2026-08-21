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

    # Create the driver matching the URI scheme (BoltDriver or Neo4jDriver). It owns the
    # credentials and the connection pool shared by every session.
    async def init_driver(self):
        # Reuse the driver if it was already initialized
        if self._driver:
            return
        try:
            self._driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
        except Exception as e:
            print(f"Error occurred while initializing Neo4j driver: {e}")

    # execute_query sits above the session API and closes the session for us
    async def execute(self, query: LiteralString, parameters: Optional[dict] = None) -> list[Record]:
        if not self._driver:
            raise RuntimeError("Neo4j Driver is not initialized. Call init_driver first.")
        
        records, _, _ = await self._driver.execute_query(
            query,
            parameters_=parameters,
            database_=settings.NEO4J_DATABASE
        )

        # execute_query returns (records, summary, keys); only records are needed here
        return records

neo4j_database = Neo4jDatabase()