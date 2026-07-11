# Neo4j CRUD
from __future__ import annotations

from app.core.db import neo4j_database
from app.schemas import (
    CompanyNetworkResponse,
    NetworkEdge,
    NetworkNode,
    NodeType,
    ThemeCompaniesResponse,
    ThemeCompanyItem,
)


def _node_identity(node) -> tuple[str, NodeType, str]:
    if "Company" in node.labels:
        return node["srtnCd"], "COMPANY", node["name"]
    return str(node["theme_id"]), "THEME", node["name"]


async def get_company_network(
    company_id: str, hops: int, filter_types: set[str]
) -> CompanyNetworkResponse | None:
    """companyId를 기준으로 BELONGS_TO 관계를 hops만큼 양방향 탐색해 서브그래프를 반환한다."""
    async with neo4j_database.get_session() as session:
        source_result = await session.run(
            "MATCH (c:Company {srtnCd: $companyId}) RETURN c.srtnCd AS id, c.name AS name",
            companyId=company_id,
        )
        source_record = await source_result.single()
        if source_record is None:
            return None

        # 가변 길이 경로의 범위는 Cypher 파라미터 바인딩이 불가능하므로, FastAPI Query(ge=1, le=3)로
        # 이미 검증된 int인 hops를 쿼리 문자열에 직접 삽입한다.
        path_result = await session.run(
            f"""
            MATCH (source:Company {{srtnCd: $companyId}})
            MATCH path = (source)-[:BELONGS_TO*1..{hops}]-(other)
            WHERE other <> source
            RETURN path
            """,
            companyId=company_id,
        )
        paths = [record["path"] async for record in path_result]

    nodes: dict[str, NetworkNode] = {
        company_id: NetworkNode(id=company_id, type="COMPANY", name=source_record["name"], hop=0)
    }
    edges: dict[tuple[str, str], NetworkEdge] = {}

    for path in paths:
        path_nodes = list(path.nodes)
        for hop, (prev_node, next_node) in enumerate(zip(path_nodes, path_nodes[1:]), start=1):
            prev_id, prev_type, prev_name = _node_identity(prev_node)
            next_id, next_type, next_name = _node_identity(next_node)

            if prev_id not in nodes or nodes[prev_id].hop > hop - 1:
                nodes[prev_id] = NetworkNode(id=prev_id, type=prev_type, name=prev_name, hop=hop - 1)
            if next_id not in nodes or nodes[next_id].hop > hop:
                nodes[next_id] = NetworkNode(id=next_id, type=next_type, name=next_name, hop=hop)

            edge_key = (prev_id, next_id)
            if edge_key not in edges or edges[edge_key].hop > hop:
                edges[edge_key] = NetworkEdge(from_=prev_id, to=next_id, relation="BELONGS_TO", hop=hop)

    kept_node_ids = {
        node_id for node_id, node in nodes.items() if node_id == company_id or node.type in filter_types
    }
    final_nodes = [node for node_id, node in nodes.items() if node_id in kept_node_ids]
    final_edges = [
        edge for edge in edges.values() if edge.from_ in kept_node_ids and edge.to in kept_node_ids
    ]

    return CompanyNetworkResponse(sourceCompanyId=company_id, nodes=final_nodes, edges=final_edges)


async def get_theme_companies(theme_id: int, top: int) -> ThemeCompaniesResponse | None:
    """themeId에 속한 기업을 시가총액(mrkTotAmt) 내림차순으로 top개 반환한다."""
    async with neo4j_database.get_session() as session:
        theme_result = await session.run(
            "MATCH (t:Theme {theme_id: $themeId}) RETURN t.name AS name",
            themeId=theme_id,
        )
        theme_record = await theme_result.single()
        if theme_record is None:
            return None

        companies_result = await session.run(
            """
            MATCH (:Theme {theme_id: $themeId})<-[:BELONGS_TO]-(c:Company)
            RETURN c.srtnCd AS id, c.name AS name, c.market AS market, c.mrkTotAmt AS mrkTotAmt
            ORDER BY c.mrkTotAmt DESC
            LIMIT $top
            """,
            themeId=theme_id,
            top=top,
        )
        companies = [
            ThemeCompanyItem(
                id=record["id"],
                name=record["name"],
                market=record["market"],
                mrkTotAmt=record["mrkTotAmt"],
            )
            async for record in companies_result
        ]

    return ThemeCompaniesResponse(themeId=theme_id, themeName=theme_record["name"], companies=companies)
