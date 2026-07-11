# Request Response DTO
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal["COMPANY", "THEME"]


class NetworkNode(BaseModel):
    id: str
    type: NodeType
    name: str
    hop: int


class NetworkEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    relation: str
    hop: int


class CompanyNetworkResponse(BaseModel):
    sourceCompanyId: str
    nodes: list[NetworkNode]
    edges: list[NetworkEdge]


class ThemeCompanyItem(BaseModel):
    id: str
    name: str
    market: Literal["KOSPI", "KOSDAQ"]
    mrkTotAmt: int


class ThemeCompaniesResponse(BaseModel):
    themeId: int
    themeName: str
    companies: list[ThemeCompanyItem]
