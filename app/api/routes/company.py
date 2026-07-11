from fastapi import APIRouter, HTTPException, Query

from app.crud import get_company_network
from app.schemas import CompanyNetworkResponse

router = APIRouter(prefix="/api/v1", tags=["company"])

_VALID_FILTER_TYPES = {"COMPANY", "THEME"}


@router.get("/companies/{company_id}/network", response_model=CompanyNetworkResponse)
async def get_network(
    company_id: str,
    hops: int = Query(1, ge=1, le=3, description="탐색할 깊이"),
    filter: str | None = Query(None, description="포함할 노드 타입, 콤마로 구분 (예: THEME,COMPANY)"),
):
    if filter:
        filter_types = {value.strip().upper() for value in filter.split(",")}
        invalid = filter_types - _VALID_FILTER_TYPES
        if invalid:
            raise HTTPException(status_code=422, detail=f"Invalid filter value(s): {sorted(invalid)}")
    else:
        filter_types = _VALID_FILTER_TYPES

    network = await get_company_network(company_id, hops, filter_types)
    if network is None:
        raise HTTPException(status_code=404, detail=f"Company {company_id} not found")
    return network
