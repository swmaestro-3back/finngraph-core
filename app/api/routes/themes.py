from fastapi import APIRouter, HTTPException, Query

from app.crud import get_theme_companies
from app.schemas import ThemeCompaniesResponse

router = APIRouter(prefix="/api/v1", tags=["themes"])


@router.get("/themes/{theme_id}/companies", response_model=ThemeCompaniesResponse)
async def get_companies(
    theme_id: int,
    top: int = Query(5, ge=1, le=20, description="조회할 기업 수"),
):
    result = await get_theme_companies(theme_id, top)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Theme {theme_id} not found")
    return result
