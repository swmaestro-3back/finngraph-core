from fastapi import APIRouter, FastAPI

from app.api.routes import company, themes

api_router = APIRouter()
api_router.include_router(company.router)
api_router.include_router(themes.router)

app = FastAPI()
app.include_router(api_router)
