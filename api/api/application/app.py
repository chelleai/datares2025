from fastapi import FastAPI

from api.application.assets.routes import router as assets_router
from api.application.guides.routes import router as guides_router

app = FastAPI()

app.include_router(assets_router)
app.include_router(guides_router)
