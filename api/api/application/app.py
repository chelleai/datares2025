from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.application.assets.routes import router as assets_router
from api.application.guides.routes import router as guides_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # All HTTP methods
    allow_headers=["*"],  # All HTTP headers
)

app.include_router(assets_router)
app.include_router(guides_router)
