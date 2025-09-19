from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import SETTINGS
from .api import router as api_router
from .health import router as health_router

app = FastAPI(title="OpenAI-Compatible API (PyTorch + FastAPI)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in SETTINGS.allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(api_router)