from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import SETTINGS
from .api import router as api_router
from .health import router as health_router

app = FastAPI(
    title="Fidel-AI Inference API",
    version="1.0.0",
    description="OpenAI-compatible chat completion API backed by a local model.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in SETTINGS.allow_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, tags=["inference"])
app.include_router(health_router, tags=["health"])