from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1.auth import router as auth_router
from app.api.v1.schemes import router as schemes_router
from app.api.v1.analytics import router as analytics_router
from app.api.v1.predictions import router as predictions_router
from app.api.v1.watchlist import router as watchlist_router
from app.api.v1.portfolio import router as portfolio_router
from app.api.v1.recommendations import router as recommendations_router
from app.ml.model_loader import model_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mutual_fund_ai")

# Auto-create tables if running against SQLite or local instance
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    description="AI Mutual Fund Analytics & Prediction Platform Backend API"
)

# CORS configuration
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
def startup_event():
    logger.info("Initializing ML Model Registry...")
    model_registry.load_models()

@app.get("/health")
def health_check():
    return {
        "name": settings.PROJECT_NAME,
        "status": "healthy",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR
    }

from fastapi.responses import RedirectResponse

@app.get("/")
def read_root():
    return RedirectResponse(url="http://localhost:8080/")


import os
from fastapi.staticfiles import StaticFiles

# Include V1 API Routers
app.include_router(auth_router, prefix=f"{settings.API_V1_STR}/auth", tags=["Authentication"])
app.include_router(schemes_router, prefix=f"{settings.API_V1_STR}/schemes", tags=["Schemes"])
app.include_router(analytics_router, prefix=f"{settings.API_V1_STR}/analytics", tags=["Analytics"])
app.include_router(predictions_router, prefix=f"{settings.API_V1_STR}/predictions", tags=["Predictions"])
app.include_router(watchlist_router, prefix=f"{settings.API_V1_STR}/watchlist", tags=["Watchlist"])
app.include_router(portfolio_router, prefix=f"{settings.API_V1_STR}/portfolio", tags=["Portfolio"])
app.include_router(recommendations_router, prefix=f"{settings.API_V1_STR}/recommendations", tags=["Recommendations"])


# Mount Web Application Static Directory
base_backend = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fundiq_root = os.path.dirname(base_backend)

web_dirs = [
    os.path.join(fundiq_root, "frontend", "dist"),
    os.path.join(fundiq_root, "frontend", ".output", "public"),
    os.path.join(fundiq_root, "frontend", "public"),
    os.path.join(fundiq_root, "frontend"),
    os.path.join(base_backend, "..", "frontend")
]

target_web_dir = next((d for d in web_dirs if os.path.exists(d)), None)
if target_web_dir:
    app.mount("/", StaticFiles(directory=target_web_dir, html=True), name="web")


