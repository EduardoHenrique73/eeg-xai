"""Aplicação FastAPI — plataforma EEG-XAI (TCC 2)."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes.auth import router as auth_router
from app.api.routes.exames import router as exames_router
from app.api.routes.pacientes import router as pacientes_router
from app.api.routes.stats import router as stats_router
from app.config import get_settings
from app.database import engine
from app.startup import criar_tabelas, semear_dados_desenvolvimento

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida da aplicação (startup/shutdown)."""
    settings.edf_storage_path.mkdir(parents=True, exist_ok=True)
    settings.shap_storage_path.mkdir(parents=True, exist_ok=True)

    if settings.is_development:
        await criar_tabelas(engine)
        await semear_dados_desenvolvimento()

    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "Plataforma web de diagnóstico neurológico com IA explicável (XAI). "
        "Evolução do sistema legado PIBITI/EEG."
    ),
    version=__version__,
    debug=settings.debug,
    lifespan=lifespan,
)

if settings.is_development:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:5174",
            "http://127.0.0.1:5174",
            "http://localhost:5175",
            "http://127.0.0.1:5175",
            "http://localhost:5176",
            "http://127.0.0.1:5176",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(auth_router)
app.include_router(exames_router)
app.include_router(pacientes_router)
app.include_router(stats_router)
app.mount(
    "/static",
    StaticFiles(directory=str(settings.storage_root)),
    name="static",
)


@app.get("/health", tags=["Sistema"])
async def health_check() -> dict[str, str]:
    """Endpoint de saúde para monitoramento e deploy."""
    return {"status": "ok", "service": "eeg-xai", "environment": settings.app_env}


@app.get("/", tags=["Sistema"])
async def root() -> dict[str, str]:
    """Informações básicas da API."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
