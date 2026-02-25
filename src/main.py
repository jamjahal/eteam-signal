from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.core.config import settings
from src.core.logger import configure_logger
from src.api.routes import router
from src.api.insider_routes import insider_router, set_store
from src.services.insider_store import InsiderStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = InsiderStore()
    try:
        await store.connect()
        set_store(store)
    except Exception:
        pass  # graceful degradation when DB is unavailable
    yield
    try:
        await store.close()
    except Exception:
        pass


def create_app() -> FastAPI:
    configure_logger()
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="SEC Alpha-Sentinel API",
        lifespan=lifespan,
    )

    app.include_router(router, prefix="/api/v1")
    app.include_router(insider_router, prefix="/api/v1")

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()
