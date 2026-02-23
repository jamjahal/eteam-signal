from fastapi import FastAPI
from src.core.config import settings
from src.core.logger import configure_logger
from src.api.routes import router

def create_app() -> FastAPI:
    configure_logger()
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="SEC Alpha-Sentinel API"
    )
    
    app.include_router(router, prefix="/api/v1")
    
    @app.get("/health")
    async def health_check():
        return {"status": "ok"}
        
    return app

app = create_app()
