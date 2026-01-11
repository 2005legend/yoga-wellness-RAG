"""
Main application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from backend.config import settings
from .routes import router
from backend.core.logging import configure_logging, get_logger

# Setup logging
configure_logging()
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    # Shutdown - cleanup resources
    logger.info("Shutting down application")
    # Close any embedding service sessions
    try:
        from backend.api.dependencies import get_embedding_service
        service = get_embedding_service()
        if hasattr(service, '_service') and hasattr(service._service, 'close'):
            await service._service.close()
    except Exception as e:
        logger.debug(f"Error closing embedding service: {e}")

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Mount static files
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    import os
    
    # Ensure frontend directory exists (it should)
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
    if os.path.exists(frontend_dir):
        # Mount static files at root level so CSS/JS can be accessed directly
        app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
        
        # Serve CSS and JS files directly at root level
        @app.get("/style.css")
        async def get_style():
            css_path = os.path.join(frontend_dir, "style.css")
            if os.path.exists(css_path):
                return FileResponse(css_path, media_type="text/css")
            return {"error": "CSS file not found"}, 404
        
        @app.get("/app.js")
        async def get_app_js():
            js_path = os.path.join(frontend_dir, "app.js")
            if os.path.exists(js_path):
                return FileResponse(js_path, media_type="application/javascript")
            return {"error": "JS file not found"}, 404
        
        @app.get("/")
        async def read_root():
            index_path = os.path.join(frontend_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "Frontend not found"}
    else:
        logger.warning(f"Frontend directory not found at {frontend_dir}")
        
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app", 
        host=settings.api_host, 
        port=settings.api_port, 
        reload=settings.api_reload
    )

