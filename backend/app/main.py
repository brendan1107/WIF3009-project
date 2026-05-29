from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.api.routes import router as api_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Industry practice for initializing database pools, HTTP clients,
    or loading heavy ML models into memory.
    """
    # [STARTUP] Code here runs BEFORE the server starts accepting requests
    print(f"Starting up {settings.PROJECT_NAME}...")

    yield  # The application runs while paused here

    # [SHUTDOWN] Code here runs WHEN the server is turning off
    print(f"Shutting down {settings.PROJECT_NAME}...")


# Initialize FastAPI with conditional documentation based on environment
app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="Production API wrapper for Gemini Autonomous Agent",
    lifespan=lifespan,
    # Hide Swagger and Redoc UIs in production for security
    docs_url="/docs" if settings.DEBUG_MODE else None,
    redoc_url="/redoc" if settings.DEBUG_MODE else None,
)

# ---------------------------------------------------------------------
# MIDDLEWARE
# ---------------------------------------------------------------------
# Configure CORS (Cross-Origin Resource Sharing) so your frontend can talk to it
# In production, replace ["*"] with specific origins like ["https://myfrontend.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# ROUTER INCLUSION
# ---------------------------------------------------------------------
# Mount your agent router under a versioned API prefix
app.include_router(api_router, prefix="/api/v1", tags=["Agent Blueprint"])


# ---------------------------------------------------------------------
# SYSTEM ENDPOINTS (Health Checks & Redirects)
# ---------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root_redirect():
    """Redirects root traffic to the interactive docs if in debug mode."""
    if settings.DEBUG_MODE:
        return RedirectResponse(url="/docs")
    return {"status": "healthy"}


@app.get("/health", tags=["System"])
async def health_check():
    """
    Standard health check endpoint.
    Used by Cloud Load Balancers, Kubernetes, or Docker to verify
    the container is alive and handling traffic.
    """
    return {
        "status": "healthy",
        "app_name": settings.PROJECT_NAME,
        "environment": "development" if settings.DEBUG_MODE else "production",
    }
