from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.api.routes import router as api_router
from app.core.config import settings


import os
import pickle
import polars as pl
import shap

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    Pre-loads heavy serialized classifiers and parquets into memory.
    """
    # [STARTUP] Code here runs BEFORE the server starts accepting requests
    print(f"Starting up {settings.PROJECT_NAME}...")
    
    try:
        # Get base directory of the backend package (parent folder of 'app')
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Load parquets
        print("Loading parquets...")
        champ_wr_path = os.path.join(BASE_DIR, "data", "Parquets", "champ_winrates.parquet")
        champ_syn_path = os.path.join(BASE_DIR, "data", "Parquets", "champ_synergies.parquet")
        
        app.state.champ_wr = pl.read_parquet(champ_wr_path)
        app.state.champ_synergies = pl.read_parquet(champ_syn_path)

        # Pre-build lookup maps for optimal route performance
        app.state.wr_map = dict(zip(app.state.champ_wr["champion"].to_list(), app.state.champ_wr["win_rate"].to_list()))
        app.state.global_avg_wr = app.state.champ_wr["win_rate"].mean()

        app.state.pair_map = {}
        for row in app.state.champ_synergies.to_dicts():
            key = tuple(sorted([row["champion"], row["champ2"]]))
            app.state.pair_map[key] = row["pair_win_rate"]

        # Load pickle models
        print("Loading pickles...")
        model_path = os.path.join(BASE_DIR, "data", "calibrated_model.pkl")
        encoders_path = os.path.join(BASE_DIR, "data", "encoders.pkl")
        feature_cols_path = os.path.join(BASE_DIR, "data", "feature_cols.pkl")
        shap_output_path = os.path.join(BASE_DIR, "data", "shap_output.pkl")

        with open(model_path, 'rb') as f:
            app.state.calibrated_model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            app.state.encoders = pickle.load(f)
        with open(feature_cols_path, 'rb') as f:
            app.state.feature_cols = pickle.load(f)
        with open(shap_output_path, 'rb') as f:
            app.state.shap_output = pickle.load(f)

        # Pre-initialize SHAP TreeExplainer once to conserve route memory
        lgbm_model = app.state.calibrated_model.calibrated_classifiers_[0].estimator
        app.state.shap_explainer = shap.TreeExplainer(lgbm_model)
        
        print("All models, encoders, and parquets successfully cached in application state.")
    except Exception as exc:
        print(f"CRITICAL: Failed to load serialized assets during lifespan startup: {exc}")
        raise exc

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
