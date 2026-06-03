from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse

from app.core.config import settings


import os
import pickle
import polars as pl
import shap


def get_calibrated_estimator(calibrated_model):
    """Return the wrapped estimator across sklearn CalibratedClassifierCV versions."""
    calibrated_classifiers = getattr(calibrated_model, "calibrated_classifiers_", None)
    if calibrated_classifiers:
        classifier = calibrated_classifiers[0]
        estimator = getattr(classifier, "estimator", None)
        if estimator is not None:
            return estimator
        estimator = getattr(classifier, "base_estimator", None)
        if estimator is not None:
            return estimator
    return getattr(calibrated_model, "estimator", calibrated_model)


def build_role_strengths(champ_role_wr_map, champ_wr_global_map, global_avg_wr):
    role_strengths = []
    champion_roles = {}

    for key, win_rate in champ_role_wr_map.items():
        if "_" not in key:
            continue
        champion, role_part = key.rsplit("_", 1)
        item = {
            "champion": champion,
            "role": role_part,
            "win_rate": float(win_rate),
            "mean_contribution": float(win_rate) - float(global_avg_wr),
            "source": "v2_role_win_rate",
        }
        role_strengths.append(item)
        champion_roles.setdefault(champion, []).append(item)

    op_champions = []
    for champion, global_wr in champ_wr_global_map.items():
        roles = champion_roles.get(champion, [])
        best_role = max(roles, key=lambda item: item["win_rate"]) if roles else None
        op_champions.append(
            {
                "champion": champion,
                "role": best_role["role"] if best_role else None,
                "win_rate": float(global_wr),
                "mean_contribution": float(global_wr) - float(global_avg_wr),
                "mean_abs_contribution": abs(float(global_wr) - float(global_avg_wr)),
                "count": len(roles),
                "source": "v2_global_win_rate",
            }
        )

    return (
        sorted(role_strengths, key=lambda item: item["win_rate"], reverse=True),
        sorted(op_champions, key=lambda item: item["mean_contribution"], reverse=True),
    )


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
        
        # Load parquet metadata that is still independent of the v2 model.
        print("Loading matchup counter parquet...")
        champ_cnt_path = os.path.join(BASE_DIR, "data", "Parquets", "champ_counters.parquet")
        app.state.champ_counters = pl.read_parquet(champ_cnt_path)

        app.state.counter_map = {}
        for row in app.state.champ_counters.to_dicts():
            key = (row["champion"], row["enemy_champion"], row["role"])
            app.state.counter_map[key] = {
                "winrate": float(row["head_to_head_winrate"]),
                "matches": int(row["matches"])
            }

        # Load v2 pickle model and lookup maps.
        print("Loading v2 model assets...")
        model_dir = os.path.join(BASE_DIR, "data", "models", "v2")
        model_path = os.path.join(model_dir, "calibrated_model_v2.pkl")
        encoders_path = os.path.join(model_dir, "encoders_v2.pkl")
        feature_cols_path = os.path.join(model_dir, "feature_cols_v2.pkl")
        shap_output_path = os.path.join(model_dir, "shap_output_v2.pkl")
        champ_role_wr_path = os.path.join(model_dir, "champ_role_wr_map.pkl")
        champ_global_wr_path = os.path.join(model_dir, "champ_wr_global_map.pkl")
        global_avg_wr_path = os.path.join(model_dir, "global_avg_wr.pkl")
        pair_map_path = os.path.join(model_dir, "pair_map.pkl")

        with open(model_path, 'rb') as f:
            app.state.calibrated_model = pickle.load(f)
        with open(encoders_path, 'rb') as f:
            app.state.encoders = pickle.load(f)
        with open(feature_cols_path, 'rb') as f:
            app.state.feature_cols = pickle.load(f)
        with open(shap_output_path, 'rb') as f:
            app.state.shap_output = pickle.load(f)
        with open(champ_role_wr_path, 'rb') as f:
            app.state.champ_role_wr_map = pickle.load(f)
        with open(champ_global_wr_path, 'rb') as f:
            app.state.wr_map = pickle.load(f)
        with open(global_avg_wr_path, 'rb') as f:
            app.state.global_avg_wr = float(pickle.load(f))
        with open(pair_map_path, 'rb') as f:
            app.state.pair_map = pickle.load(f)

        app.state.model_version = "v2"

        # Pre-initialize SHAP TreeExplainer once to conserve route memory
        lgbm_model = get_calibrated_estimator(app.state.calibrated_model)
        app.state.shap_explainer = shap.TreeExplainer(lgbm_model)

        app.state.role_strengths, app.state.op_champions = build_role_strengths(
            app.state.champ_role_wr_map,
            app.state.wr_map,
            app.state.global_avg_wr,
        )
        print(f"Loaded v2 model with {len(app.state.feature_cols)} features.")
        print("All v2 model assets and matchup metadata cached in application state.")
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
from app.api.routes import router as api_router, root_router

# Mount your agent router under a versioned API prefix
app.include_router(api_router, prefix="/api/v1", tags=["Agent Blueprint"])
# Mount root level router for tool compatibility
app.include_router(root_router)


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
