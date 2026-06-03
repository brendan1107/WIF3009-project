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
import pandas as pd
import numpy as np


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
        champ_cnt_path = os.path.join(BASE_DIR, "data", "Parquets", "champ_counters.parquet")
        final_draft_path = os.path.join(BASE_DIR, "data", "Parquets", "final_draft.parquet")
        
        app.state.champ_wr = pl.read_parquet(champ_wr_path)
        app.state.champ_synergies = pl.read_parquet(champ_syn_path)
        app.state.champ_counters = pl.read_parquet(champ_cnt_path)

        # Pre-build lookup maps for optimal route performance
        app.state.wr_map = dict(zip(app.state.champ_wr["champion"].to_list(), app.state.champ_wr["win_rate"].to_list()))
        app.state.global_avg_wr = app.state.champ_wr["win_rate"].mean()

        app.state.pair_map = {}
        for row in app.state.champ_synergies.to_dicts():
            key = tuple(sorted([row["champion"], row["champ2"]]))
            app.state.pair_map[key] = row["pair_win_rate"]

        app.state.counter_map = {}
        for row in app.state.champ_counters.to_dicts():
            key = (row["champion"], row["enemy_champion"], row["role"])
            app.state.counter_map[key] = {
                "winrate": float(row["head_to_head_winrate"]),
                "matches": int(row["matches"])
            }

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
        
        # Pre-calculate OP champions from shap_output
        print("Pre-calculating global OP champions from SHAP values...")
        shap_values = app.state.shap_output["values"]
        shap_data = app.state.shap_output["data"]
        features = app.state.shap_output["features"]
        
        shap_df = pd.DataFrame(shap_values, columns=features)
        data_df = pd.DataFrame(shap_data, columns=features)
        
        role_cols = [
            "blue_top", "blue_jng", "blue_mid", "blue_bot", "blue_sup",
            "red_top", "red_jng", "red_mid", "red_bot", "red_sup"
        ]
        
        champ_contributions = {}
        for i in range(len(data_df)):
            for col in role_cols:
                enc_col = col + "_enc"
                le = app.state.encoders[col]
                idx = int(data_df.loc[i, enc_col])
                champ_name = le.classes_[idx]
                shap_val = float(shap_df.loc[i, enc_col])
                
                contrib = shap_val if col.startswith("blue") else -shap_val
                
                if champ_name not in champ_contributions:
                    champ_contributions[champ_name] = []
                champ_contributions[champ_name].append(contrib)
                
        op_list = []
        for name, contribs in champ_contributions.items():
            op_list.append({
                "champion": name,
                "mean_contribution": float(np.mean(contribs)),
                "mean_abs_contribution": float(np.mean(np.abs(contribs))),
                "count": len(contribs)
            })
            
        op_list = sorted(op_list, key=lambda x: x["mean_contribution"], reverse=True)
        app.state.op_champions = op_list
        print(f"Pre-calculated {len(op_list)} global OP champions successfully.")

        # --- Neutral inference defaults (used to stabilise mid-draft predictions) ---
        # 1. Latest patch in correct encoder format (avoids silent fallback to "16.01")
        app.state.latest_patch = app.state.encoders["patch"].classes_[-1]
        print(f"Latest patch resolved to: {app.state.latest_patch}")

        # 2. Modal (most common) league from training data — avoids hardcoding LCK
        try:
            draft_df = pl.read_parquet(final_draft_path)
            modal_league = draft_df["league"].value_counts().sort("count", descending=True)["league"][0]
            if modal_league in app.state.encoders["league"].classes_:
                app.state.neutral_league = modal_league
            else:
                app.state.neutral_league = app.state.encoders["league"].classes_[len(app.state.encoders["league"].classes_) // 2]
        except Exception as e:
            print(f"Warning: could not compute modal league ({e}), using median class.")
            app.state.neutral_league = app.state.encoders["league"].classes_[len(app.state.encoders["league"].classes_) // 2]
        print(f"Neutral league resolved to: {app.state.neutral_league}")

        # 3. Median encoding index per role column — used as a neutral placeholder for empty slots
        #    instead of classes_[0] which injects a specific phantom champion (e.g. Aatrox)
        champion_role_cols = [
            "blue_top", "blue_jng", "blue_mid", "blue_bot", "blue_sup",
            "red_top",  "red_jng",  "red_mid",  "red_bot",  "red_sup",
        ]
        app.state.median_enc = {
            col: len(app.state.encoders[col].classes_) // 2
            for col in champion_role_cols
        }
        print("Neutral median encodings pre-computed for empty draft slots.")
        # --- End neutral defaults ---

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
