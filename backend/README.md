# Rift Draft: Backend Service

This directory houses the FastAPI service, predictive models, explainability structures, historical datasets, and LangGraph agent pipelines that power the **Rift Draft** application. For full-stack deployment, env templates, and frontend setups, please refer to the primary [Root README](../README.md).

---

## Tech Stack

- **Framework**: FastAPI (Asynchronous lifespan handlers)
- **Data Loaders**: Polars (Multi-threaded Parquet reader)
- **ML Classifier**: LightGBM (Calibrated via Isotonic Regression)
- **Interpretability**: SHAP (TreeExplainer integration)
- **Agent Framework**: LangGraph (Single-turn analytical graph)
- **Cognitive Model**: Google Gemini (`google-genai` model `gemini-3.1-flash-lite`)

---

## Directory Structure

- `app/`
  - `agents/`: Contains Gemini and LangGraph client logic.
  - `api/`: Houses routing controllers and schemas.
  - `core/`: Config settings and credentials setup.
  - `tools/`: Extensible agentic tools.
  - `tests/`: End-to-end endpoint tests (`.http`).
- `data/`: Holds the active v2 model bundle in `data/models/v2/`, plus historical Parquets used for matchup counters.
- `data/models/v2/`: Active LightGBM calibrated model (`calibrated_model_v2.pkl`), encoders, feature list, SHAP payload, role/global win-rate maps, global average, and role-qualified synergy pair map.
- `scripts/`: Data pre-processing, matching, and feature extraction scripts.

---

## Quick Start (Development)

### 1. Configure Environments
Create a `.env` file in this directory:

```ini
PROJECT_NAME="Esports Predictor Backend"
DEBUG_MODE=true
GEMINI_API_KEY="your_gemini_api_key"
GEMINI_MODEL="gemini-3.1-flash-lite"
MAX_RETRY_COUNT=3
USE_MOCK_TOOLS=false
```

### 2. Install and Start

```bash
# Set up virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate    # Mac/Linux

# Install requirements
pip install -r requirements.txt

# Launch dev server
fastapi dev app/main.py
```

---

## API Endpoints

- **`POST /api/v1/predict`**: Evaluates the draft state (10 pick slots, 10 ban slots) and returns the win rate projection percentage for both blue and red teams.
- **`POST /api/v1/agent`**: A composite endpoint analyzing deep draft features, pre-calculating optimized recommendations, and passing context to the LangGraph/Gemini agent to generate tactical warnings, matchup breakdowns, and role suggestions.
- **`POST /api/v1/predict-options`**: Batch-scores candidate champion-role options using the same v2 feature builder as `/predict`.
- **`GET /api/v1/champions/metrics`**: Exposes v2 global champion win rates, role-specific win rates, role-qualified synergy maps, and the active model version.
- **`GET /api/v1/champions/op`**: Returns global and role-specific strength rankings derived from the v2 win-rate maps.

## Active Model Contract

The active v2 model uses `feature_cols_v2.pkl` as the source of truth. Core generated features include `patch_enc`, `league_enc`, the ten role win-rate features, `blue_synergy`, `red_synergy`, `blue_team_avg_wr`, `red_team_avg_wr`, `wr_diff`, `synergy_diff`, and `picks_filled`. If the model artifact expects encoded champion-role columns such as `blue_top_enc` or `red_bot_enc`, the backend now generates those too.

Champion inputs are transformed into role-qualified keys such as `Caitlyn_bot` and `Lee Sin_jng`. Role win rates fall back to global champion win rate, then the serialized global average. Missing draft slots use the global average, encode as `UNKNOWN_{role}`, and are counted through `picks_filled`. Patch and league are generated internally and default to `UNKNOWN`.

Predicted probabilities are post-processed before returning to clients: raw model output is weighted back toward 50/50 using `picks_filled / 10`, then clamped to 15-85%.
