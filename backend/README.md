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
- `data/`: Holds trained LightGBM models (`calibrated_model.pkl`), label encoders (`encoders.pkl`), feature configurations, and historical Parquets (`champ_winrates.parquet`, `champ_synergies.parquet`, `champ_counters.parquet`).
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
- **`GET /api/v1/champions/metrics`**: Exposes pre-cached champion win rates and synergy lookup matrix maps.
- **`GET /api/v1/champions/op`**: Returns mathematical global high-contribution champions based on aggregate SHAP analysis.
