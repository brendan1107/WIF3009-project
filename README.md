# Rift Draft: Esports Draft Assistant & Win-Rate Predictor

Rift Draft is a dual-layer decision support platform designed for League of Legends competitive drafts. It pairs a calibrated LightGBM machine learning classifier with a LangGraph-orchestrated Gemini cognitive agent, converting complex telemetry and SHAP (SHapley Additive exPlanations) values into real-time tactical draft coaching.

---

## Architecture & System Design

The system uses a decoupled frontend-backend architecture designed for sub-second latency and precise predictions during active drafting.

### Backend Pipeline (FastAPI)
The FastAPI service handles heavy numerical computation and houses the predictive pipeline:
- **ML Engine**: A LightGBM classifier calibrated via Isotonic Regression, scoring current draft states and simulating alternative pick/ban branches.
- **Explainability (SHAP)**: Pre-loads TreeExplainer configurations, calculating mathematical contributions for each active choice on the fly.
- **In-Memory Caches**: Leverages Polars to load three high-throughput parquet files (`champ_winrates.parquet`, `champ_synergies.parquet`, `champ_counters.parquet`) on startup, creating instant lookup maps.
- **Agentic Layer (LangGraph)**: An agentic workflow (`langgraph_agent.py`) translating numeric SHAP impact vectors and matchup tables into structured tactical coaching advice via Google Gemini API.

```
                          [ REACT FRONTEND (Rift Draft) ]
                                  /             \
                       (Inference Payload)    (Agent Prompt Payload)
                                /                 \
  [ FastAPI Predict Route ] <-----\             /-----> [ FastAPI Agent Route ]
                |                  \           /                    |
   [ LightGBM Model & SHAP ]        \         /            [ LangGraph Analyst Node ]
                |                    \       /                      |
   (Real-time Win Probabilities)     [ CACHES ]              [ Google Gemini Model ]
   (Feature Contribution SHAPs)     /   |     \                     |
                                   /    |      \           (Tactical Draft Warnings)
                     [Winrate Parquet]  |  [Counters Parquet] (Role Recommendations)
                                 [Synergy Parquet]         (Matchup Counter Analysis)
```

---

## Features

1. **Interactive Draft Board**: Simulates the official competitive draft format (Ban Phase 1, Pick Phase 1, Ban Phase 2, Pick Phase 2) with support for Blue and Red active turns.
2. **Real-Time Win Rate Prediction**: Instant LightGBM predictions updated after every draft action.
3. **Advanced Telemetry (SHAP Drivers)**: Renders the top 5 game drivers showing exactly which champion choices or matchups are contributing most heavily to or against the projected win chance.
4. **Lane-Specific Recommendation Engines**: Generates highly targeted ban suggestions or role-specific pick suggestions (TOP, JUNGLE, MID, BOTTOM, SUPPORT).
5. **LangGraph Agentic Co-pilot**: Employs Gemini to generate three critical tactical outputs:
   - **Tactical composition warning**: Detects team vulnerabilities from SHAP impacts.
   - **Role recommendations**: Suggests pick and ban options with clear, metric-driven rationale.
   - **Matchup counter analysis**: Pinpoints critical head-to-head lane counters (deviation from 50% head-to-head metrics).
6. **Smart Filtering & Sorting**: Supports alphabetical search, role filter tabs, and sorting by win rate, synergy index, matchup counter score, and name.
7. **Local Execution Fallback**: Automatically switches to a local mathematical heuristic engine if the backend is offline.

---

## Tech Stack

- **Frontend**: React 19, TypeScript, Vite, CSS Grid & Flexbox, HTML5 Semantic Elements.
- **Backend**: FastAPI, Python 3.11+, LangGraph, Google GenAI SDK, Polars, LightGBM, SHAP, Pandas, Numpy, Scikit-learn.
- **Data Stores**: Apache Parquet files for high-throughput memory mapping.

---

## Setup & Installation

### Prerequisites
- **Node.js** (v18.0.0 or higher)
- **Python** (v3.11.0 or higher)
- **Gemini API Key** (Obtained from Google AI Studio)

---

### 1. Backend Setup
Navigate to the backend directory and set up your Python environment.

```bash
# Navigate to backend folder
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### Environment Configuration
Create a `.env` file in the root of the `backend` folder based on `.env.example`:

```ini
PROJECT_NAME="Esports Predictor Backend"
DEBUG_MODE=true

# Google Gemini Configuration
GEMINI_API_KEY="your_actual_gemini_api_key_here"
GEMINI_MODEL="gemini-3.1-flash-lite"
MAX_RETRY_COUNT=3

# Development Toggle
# Set to false to use live ML/Agent inference.
# Set to true to bypass model loading and use mocks.
USE_MOCK_TOOLS=false
```

---

### 2. Frontend Setup
Navigate to the frontend directory and install the Node packages.

```bash
# Navigate to frontend folder
cd ../frontend

# Install dependencies
npm install
```

#### Environment Configuration
Create a `.env` file in the root of the `frontend` folder:

```ini
VITE_WINRATE_API_URL="http://localhost:8000/api/v1/predict"
VITE_COACHING_API_URL="http://localhost:8000/api/v1/agent"
```

## Running the Application

For a fully connected experience, you must run both the backend and frontend servers simultaneously.

### Starting the Backend Server
Make sure your virtual environment is active in the `backend` directory.

```bash
# Navigate to backend (if not already there)
cd backend

# Option A: Run via Makefile (if make is installed)
make dev

# Option B: Run via FastAPI CLI
fastapi dev app/main.py

# Option C: Run directly using Uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
The backend server will run on `http://127.0.0.1:8000`. You can view the interactive FastAPI documentation at `http://127.0.0.1:8000/docs` (when `DEBUG_MODE=true` in `.env`).

### Starting the Frontend Server
Open a new terminal window, navigate to the `frontend` directory, and launch Vite.

```bash
# Navigate to frontend
cd frontend

# Start dev server
npm run dev
```
The frontend server will boot and display the local network URL (typically `http://localhost:5173`). Open this URL in your web browser.


## Testing the API

You can test the FastAPI endpoints using the HTTP client file provided in the repository:
- Run the backend server.
- Open `backend/app/tests/agent.http` in VS Code (with REST Client extension installed) or use `curl` to send POST requests directly to `http://localhost:8000/api/v1/agent`.

