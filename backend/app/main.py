from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import ModelArtifacts, predict_draft, load_artifacts


Role = Literal["TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT"]
Team = Literal["blue", "red"]
DraftAction = Literal["pick", "ban"]


class Slot(BaseModel):
    role: Role
    champion: Optional[str] = None


class DraftPayload(BaseModel):
    bluePicks: List[Slot]
    redPicks: List[Slot]
    blueBans: List[str] = Field(default_factory=list)
    redBans: List[str] = Field(default_factory=list)
    activeTeam: Team
    activeAction: DraftAction
    activeRole: Optional[Role] = None
    stepIndex: int = 0


class PredictionResponse(BaseModel):
    blueWinRate: float
    redWinRate: float
    picksFilled: int
    confidence: str


class CoachResponse(BaseModel):
    draft_warning: str = ""
    recommendations: str = ""
    counter_analysis: str = ""
    top_drivers: List[Dict[str, float | str]] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    winrates: Dict[str, float] = Field(default_factory=dict)
    synergies: Dict[str, float] = Field(default_factory=dict)
    global_avg_wr: float = 0.5


class OpChampion(BaseModel):
    champion: str
    mean_contribution: float
    mean_abs_contribution: float
    count: int


class OpResponse(BaseModel):
    op_champions: List[OpChampion] = Field(default_factory=list)


ROLE_MAP = {
    "TOP": "top",
    "JUNGLE": "jng",
    "MID": "mid",
    "BOTTOM": "bot",
    "SUPPORT": "sup",
}


app = FastAPI(title="Rift Draft Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
def load_model() -> None:
    model_dir = os.getenv("MODEL_DIR")
    base_path = Path(model_dir) if model_dir else Path(__file__).resolve().parents[1] / "data"
    try:
        app.state.artifacts = load_artifacts(base_path)
    except FileNotFoundError:
        app.state.artifacts = None


def artifacts_or_503() -> ModelArtifacts:
    artifacts = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Verify files in backend/data.",
        )
    return artifacts


@app.get("/api/v1/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(payload: DraftPayload) -> PredictionResponse:
    artifacts = artifacts_or_503()

    picks: Dict[str, Optional[str]] = {col: None for col in (
        "blue_top",
        "blue_jng",
        "blue_mid",
        "blue_bot",
        "blue_sup",
        "red_top",
        "red_jng",
        "red_mid",
        "red_bot",
        "red_sup",
    )}

    for slot in payload.bluePicks:
        role_key = ROLE_MAP[slot.role]
        picks[f"blue_{role_key}"] = slot.champion
    for slot in payload.redPicks:
        role_key = ROLE_MAP[slot.role]
        picks[f"red_{role_key}"] = slot.champion

    result = predict_draft(picks, artifacts)
    blue_rate = round(result["blue_win_prob"] * 100, 1)
    red_rate = round(100 - blue_rate, 1)

    return PredictionResponse(
        blueWinRate=blue_rate,
        redWinRate=red_rate,
        picksFilled=int(result["picks_filled"]),
        confidence=str(result["confidence"]),
    )


@app.post("/api/v1/agent", response_model=CoachResponse)
def coach() -> CoachResponse:
    return CoachResponse()


@app.get("/api/v1/champions/metrics", response_model=MetricsResponse)
def champion_metrics() -> MetricsResponse:
    artifacts = getattr(app.state, "artifacts", None)
    if artifacts is None:
        return MetricsResponse()

    return MetricsResponse(
        winrates=artifacts.champ_wr_global_map,
        global_avg_wr=artifacts.global_avg_wr,
    )


@app.get("/api/v1/champions/op", response_model=OpResponse)
def op_champions() -> OpResponse:
    return OpResponse()
