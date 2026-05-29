import numpy as np
import pandas as pd
import shap
from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional, List
from itertools import combinations
from app.agents import agent_app

router = APIRouter()


class SlotPayload(BaseModel):
    role: str
    player: str
    championId: Optional[str] = None


class DraftPayload(BaseModel):
    bluePicks: List[SlotPayload]
    redPicks: List[SlotPayload]
    blueBans: List[str]
    redBans: List[str]
    activeTeam: str
    activeAction: str
    activeRole: Optional[str] = None
    stepIndex: int


class TelemetryDriver(BaseModel):
    feature: str
    value: float
    impact_on_win_prob: float


class AgentRequest(BaseModel):
    # New direct telemetry input format
    expected_base_win_prob: Optional[float] = None
    top_drivers: Optional[List[TelemetryDriver]] = None

    # Legacy format (now structured)
    draft_state: Optional[DraftPayload] = None
    user_question: Optional[str] = None
    interaction_id: Optional[str] = None


@router.post("/agent")
def agent_endpoint(req: AgentRequest, request: Request) -> dict:
    # 1. Direct Telemetry Input Path
    if req.expected_base_win_prob is not None and req.top_drivers is not None:
        initial_state = {
            "expected_base_win_prob": req.expected_base_win_prob,
            "top_drivers": [driver.model_dump() for driver in req.top_drivers],
            "explanation": "",
        }
        final_state = agent_app.invoke(initial_state)
        return {"response": final_state["explanation"]}

    # 2. Legacy/Local Calibrated Model Path (Zero Load-in-Route Execution)
    if req.draft_state is not None:
        # Extract pre-loaded caches from app state
        wr_map = request.app.state.wr_map
        global_avg_wr = request.app.state.global_avg_wr
        pair_map = request.app.state.pair_map
        calibrated_model = request.app.state.calibrated_model
        encoders = request.app.state.encoders
        feature_cols = request.app.state.feature_cols
        shap_explainer = request.app.state.shap_explainer

        # Local helper for synergy
        def team_synergy_score(champs):
            scores = []
            valid = [c for c in champs if pd.notna(c) and c != ""]
            for c1, c2 in combinations(valid, 2):
                key = tuple(sorted([c1, c2]))
                if key in pair_map:
                    scores.append(pair_map[key])
            return np.mean(scores) if scores else 0.5

        # Format input picks into top, jng, mid, bot, sup slots
        role_cols = [
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
        ]

        # Map slot list to roles correctly (aligning with /predict!)
        blue_slots = {s.role: s.championId for s in req.draft_state.bluePicks if s.championId}
        red_slots = {s.role: s.championId for s in req.draft_state.redPicks if s.championId}

        row = {}
        for col in role_cols:
            role_key = col.split("_")[1].upper()  # 'top' -> 'TOP'
            if col.startswith("blue"):
                row[col] = blue_slots.get(role_key) or encoders[col].classes_[0]
            else:
                row[col] = red_slots.get(role_key) or encoders[col].classes_[0]

        row["patch"] = "16.1"  # Default test patch
        row["league"] = "LCK"  # Default competitive league
        row["blue_team"] = encoders["blue_team"].classes_[0]
        row["red_team"] = encoders["red_team"].classes_[0]

        # Populate features
        for col in role_cols:
            row[f"{col}_wr"] = wr_map.get(row[col], global_avg_wr)

        blue_roles = ["blue_top", "blue_jng", "blue_mid", "blue_bot", "blue_sup"]
        red_roles = ["red_top", "red_jng", "red_mid", "red_bot", "red_sup"]

        row["blue_synergy"] = team_synergy_score([row[r] for r in blue_roles])
        row["red_synergy"] = team_synergy_score([row[r] for r in red_roles])
        row["blue_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in blue_roles])
        row["red_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in red_roles])
        row["wr_diff"] = row["blue_team_avg_wr"] - row["red_team_avg_wr"]
        row["synergy_diff"] = row["blue_synergy"] - row["red_synergy"]

        def encode_val(encoder_name, val):
            le = encoders[encoder_name]
            if val in le.classes_:
                return le.transform([val])[0]
            return le.transform([le.classes_[0]])[0]

        for col in role_cols:
            row[f"{col}_enc"] = encode_val(col, row[col])

        row["patch_enc"] = encode_val("patch", row["patch"])
        row["league_enc"] = encode_val("league", row["league"])
        row["blue_team_enc"] = encode_val("blue_team", row["blue_team"])
        row["red_team_enc"] = encode_val("red_team", row["red_team"])

        # Construct dataframe matching exact feature columns order
        X_test = pd.DataFrame([row])
        X_features = X_test[feature_cols].copy()

        # Critical: Cast categorical indexes to pandas category dtype
        cat_cols = [c + "_enc" for c in role_cols] + ["patch_enc", "league_enc"]
        for c in cat_cols:
            X_features[c] = X_features[c].astype("category")

        # Inference
        probs = calibrated_model.predict_proba(X_features)
        blue_win_prob = float(probs[0][1])

        # SHAP Explainability via cached TreeExplainer
        shap_values = shap_explainer(X_features)
        contributions = {}
        for feat, val in zip(feature_cols, shap_values.values[0]):
            contributions[feat] = float(val)

        # Sort and pick top drivers
        top_drivers = []
        for feature, impact in contributions.items():
            top_drivers.append(
                {"feature": feature, "value": 1.0, "impact_on_win_prob": impact}
            )
        top_drivers = sorted(
            top_drivers, key=lambda x: abs(x["impact_on_win_prob"]), reverse=True
        )[:5]

        # Call LangGraph Node
        initial_state = {
            "expected_base_win_prob": blue_win_prob,
            "top_drivers": top_drivers,
            "explanation": "",
        }
        final_state = agent_app.invoke(initial_state)
        return {
            "response": final_state["explanation"],
            "expected_base_win_prob": blue_win_prob,
            "top_drivers": top_drivers,
        }

    return {
        "error": "Invalid request format. Must provide either telemetry or draft_state."
    }


# Duplicated definitions moved to top of file


@router.post("/predict")
def predict_endpoint(payload: DraftPayload, request: Request) -> dict:
    """Fast, zero-load local inference route for real-time draft prediction."""
    # Extract pre-loaded caches from app state
    wr_map = request.app.state.wr_map
    global_avg_wr = request.app.state.global_avg_wr
    pair_map = request.app.state.pair_map
    calibrated_model = request.app.state.calibrated_model
    encoders = request.app.state.encoders
    feature_cols = request.app.state.feature_cols

    def team_synergy_score(champs):
        scores = []
        valid = [c for c in champs if pd.notna(c) and c != ""]
        for c1, c2 in combinations(valid, 2):
            key = tuple(sorted([c1, c2]))
            if key in pair_map:
                scores.append(pair_map[key])
        return np.mean(scores) if scores else 0.5

    role_cols = [
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
    ]

    # Map slot list to roles
    blue_slots = {s.role: s.championId for s in payload.bluePicks if s.championId}
    red_slots = {s.role: s.championId for s in payload.redPicks if s.championId}

    row = {}
    # Use encoder default first class as a safe fallback if champion not yet selected
    for col in role_cols:
        role_key = col.split("_")[1].upper()  # 'top' -> 'TOP'
        if col.startswith("blue"):
            row[col] = blue_slots.get(role_key) or encoders[col].classes_[0]
        else:
            row[col] = red_slots.get(role_key) or encoders[col].classes_[0]

    row["patch"] = "16.1"
    row["league"] = "LCK"
    row["blue_team"] = encoders["blue_team"].classes_[0]
    row["red_team"] = encoders["red_team"].classes_[0]

    # Populate features
    for col in role_cols:
        row[f"{col}_wr"] = wr_map.get(row[col], global_avg_wr)

    blue_roles = ["blue_top", "blue_jng", "blue_mid", "blue_bot", "blue_sup"]
    red_roles = ["red_top", "red_jng", "red_mid", "red_bot", "red_sup"]

    row["blue_synergy"] = team_synergy_score([row[r] for r in blue_roles])
    row["red_synergy"] = team_synergy_score([row[r] for r in red_roles])
    row["blue_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in blue_roles])
    row["red_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in red_roles])
    row["wr_diff"] = row["blue_team_avg_wr"] - row["red_team_avg_wr"]
    row["synergy_diff"] = row["blue_synergy"] - row["red_synergy"]

    def encode_val(encoder_name, val):
        le = encoders[encoder_name]
        if val in le.classes_:
            return le.transform([val])[0]
        return le.transform([le.classes_[0]])[0]

    for col in role_cols:
        row[f"{col}_enc"] = encode_val(col, row[col])

    row["patch_enc"] = encode_val("patch", row["patch"])
    row["league_enc"] = encode_val("league", row["league"])
    row["blue_team_enc"] = encode_val("blue_team", row["blue_team"])
    row["red_team_enc"] = encode_val("red_team", row["red_team"])

    # Build DataFrame
    X_test = pd.DataFrame([row])
    X_features = X_test[feature_cols].copy()

    # Cast categoricals
    cat_cols = [c + "_enc" for c in role_cols] + ["patch_enc", "league_enc"]
    for c in cat_cols:
        X_features[c] = X_features[c].astype("category")

    # Inference
    probs = calibrated_model.predict_proba(X_features)
    blue_win_rate = float(probs[0][1]) * 100

    return {"blueWinRate": blue_win_rate, "redWinRate": 100.0 - blue_win_rate}


@router.get("/champions/metrics")
def get_champions_metrics(request: Request) -> dict:
    """Get pre-loaded actual champion win rates and synergy pairs from parquet caches."""
    return {
        "winrates": request.app.state.wr_map,
        "synergies": {f"{c1}_{c2}": rate for (c1, c2), rate in request.app.state.pair_map.items()},
        "global_avg_wr": request.app.state.global_avg_wr
    }
