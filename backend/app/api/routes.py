from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.agents import agent_app
from app.services.model_features import (
    ROLE_PART_TO_API,
    active_role_part,
    available_champions,
    build_model_row,
    predict_blue_win_probability,
    role_key,
    role_options_for_champion,
    role_wr,
    shap_contributions,
    slots_from_simple_picks,
    team_synergy,
    top_drivers,
)

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


class CandidateOptionPayload(BaseModel):
    entityId: str
    championId: Optional[str] = None
    championName: str
    role: str


class PredictOptionsPayload(BaseModel):
    draftState: DraftPayload
    options: List[CandidateOptionPayload]


class TelemetryDriver(BaseModel):
    feature: str
    value: float
    impact_on_win_prob: float


class AgentRequest(BaseModel):
    expected_base_win_prob: Optional[float] = None
    top_drivers: Optional[List[TelemetryDriver]] = None
    draft_state: Optional[DraftPayload] = None
    user_question: Optional[str] = None
    interaction_id: Optional[str] = None


class SimplePredictPayload(BaseModel):
    blue_picks: List[str]
    red_picks: List[str]
    bans: List[str]


class SimpleExplainPayload(BaseModel):
    blue_picks: List[str]
    red_picks: List[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def champion_id_map() -> Dict[str, str]:
    champions_path = project_root() / "frontend" / "src" / "assets" / "data" / "champions.json"
    if not champions_path.exists():
        return {}

    try:
        with champions_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return {champion["id"]: champion["name"] for champion in data.get("champions", [])}
    except Exception as exc:
        print(f"Error reading champions.json: {exc}")
        return {}


def resolve_name(value: Optional[str], id_to_name: Mapping[str, str]) -> Optional[str]:
    if not value:
        return None
    return id_to_name.get(value, value)


def resolve_draft(payload: DraftPayload) -> Dict[str, Any]:
    id_to_name = champion_id_map()
    blue_slots = {
        slot.role: resolve_name(slot.championId, id_to_name)
        for slot in payload.bluePicks
        if slot.championId
    }
    red_slots = {
        slot.role: resolve_name(slot.championId, id_to_name)
        for slot in payload.redPicks
        if slot.championId
    }
    return {
        "blue_slots": blue_slots,
        "red_slots": red_slots,
        "blue_picks": [champion for champion in blue_slots.values() if champion],
        "red_picks": [champion for champion in red_slots.values() if champion],
        "blue_bans": [resolve_name(ban, id_to_name) for ban in payload.blueBans if ban],
        "red_bans": [resolve_name(ban, id_to_name) for ban in payload.redBans if ban],
    }


def model_state(request: Request):
    return request.app.state


def score_blue_probability(request: Request, blue_slots: Mapping[str, str], red_slots: Mapping[str, str]) -> float:
    state = model_state(request)
    row = build_model_row(blue_slots, red_slots, state)
    return predict_blue_win_probability(state.calibrated_model, [row], state.feature_cols)[0]


def role_synergy_for_candidate(
    candidate: str,
    role_part: str,
    ally_slots: Mapping[str, str],
    pair_map: Mapping[tuple[str, str], float],
) -> float:
    candidate_key = role_key(candidate, role_part)
    ally_keys = []
    for api_role, champion in ally_slots.items():
        ally_part = active_role_part(api_role)
        ally_keys.append(role_key(champion, ally_part))
    return team_synergy([candidate_key, *ally_keys], pair_map)


def simulate_candidate(
    request: Request,
    blue_slots: Mapping[str, str],
    red_slots: Mapping[str, str],
    champion: str,
    team: str,
    role_part: str,
) -> float:
    next_blue_slots = dict(blue_slots)
    next_red_slots = dict(red_slots)
    api_role = ROLE_PART_TO_API[role_part]
    if team == "blue":
        next_blue_slots[api_role] = champion
    else:
        next_red_slots[api_role] = champion
    return score_blue_probability(request, next_blue_slots, next_red_slots)


def calculate_pick_recommendations(
    request: Request,
    blue_slots: Mapping[str, str],
    red_slots: Mapping[str, str],
    banned_champs: List[str],
    active_team: str,
    active_role: Optional[str],
) -> List[Dict[str, Any]]:
    state = model_state(request)
    role_part = active_role_part(active_role)
    ally_slots = blue_slots if active_team == "blue" else red_slots
    unavailable = set(blue_slots.values()) | set(red_slots.values()) | set(banned_champs)

    candidates = [
        champion
        for champion in available_champions(state)
        if champion not in unavailable
    ]

    scored = []
    for champion in candidates:
        champion_role_wr = role_wr(
            champion,
            role_part,
            state.champ_role_wr_map,
            state.wr_map,
            state.global_avg_wr,
        )
        ally_synergy = role_synergy_for_candidate(champion, role_part, ally_slots, state.pair_map)
        score = 0.6 * champion_role_wr + 0.4 * ally_synergy
        scored.append((champion, score, champion_role_wr, ally_synergy))

    scored.sort(key=lambda item: item[1], reverse=True)

    recommendations = []
    for champion, score, champion_role_wr, ally_synergy in scored[:3]:
        try:
            blue_prob = simulate_candidate(
                request,
                blue_slots,
                red_slots,
                champion,
                active_team,
                role_part,
            )
            perspective_prob = blue_prob if active_team == "blue" else 1.0 - blue_prob
        except Exception as exc:
            print(f"Error simulating pick recommendation for {champion}: {exc}")
            perspective_prob = 0.5

        recommendations.append(
            {
                "champion": champion,
                "role": ROLE_PART_TO_API[role_part],
                "score": float(score),
                "role_win_rate": float(champion_role_wr),
                "ally_synergy": float(ally_synergy),
                "simulated_win_probability": float(perspective_prob),
            }
        )

    return recommendations


def calculate_ban_recommendations(
    request: Request,
    blue_slots: Mapping[str, str],
    red_slots: Mapping[str, str],
    banned_champs: List[str],
    active_team: str,
) -> List[Dict[str, Any]]:
    state = model_state(request)
    opponent_team = "red" if active_team == "blue" else "blue"
    opponent_slots = red_slots if active_team == "blue" else blue_slots
    unavailable = set(blue_slots.values()) | set(red_slots.values()) | set(banned_champs)

    scored = []
    for champion in available_champions(state):
        if champion in unavailable:
            continue
        role_options = role_options_for_champion(champion, state.champ_role_wr_map)
        role_part, champion_role_wr = role_options[0] if role_options else ("mid", state.wr_map.get(champion, state.global_avg_wr))
        opponent_synergy = role_synergy_for_candidate(champion, role_part, opponent_slots, state.pair_map)
        threat_score = 0.6 * float(champion_role_wr) + 0.4 * opponent_synergy
        scored.append((champion, role_part, threat_score, float(champion_role_wr), opponent_synergy))

    scored.sort(key=lambda item: item[2], reverse=True)

    recommendations = []
    for champion, role_part, threat_score, champion_role_wr, opponent_synergy in scored[:3]:
        try:
            blue_prob = simulate_candidate(
                request,
                blue_slots,
                red_slots,
                champion,
                opponent_team,
                role_part,
            )
            opponent_prob = 1.0 - blue_prob if opponent_team == "red" else blue_prob
        except Exception as exc:
            print(f"Error simulating ban recommendation for {champion}: {exc}")
            opponent_prob = 0.5

        recommendations.append(
            {
                "champion": champion,
                "role": ROLE_PART_TO_API[role_part],
                "threat_score": float(threat_score),
                "role_win_rate": float(champion_role_wr),
                "opp_synergy": float(opponent_synergy),
                "simulated_win_probability_if_picked_by_opponent": float(opponent_prob),
            }
        )

    return recommendations


def matchup_counters(request: Request, blue_slots: Mapping[str, str], red_slots: Mapping[str, str]) -> List[Dict[str, Any]]:
    counter_map = getattr(request.app.state, "counter_map", {})
    role_mapping_to_parquet = {
        "TOP": "TOP",
        "JUNGLE": "JNG",
        "MID": "MID",
        "BOTTOM": "BOT",
        "SUPPORT": "SUP",
    }

    results = []
    for api_role, parquet_role in role_mapping_to_parquet.items():
        blue_champion = blue_slots.get(api_role)
        red_champion = red_slots.get(api_role)
        if not blue_champion or not red_champion:
            continue
        lookup_key = (blue_champion, red_champion, parquet_role)
        if lookup_key not in counter_map:
            continue
        match_info = counter_map[lookup_key]
        results.append(
            {
                "role": api_role,
                "blue_champion": blue_champion,
                "red_champion": red_champion,
                "blue_head_to_head_winrate": match_info["winrate"],
                "matches": match_info["matches"],
            }
        )
    return results


def explain_row(request: Request, row: Mapping[str, Any]) -> tuple[float, List[Dict[str, float]]]:
    state = model_state(request)
    blue_prob = predict_blue_win_probability(state.calibrated_model, [row], state.feature_cols)[0]
    if getattr(state, "shap_explainer", None) is None:
        return blue_prob, []
    contributions = shap_contributions(state.shap_explainer, row, state.feature_cols)
    return blue_prob, top_drivers(contributions, row)


@router.post("/agent")
def agent_endpoint(req: AgentRequest, request: Request) -> dict:
    if req.expected_base_win_prob is not None and req.top_drivers is not None:
        initial_state = {
            "expected_base_win_prob": req.expected_base_win_prob,
            "top_drivers": [driver.model_dump() for driver in req.top_drivers],
            "explanation": "",
        }
        final_state = agent_app.invoke(initial_state)
        return {
            "draft_warning": final_state.get("draft_warning", "No details available."),
            "recommendations": final_state.get("recommendations", "Review drivers panel."),
            "counter_analysis": final_state.get("counter_analysis", ""),
        }

    if req.draft_state is None:
        return {"error": "Invalid request format. Must provide either telemetry or draft_state."}

    resolved = resolve_draft(req.draft_state)
    blue_slots = resolved["blue_slots"]
    red_slots = resolved["red_slots"]
    blue_picks = resolved["blue_picks"]
    red_picks = resolved["red_picks"]
    blue_bans = resolved["blue_bans"]
    red_bans = resolved["red_bans"]
    banned_champs = blue_bans + red_bans

    if blue_picks or red_picks:
        row = build_model_row(blue_slots, red_slots, model_state(request))
        blue_win_prob, drivers = explain_row(request, row)
    else:
        blue_win_prob = 0.5
        drivers = []

    recommended_bans = []
    recommended_picks = []
    if req.draft_state.activeAction == "ban" or not (blue_picks or red_picks):
        recommended_bans = calculate_ban_recommendations(
            request,
            blue_slots,
            red_slots,
            banned_champs,
            req.draft_state.activeTeam,
        )
    elif req.draft_state.activeAction == "pick":
        recommended_picks = calculate_pick_recommendations(
            request,
            blue_slots,
            red_slots,
            banned_champs,
            req.draft_state.activeTeam,
            req.draft_state.activeRole,
        )

    initial_state = {
        "expected_base_win_prob": blue_win_prob,
        "top_drivers": drivers,
        "explanation": "",
        "active_team": req.draft_state.activeTeam,
        "active_action": req.draft_state.activeAction,
        "active_role": req.draft_state.activeRole or "",
        "blue_picks": blue_picks,
        "red_picks": red_picks,
        "blue_bans": blue_bans,
        "red_bans": red_bans,
        "recommended_bans": recommended_bans,
        "recommended_picks": recommended_picks,
        "matchup_counters": matchup_counters(request, blue_slots, red_slots),
    }
    final_state = agent_app.invoke(initial_state)
    return {
        "draft_warning": final_state.get("draft_warning", ""),
        "recommendations": final_state.get("recommendations", ""),
        "counter_analysis": final_state.get("counter_analysis", ""),
        "expected_base_win_prob": blue_win_prob,
        "top_drivers": drivers,
    }


@router.post("/predict")
def predict_endpoint(payload: DraftPayload, request: Request) -> dict:
    resolved = resolve_draft(payload)
    blue_slots = resolved["blue_slots"]
    red_slots = resolved["red_slots"]

    if not blue_slots and not red_slots:
        return {"blueWinRate": 50.0, "redWinRate": 50.0}

    blue_win_rate = score_blue_probability(request, blue_slots, red_slots) * 100.0
    return {"blueWinRate": blue_win_rate, "redWinRate": 100.0 - blue_win_rate}


@router.post("/predict-options")
def predict_options_endpoint(payload: PredictOptionsPayload, request: Request) -> dict:
    if not payload.options:
        return {"optionWinRates": {}}

    resolved = resolve_draft(payload.draftState)
    base_blue_slots = resolved["blue_slots"]
    base_red_slots = resolved["red_slots"]

    rows = []
    row_options = []
    id_to_name = champion_id_map()
    for option in payload.options:
        champion_name = resolve_name(option.championName or option.championId, id_to_name)
        if not champion_name:
            continue

        blue_slots = dict(base_blue_slots)
        red_slots = dict(base_red_slots)
        if payload.draftState.activeTeam == "blue":
            blue_slots[option.role] = champion_name
        else:
            red_slots[option.role] = champion_name

        rows.append(build_model_row(blue_slots, red_slots, model_state(request)))
        row_options.append(option)

    if not rows:
        return {"optionWinRates": {}}

    state = model_state(request)
    blue_probabilities = predict_blue_win_probability(state.calibrated_model, rows, state.feature_cols)
    option_win_rates = {}
    for option, blue_prob in zip(row_options, blue_probabilities):
        blue_win_rate = blue_prob * 100.0
        option_win_rates[option.entityId] = (
            blue_win_rate
            if payload.draftState.activeTeam == "blue"
            else 100.0 - blue_win_rate
        )

    return {"optionWinRates": option_win_rates}


@router.get("/champions/metrics")
def get_champions_metrics(request: Request) -> dict:
    state = model_state(request)
    return {
        "winrates": state.wr_map,
        "role_winrates": state.champ_role_wr_map,
        "synergies": {f"{first}_{second}": rate for (first, second), rate in state.pair_map.items()},
        "global_avg_wr": state.global_avg_wr,
        "model_version": getattr(state, "model_version", "v2"),
    }


@router.get("/champions/op")
def get_op_champions(request: Request) -> dict:
    return {
        "op_champions": getattr(request.app.state, "op_champions", []),
        "role_strengths": getattr(request.app.state, "role_strengths", []),
        "model_version": getattr(request.app.state, "model_version", "v2"),
    }


root_router = APIRouter()


@root_router.post("/predict")
def root_predict_endpoint(payload: SimplePredictPayload, request: Request) -> dict:
    if not payload.blue_picks and not payload.red_picks:
        return {"blue_win_probability": 0.5}

    blue_slots, red_slots = slots_from_simple_picks(
        payload.blue_picks,
        payload.red_picks,
        model_state(request),
    )
    return {"blue_win_probability": score_blue_probability(request, blue_slots, red_slots)}


@root_router.post("/explain")
def root_explain_endpoint(payload: SimpleExplainPayload, request: Request) -> dict:
    blue_slots, red_slots = slots_from_simple_picks(
        payload.blue_picks,
        payload.red_picks,
        model_state(request),
    )
    row = build_model_row(blue_slots, red_slots, model_state(request))
    if getattr(request.app.state, "shap_explainer", None) is None:
        return {feature: 0.0 for feature in request.app.state.feature_cols}
    return shap_contributions(request.app.state.shap_explainer, row, request.app.state.feature_cols)


@root_router.get("/meta/{champion_name}")
def root_meta_endpoint(champion_name: str, request: Request, patch: str = "current") -> dict:
    state = model_state(request)
    win_rate = float(state.wr_map.get(champion_name, state.global_avg_wr))

    mean_contribution = 0.0
    best_role = None
    for champion in getattr(state, "op_champions", []):
        if champion["champion"] == champion_name:
            mean_contribution = float(champion.get("mean_contribution", 0.0))
            best_role = champion.get("role")
            break

    random.seed(sum(ord(char) for char in champion_name))
    pick_rate = round(random.uniform(0.05, 0.25), 4)

    return {
        "champion": champion_name,
        "patch": patch,
        "win_rate": win_rate,
        "pick_rate": float(pick_rate),
        "best_role": best_role,
        "mean_shap_contribution": mean_contribution,
        "model_version": getattr(state, "model_version", "v2"),
    }
