import random
import requests
from app.core.config import settings

MODEL_API = "http://localhost:8000"  # Member B/C's FastAPI service

# ---------------------------------------------------------------------------
# Set USE_MOCK_TOOLS=true in your .env to bypass the prediction service and
# return plausible fake data instead. Useful while the model API is not ready.
# ---------------------------------------------------------------------------
_USE_MOCK = settings.USE_MOCK_TOOLS


# --- Mock helpers -----------------------------------------------------------


def _mock_win_prob(blue_picks: list[str], red_picks: list[str]) -> float:
    """Deterministic-ish fake probability based on pick counts."""
    seed = sum(ord(c) for p in blue_picks + red_picks for c in p)
    random.seed(seed)
    return round(random.uniform(0.42, 0.68), 4)


# --- Public tool functions --------------------------------------------------


def get_win_probability(
    blue_picks: list[str], red_picks: list[str], bans: list[str]
) -> dict:
    """Get the calibrated win probability for blue team given the current draft state.
    Call this first to establish a baseline before simulating any picks.

    Args:
        blue_picks: List of champion names picked by blue team so far.
        red_picks: List of champion names picked by red team so far.
        bans: List of all banned champion names.

    Returns:
        Dictionary with blue_win_probability as a float between 0 and 1.
    """
    if _USE_MOCK:
        return {"blue_win_probability": _mock_win_prob(blue_picks, red_picks)}

    r = requests.post(
        f"{MODEL_API}/predict",
        json={"blue_picks": blue_picks, "red_picks": red_picks, "bans": bans},
    )
    return r.json()


def get_shap_explanation(blue_picks: list[str], red_picks: list[str]) -> dict:
    """Get SHAP feature contributions explaining why the win probability is what it is.
    Use this to identify which aspects of the draft are helping or hurting blue team.

    Args:
        blue_picks: Current blue team picks.
        red_picks: Current red team picks.

    Returns:
        Dictionary mapping feature names to their SHAP contribution values.
    """
    if _USE_MOCK:
        features = blue_picks + [f"vs_{p}" for p in red_picks]
        random.seed(42)
        return {f: round(random.uniform(-0.15, 0.20), 4) for f in features}

    r = requests.post(
        f"{MODEL_API}/explain", json={"blue_picks": blue_picks, "red_picks": red_picks}
    )
    return r.json()


def simulate_pick(
    blue_picks: list[str],
    red_picks: list[str],
    bans: list[str],
    proposed_champion: str,
    team: str,
) -> dict:
    """Simulate adding a champion to the draft and return the resulting win probability.
    Use this to compare multiple candidate picks before making a recommendation.
    Call this at least 2-3 times with different champions to find the optimal pick.

    Args:
        blue_picks: Current blue team picks.
        red_picks: Current red team picks.
        bans: Current bans.
        proposed_champion: The champion name to simulate adding.
        team: Either 'blue' or 'red'.

    Returns:
        Dictionary with proposed_champion and resulting win_probability.
    """
    draft = {
        "blue_picks": blue_picks.copy(),
        "red_picks": red_picks.copy(),
        "bans": bans,
    }
    if team == "blue":
        draft["blue_picks"].append(proposed_champion)
    else:
        draft["red_picks"].append(proposed_champion)

    if _USE_MOCK:
        prob = _mock_win_prob(draft["blue_picks"], draft["red_picks"])
        return {"champion": proposed_champion, "win_probability": prob}

    r = requests.post(f"{MODEL_API}/predict", json=draft)
    return {
        "champion": proposed_champion,
        "win_probability": r.json()["blue_win_probability"],
    }


def get_champion_meta_stats(champion_name: str, patch: str = "current") -> dict:
    """Get aggregated meta statistics for a champion across recent competitive matches.
    Use this when the user asks about a specific champion's strength in the current meta.

    Args:
        champion_name: The champion to look up.
        patch: The patch version to query. Defaults to current patch.

    Returns:
        Dictionary with win_rate, pick_rate, and mean_shap_contribution.
    """
    if _USE_MOCK:
        random.seed(sum(ord(c) for c in champion_name))
        return {
            "champion": champion_name,
            "patch": patch,
            "win_rate": round(random.uniform(0.44, 0.58), 4),
            "pick_rate": round(random.uniform(0.05, 0.25), 4),
            "mean_shap_contribution": round(random.uniform(-0.05, 0.12), 4),
        }

    r = requests.get(f"{MODEL_API}/meta/{champion_name}", params={"patch": patch})
    return r.json()


# ---------------------------------------------------------------------------
# Interactions API — JSON Schema tool definitions
# Pass these dicts (not the Python functions) in the `tools` list when calling
# client.interactions.create(). Execute the matching Python function locally
# when a `function_call` step comes back from the model.
# ---------------------------------------------------------------------------

TOOL_GET_WIN_PROBABILITY = {
    "type": "function",
    "name": "get_win_probability",
    "description": (
        "Get the calibrated win probability for blue team given the current draft state. "
        "Call this first to establish a baseline before simulating any picks."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "blue_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of champion names picked by blue team so far.",
            },
            "red_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of champion names picked by red team so far.",
            },
            "bans": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of all banned champion names.",
            },
        },
        "required": ["blue_picks", "red_picks", "bans"],
    },
}

TOOL_GET_SHAP_EXPLANATION = {
    "type": "function",
    "name": "get_shap_explanation",
    "description": (
        "Get SHAP feature contributions explaining why the win probability is what it is. "
        "Use this to identify which aspects of the draft are helping or hurting blue team."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "blue_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current blue team picks.",
            },
            "red_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current red team picks.",
            },
        },
        "required": ["blue_picks", "red_picks"],
    },
}

TOOL_SIMULATE_PICK = {
    "type": "function",
    "name": "simulate_pick",
    "description": (
        "Simulate adding a champion to the draft and return the resulting win probability. "
        "Use this to compare multiple candidate picks before making a recommendation. "
        "Call this at least 2-3 times with different champions to find the optimal pick."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "blue_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current blue team picks.",
            },
            "red_picks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current red team picks.",
            },
            "bans": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Current bans.",
            },
            "proposed_champion": {
                "type": "string",
                "description": "The champion name to simulate adding.",
            },
            "team": {
                "type": "string",
                "enum": ["blue", "red"],
                "description": "The team to add the champion to.",
            },
        },
        "required": ["blue_picks", "red_picks", "bans", "proposed_champion", "team"],
    },
}

TOOL_GET_CHAMPION_META_STATS = {
    "type": "function",
    "name": "get_champion_meta_stats",
    "description": (
        "Get aggregated meta statistics for a champion across recent competitive matches. "
        "Use this when the user asks about a specific champion's strength in the current meta."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "champion_name": {
                "type": "string",
                "description": "The champion to look up.",
            },
            "patch": {
                "type": "string",
                "description": "The patch version to query. Defaults to 'current'.",
            },
        },
        "required": ["champion_name"],
    },
}

# Mapping from tool name → local Python callable (used when dispatching function_call steps)
TOOL_DISPATCH: dict = {
    "get_win_probability": get_win_probability,
    "get_shap_explanation": get_shap_explanation,
    "simulate_pick": simulate_pick,
    "get_champion_meta_stats": get_champion_meta_stats,
}
