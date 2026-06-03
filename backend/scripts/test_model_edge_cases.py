"""
Smoke/regression checks for draft prediction edge cases.

Run from the backend folder:
    .venv\\Scripts\\python.exe scripts\\test_model_edge_cases.py

The script uses FastAPI's TestClient, so it exercises the real startup path,
cached parquet/model artifacts, and API response shape without requiring a
separate uvicorn process.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
CHAMPIONS_JSON = PROJECT_DIR / "frontend" / "src" / "assets" / "data" / "champions.json"
ROLES = ["TOP", "JUNGLE", "MID", "BOTTOM", "SUPPORT"]
V2_FEATURE_COLS = [
    "patch_enc",
    "league_enc",
    "blue_top_wr",
    "blue_jng_wr",
    "blue_mid_wr",
    "blue_bot_wr",
    "blue_sup_wr",
    "red_top_wr",
    "red_jng_wr",
    "red_mid_wr",
    "red_bot_wr",
    "red_sup_wr",
    "blue_synergy",
    "red_synergy",
    "blue_team_avg_wr",
    "red_team_avg_wr",
    "wr_diff",
    "synergy_diff",
    "picks_filled",
]

sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402


def load_champions() -> list[dict[str, Any]]:
    with CHAMPIONS_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)["champions"]


def champion_names_with_tag(champions: list[dict[str, Any]], tag: str, count: int) -> list[str]:
    names = [c["name"] for c in champions if tag in c.get("tags", [])]
    if len(names) < count:
        raise AssertionError(f"Need {count} champions tagged {tag}, found {len(names)}")
    return names[:count]


def slot_payload(champion_names: list[str]) -> list[dict[str, str]]:
    if len(champion_names) > len(ROLES):
        raise AssertionError(f"Only {len(ROLES)} slot roles are supported per team")
    return [
        {"role": role, "player": f"{role.lower()}_player", "championId": champion}
        for role, champion in zip(ROLES, champion_names)
    ]


def draft_payload(
    blue: list[str],
    red: list[str] | None = None,
    blue_bans: list[str] | None = None,
    red_bans: list[str] | None = None,
    active_team: str = "blue",
    active_action: str = "pick",
    active_role: str = "SUPPORT",
) -> dict[str, Any]:
    return {
        "bluePicks": slot_payload(blue),
        "redPicks": slot_payload(red or []),
        "blueBans": blue_bans or [],
        "redBans": red_bans or [],
        "activeTeam": active_team,
        "activeAction": active_action,
        "activeRole": active_role,
        "stepIndex": len(blue) + len(red or []) + len(blue_bans or []) + len(red_bans or []),
    }


def assert_probability_pair(body: dict[str, Any]) -> None:
    blue = body.get("blueWinRate")
    red = body.get("redWinRate")
    if not isinstance(blue, (int, float)) or not isinstance(red, (int, float)):
        raise AssertionError(f"Expected numeric blue/red win rates, got {body}")
    if not math.isfinite(blue) or not math.isfinite(red):
        raise AssertionError(f"Expected finite win rates, got {body}")
    if not 0.0 <= blue <= 100.0 or not 0.0 <= red <= 100.0:
        raise AssertionError(f"Win rates out of percentage bounds: {body}")
    if not math.isclose(blue + red, 100.0, abs_tol=1e-6):
        raise AssertionError(f"Blue/red win rates should sum to 100, got {body}")


def assert_root_probability(body: dict[str, Any]) -> None:
    prob = body.get("blue_win_probability")
    if not isinstance(prob, (int, float)):
        raise AssertionError(f"Expected numeric blue_win_probability, got {body}")
    if not math.isfinite(prob) or not 0.0 <= prob <= 1.0:
        raise AssertionError(f"Probability out of [0, 1] bounds: {body}")


def assert_feature_contributions(body: dict[str, Any]) -> None:
    missing = [feature for feature in V2_FEATURE_COLS if feature not in body]
    if missing:
        raise AssertionError(f"Missing v2 SHAP features: {missing}")
    for feature in V2_FEATURE_COLS:
        value = body[feature]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AssertionError(f"Expected finite contribution for {feature}, got {value!r}")


def post_json(client: TestClient, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    if response.status_code != 200:
        raise AssertionError(f"{path} returned {response.status_code}: {response.text}")
    return response.json()


def run_case(name: str, fn) -> None:
    print(f"[RUN] {name}")
    fn()
    print(f"[OK]  {name}")


def main() -> int:
    champions = load_champions()
    supports = champion_names_with_tag(champions, "Support", 5)
    marksmen = champion_names_with_tag(champions, "Marksman", 5)
    fighters = champion_names_with_tag(champions, "Fighter", 5)
    unique_bans = champion_names_with_tag(champions, "Mage", 3)

    print(f"Using support edge-case team: {', '.join(supports)}")

    with TestClient(app) as client:

        def model_contract_is_v2() -> None:
            if list(client.app.state.feature_cols) != V2_FEATURE_COLS:
                raise AssertionError(f"Unexpected v2 feature columns: {client.app.state.feature_cols}")
            if not hasattr(client.app.state, "champ_role_wr_map"):
                raise AssertionError("Missing v2 champion-role win-rate map")
            if not hasattr(client.app.state, "wr_map"):
                raise AssertionError("Missing v2 global champion win-rate map")

        def empty_draft_returns_neutral_prediction() -> None:
            body = post_json(client, "/api/v1/predict", draft_payload([], []))
            assert_probability_pair(body)
            if body != {"blueWinRate": 50.0, "redWinRate": 50.0}:
                raise AssertionError(f"Expected exact neutral empty draft, got {body}")

        def five_supports_are_accepted_and_finite() -> None:
            body = post_json(client, "/api/v1/predict", draft_payload(supports, marksmen))
            assert_probability_pair(body)

        def five_supports_are_deterministic() -> None:
            payload = draft_payload(supports, fighters, blue_bans=unique_bans[:1])
            first = post_json(client, "/api/v1/predict", payload)
            second = post_json(client, "/api/v1/predict", payload)
            assert_probability_pair(first)
            assert_probability_pair(second)
            if first != second:
                raise AssertionError(f"Expected deterministic output, got {first} then {second}")

        def duplicate_champion_niche_case_does_not_crash() -> None:
            repeated_support = [supports[0]] * 5
            body = post_json(client, "/api/v1/predict", draft_payload(repeated_support, marksmen))
            assert_probability_pair(body)

        def unknown_champion_falls_back_safely() -> None:
            unusual = supports[:4] + ["Definitely Not A Champion"]
            body = post_json(client, "/api/v1/predict", draft_payload(unusual, marksmen))
            assert_probability_pair(body)

        def root_predict_simple_tool_endpoint_stays_valid() -> None:
            payload = {"blue_picks": supports, "red_picks": marksmen, "bans": unique_bans}
            first = post_json(client, "/predict", payload)
            second = post_json(client, "/predict", payload)
            assert_root_probability(first)
            assert_root_probability(second)
            if first != second:
                raise AssertionError(f"Expected deterministic root /predict output, got {first} then {second}")

        def batch_predict_options_uses_v2_contract() -> None:
            payload = {
                "draftState": draft_payload(supports[:1], marksmen[:1], active_role="MID"),
                "options": [
                    {
                        "entityId": "ahri-MID",
                        "championId": "Ahri",
                        "championName": "Ahri",
                        "role": "MID",
                    },
                    {
                        "entityId": "orianna-MID",
                        "championId": "Orianna",
                        "championName": "Orianna",
                        "role": "MID",
                    },
                ],
            }
            body = post_json(client, "/api/v1/predict-options", payload)
            option_rates = body.get("optionWinRates")
            if set(option_rates or {}) != {"ahri-MID", "orianna-MID"}:
                raise AssertionError(f"Unexpected option win-rate response: {body}")
            for value in option_rates.values():
                if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 100.0:
                    raise AssertionError(f"Invalid option win rate: {body}")

        def metrics_expose_v2_maps() -> None:
            response = client.get("/api/v1/champions/metrics")
            if response.status_code != 200:
                raise AssertionError(f"metrics returned {response.status_code}: {response.text}")
            body = response.json()
            if body.get("model_version") != "v2":
                raise AssertionError(f"Expected v2 metrics payload, got {body}")
            if not body.get("role_winrates") or not body.get("winrates"):
                raise AssertionError(f"Expected v2 win-rate maps in metrics payload, got {body.keys()}")

        def root_explain_uses_v2_features() -> None:
            payload = {"blue_picks": supports[:3], "red_picks": marksmen[:3]}
            body = post_json(client, "/explain", payload)
            assert_feature_contributions(body)

        cases = [
            ("model contract is v2", model_contract_is_v2),
            ("empty draft returns 50/50", empty_draft_returns_neutral_prediction),
            ("five support heroes are accepted", five_supports_are_accepted_and_finite),
            ("five support heroes are deterministic", five_supports_are_deterministic),
            ("duplicate champion niche case does not crash", duplicate_champion_niche_case_does_not_crash),
            ("unknown champion falls back safely", unknown_champion_falls_back_safely),
            ("root simple /predict endpoint stays valid", root_predict_simple_tool_endpoint_stays_valid),
            ("batch predict-options uses v2 contract", batch_predict_options_uses_v2_contract),
            ("metrics expose v2 maps", metrics_expose_v2_maps),
            ("root explain uses v2 features", root_explain_uses_v2_features),
        ]

        for name, fn in cases:
            run_case(name, fn)

    print("All edge-case model checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
