from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


ROLE_PART_TO_API = {
    "top": "TOP",
    "jng": "JUNGLE",
    "mid": "MID",
    "bot": "BOTTOM",
    "sup": "SUPPORT",
}

API_ROLE_TO_PART = {value: key for key, value in ROLE_PART_TO_API.items()}
API_ROLE_TO_PART.update(
    {
        "TOP": "top",
        "JNG": "jng",
        "JUNGLE": "jng",
        "MID": "mid",
        "BOT": "bot",
        "BOTTOM": "bot",
        "ADC": "bot",
        "SUP": "sup",
        "SUPPORT": "sup",
    }
)

ROLE_PARTS = ["top", "jng", "mid", "bot", "sup"]
ROLE_COLS = [
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

BLUE_ROLE_COLS = ROLE_COLS[:5]
RED_ROLE_COLS = ROLE_COLS[5:]
DEFAULT_PATCH = "UNKNOWN"
DEFAULT_LEAGUE = "UNKNOWN"


def normalize_role_part(role: Optional[str]) -> Optional[str]:
    if not role:
        return None
    cleaned = role.strip()
    if not cleaned:
        return None
    return API_ROLE_TO_PART.get(cleaned.upper(), cleaned.lower())


def get_slot_champion(slots: Mapping[str, str], role_part: str) -> Optional[str]:
    api_role = ROLE_PART_TO_API.get(role_part)
    if api_role and slots.get(api_role):
        return slots[api_role]
    if slots.get(role_part):
        return slots[role_part]
    upper_part = role_part.upper()
    if slots.get(upper_part):
        return slots[upper_part]
    return None


def role_key(champion: Optional[str], role_part: Optional[str]) -> Optional[str]:
    if not champion or not role_part:
        return None
    return f"{champion}_{role_part}"


def encode_value(encoders: Mapping[str, Any], encoder_name: str, value: str) -> int:
    encoder = encoders.get(encoder_name)
    if encoder is None:
        return 0
    classes = list(encoder.classes_)
    if value in classes:
        return int(encoder.transform([value])[0])
    return int(encoder.transform([classes[0]])[0])


def safe_encode_role_champion(
    encoders: Mapping[str, Any],
    encoder_name: str,
    champion: Optional[str],
    role_part: str,
) -> int:
    encoder = encoders.get(encoder_name)
    if encoder is None:
        return -1

    if champion:
        value = f"{champion}_{role_part}"
    else:
        value = f"UNKNOWN_{role_part}"

    classes = list(encoder.classes_)

    if value in classes:
        return int(encoder.transform([value])[0])

    unknown_role = f"UNKNOWN_{role_part}"
    if unknown_role in classes:
        return int(encoder.transform([unknown_role])[0])

    if "UNKNOWN" in classes:
        return int(encoder.transform(["UNKNOWN"])[0])

    return -1


def role_wr(
    champion: Optional[str],
    role_part: str,
    champ_role_wr_map: Mapping[str, float],
    champ_wr_global_map: Mapping[str, float],
    global_avg_wr: float,
) -> float:
    key = role_key(champion, role_part)
    if key and key in champ_role_wr_map:
        return float(champ_role_wr_map[key])
    if champion and champion in champ_wr_global_map:
        return float(champ_wr_global_map[champion])
    return float(global_avg_wr)


def team_synergy(role_keys: Sequence[Optional[str]], pair_map: Mapping[Tuple[str, str], float]) -> float:
    valid_keys = [key for key in role_keys if key]
    if len(valid_keys) < 2:
        return 0.5

    scores: List[float] = []
    for first, second in combinations(valid_keys, 2):
        lookup = tuple(sorted([first, second]))
        if lookup in pair_map:
            scores.append(float(pair_map[lookup]))
    return float(np.mean(scores)) if scores else 0.5


def build_model_row(
    blue_slots: Mapping[str, str],
    red_slots: Mapping[str, str],
    model_state: Any,
    *,
    patch: str = DEFAULT_PATCH,
    league: str = DEFAULT_LEAGUE,
) -> Dict[str, Any]:
    champ_role_wr_map = model_state.champ_role_wr_map
    champ_wr_global_map = model_state.wr_map
    global_avg_wr = float(model_state.global_avg_wr)
    pair_map = model_state.pair_map
    encoders = model_state.encoders

    row: Dict[str, Any] = {
        "patch": patch,
        "league": league,
        "patch_enc": encode_value(encoders, "patch", patch),
        "league_enc": encode_value(encoders, "league", league),
    }

    selected_role_keys: Dict[str, Optional[str]] = {}
    picks_filled = 0

    for col in ROLE_COLS:
        team, role_part = col.split("_", 1)
        slots = blue_slots if team == "blue" else red_slots
        champion = get_slot_champion(slots, role_part)
        if champion:
            picks_filled += 1

        keyed_champion = role_key(champion, role_part)
        row[col] = champion or ""
        row[f"{col}_enc"] = safe_encode_role_champion(encoders, col, champion, role_part)
        selected_role_keys[col] = keyed_champion
        row[f"{col}_wr"] = role_wr(
            champion,
            role_part,
            champ_role_wr_map,
            champ_wr_global_map,
            global_avg_wr,
        )

    blue_keys = [selected_role_keys[col] for col in BLUE_ROLE_COLS]
    red_keys = [selected_role_keys[col] for col in RED_ROLE_COLS]

    row["blue_synergy"] = team_synergy(blue_keys, pair_map)
    row["red_synergy"] = team_synergy(red_keys, pair_map)
    row["blue_team_avg_wr"] = float(np.mean([row[f"{col}_wr"] for col in BLUE_ROLE_COLS]))
    row["red_team_avg_wr"] = float(np.mean([row[f"{col}_wr"] for col in RED_ROLE_COLS]))
    row["wr_diff"] = row["blue_team_avg_wr"] - row["red_team_avg_wr"]
    row["synergy_diff"] = row["blue_synergy"] - row["red_synergy"]
    row["picks_filled"] = picks_filled

    return row


def feature_frame(rows: Sequence[Mapping[str, Any]], feature_cols: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    missing = [col for col in feature_cols if col not in frame.columns]
    if missing:
        print("WARNING: Missing model features:", missing)
    return frame.reindex(columns=feature_cols, fill_value=0.0).copy()


def postprocess_probability(probability: float, picks_filled: Any) -> float:
    proba = float(probability)
    try:
        filled = float(picks_filled)
    except (TypeError, ValueError):
        filled = 10.0
    confidence_weight = max(0.0, min(1.0, filled / 10.0))
    shrunk = 0.5 + (proba - 0.5) * confidence_weight
    return max(0.15, min(0.85, shrunk))


def predict_blue_win_probability(model: Any, rows: Sequence[Mapping[str, Any]], feature_cols: Sequence[str]) -> List[float]:
    features = feature_frame(rows, feature_cols)
    probabilities = model.predict_proba(features)
    return [
        postprocess_probability(float(prob[1]), row.get("picks_filled", 10))
        for prob, row in zip(probabilities, rows)
    ]


def shap_contributions(shap_explainer: Any, row: Mapping[str, Any], feature_cols: Sequence[str]) -> Dict[str, float]:
    features = feature_frame([row], feature_cols)
    shap_values = shap_explainer(features)
    values = getattr(shap_values, "values", shap_values)
    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[2] > 1:
            values = values[0, :, 1]
        else:
            values = values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]

    return {feature: float(value) for feature, value in zip(feature_cols, values)}


def top_drivers(contributions: Mapping[str, float], row: Mapping[str, Any], limit: int = 5) -> List[Dict[str, float]]:
    drivers = [
        {
            "feature": feature,
            "value": float(row.get(feature, 0.0)),
            "impact_on_win_prob": float(impact),
        }
        for feature, impact in contributions.items()
    ]
    return sorted(drivers, key=lambda driver: abs(driver["impact_on_win_prob"]), reverse=True)[:limit]


def role_options_for_champion(champion: str, champ_role_wr_map: Mapping[str, float]) -> List[Tuple[str, float]]:
    prefix = f"{champion}_"
    options: List[Tuple[str, float]] = []
    for key, win_rate in champ_role_wr_map.items():
        if not key.startswith(prefix):
            continue
        role_part = key[len(prefix) :]
        if role_part in ROLE_PARTS:
            options.append((role_part, float(win_rate)))
    return sorted(options, key=lambda item: item[1], reverse=True)


def assign_picks_to_roles(picks: Iterable[str], is_blue: bool, model_state: Any) -> Dict[str, str]:
    prefix = "blue_" if is_blue else "red_"
    remaining = set(ROLE_PARTS)
    assigned: Dict[str, str] = {}

    for champion in picks:
        preferred_roles = [role for role, _ in role_options_for_champion(champion, model_state.champ_role_wr_map)]
        preferred_roles.extend(ROLE_PARTS)
        for role_part in preferred_roles:
            if role_part in remaining:
                assigned[f"{prefix}{role_part}"] = champion
                remaining.remove(role_part)
                break

    return assigned


def slots_from_simple_picks(blue_picks: Iterable[str], red_picks: Iterable[str], model_state: Any) -> Tuple[Dict[str, str], Dict[str, str]]:
    blue_assigned = assign_picks_to_roles(blue_picks, True, model_state)
    red_assigned = assign_picks_to_roles(red_picks, False, model_state)

    blue_slots = {
        ROLE_PART_TO_API[col.split("_", 1)[1]]: champion
        for col, champion in blue_assigned.items()
    }
    red_slots = {
        ROLE_PART_TO_API[col.split("_", 1)[1]]: champion
        for col, champion in red_assigned.items()
    }
    return blue_slots, red_slots


def available_champions(model_state: Any) -> List[str]:
    return sorted(model_state.wr_map.keys())


def active_role_part(active_role: Optional[str]) -> str:
    return normalize_role_part(active_role) or "mid"
