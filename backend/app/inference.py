from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import pickle
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


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

COL_TO_ROLE = {col: col.split("_")[1] for col in ROLE_COLS}
BLUE_ROLES = [col for col in ROLE_COLS if col.startswith("blue_")]
RED_ROLES = [col for col in ROLE_COLS if col.startswith("red_")]


@dataclass
class ModelArtifacts:
    model: Any
    feature_cols: List[str]
    encoders: Dict[str, LabelEncoder]
    champ_role_wr_map: Dict[str, float]
    champ_wr_global_map: Dict[str, float]
    pair_map: Dict[Tuple[str, str], float]
    global_avg_wr: float


def load_artifacts(base_path: Path) -> ModelArtifacts:
    with open(base_path / "calibrated_model_v2.pkl", "rb") as handle:
        model = pickle.load(handle)
    with open(base_path / "feature_cols_v2.pkl", "rb") as handle:
        feature_cols = pickle.load(handle)
    with open(base_path / "encoders_v2.pkl", "rb") as handle:
        encoders = pickle.load(handle)
    with open(base_path / "champ_role_wr_map.pkl", "rb") as handle:
        champ_role_wr_map = pickle.load(handle)
    with open(base_path / "champ_wr_global_map.pkl", "rb") as handle:
        champ_wr_global_map = pickle.load(handle)
    with open(base_path / "pair_map.pkl", "rb") as handle:
        pair_map = pickle.load(handle)
    with open(base_path / "global_avg_wr.pkl", "rb") as handle:
        global_avg_wr = pickle.load(handle)

    return ModelArtifacts(
        model=model,
        feature_cols=feature_cols,
        encoders=encoders,
        champ_role_wr_map=champ_role_wr_map,
        champ_wr_global_map=champ_wr_global_map,
        pair_map=pair_map,
        global_avg_wr=float(global_avg_wr),
    )


def get_wr(champion: str | None, role: str, artifacts: ModelArtifacts) -> float:
    if champion is None or champion == "":
        return artifacts.global_avg_wr
    role_key = f"{champion}_{role}"
    if role_key in artifacts.champ_role_wr_map:
        return artifacts.champ_role_wr_map[role_key]
    if champion in artifacts.champ_wr_global_map:
        return artifacts.champ_wr_global_map[champion]
    return artifacts.global_avg_wr


def team_synergy_score(
    champs: Iterable[str | None],
    roles: Iterable[str],
    artifacts: ModelArtifacts,
) -> float:
    role_champs = [
        (champ, role)
        for champ, role in zip(champs, roles)
        if champ is not None and champ != ""
    ]
    if len(role_champs) < 2:
        return 0.5
    scores: List[float] = []
    for (c1, r1), (c2, r2) in combinations(role_champs, 2):
        key = tuple(sorted([f"{c1}_{r1}", f"{c2}_{r2}"]))
        if key in artifacts.pair_map:
            scores.append(artifacts.pair_map[key])
    return float(np.mean(scores)) if scores else 0.5


def build_features(df_pd: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    df = df_pd.copy()

    for col in ROLE_COLS:
        role = COL_TO_ROLE[col]
        df[f"{col}_wr"] = df[col].apply(lambda champ: get_wr(champ, role, artifacts))

    blue_role_names = [COL_TO_ROLE[col] for col in BLUE_ROLES]
    red_role_names = [COL_TO_ROLE[col] for col in RED_ROLES]

    df["blue_synergy"] = df[BLUE_ROLES].apply(
        lambda row: team_synergy_score(row.tolist(), blue_role_names, artifacts),
        axis=1,
    )
    df["red_synergy"] = df[RED_ROLES].apply(
        lambda row: team_synergy_score(row.tolist(), red_role_names, artifacts),
        axis=1,
    )

    df["blue_team_avg_wr"] = df[[f"{col}_wr" for col in BLUE_ROLES]].mean(axis=1)
    df["red_team_avg_wr"] = df[[f"{col}_wr" for col in RED_ROLES]].mean(axis=1)
    df["wr_diff"] = df["blue_team_avg_wr"] - df["red_team_avg_wr"]
    df["synergy_diff"] = df["blue_synergy"] - df["red_synergy"]

    df["picks_filled"] = df[ROLE_COLS].apply(
        lambda row: sum(1 for val in row if val not in (None, "")),
        axis=1,
    )

    cat_cols = ROLE_COLS + ["patch", "league", "blue_team", "red_team"]

    for col in cat_cols:
        role = COL_TO_ROLE.get(col)
        if role:
            raw = df[col].apply(
                lambda champ: f"{champ}_{role}" if champ not in (None, "") else f"UNKNOWN_{role}",
            )
        else:
            raw = df[col].fillna("UNKNOWN").astype(str)

        encoder = artifacts.encoders[col]

        def safe_encode(value: str) -> int:
            if value in encoder.classes_:
                return int(encoder.transform([value])[0])
            if "UNKNOWN" in encoder.classes_:
                return int(encoder.transform(["UNKNOWN"])[0])
            unknown_role = f"UNKNOWN_{role}" if role else "UNKNOWN"
            if unknown_role in encoder.classes_:
                return int(encoder.transform([unknown_role])[0])
            return -1

        df[col + "_enc"] = raw.apply(safe_encode)

    return df


def predict_draft(
    picks: Dict[str, str | None],
    artifacts: ModelArtifacts,
    patch: str = "UNKNOWN",
    league: str = "UNKNOWN",
    blue_team: str = "UNKNOWN",
    red_team: str = "UNKNOWN",
) -> Dict[str, float | int | str]:
    row: Dict[str, Any] = {col: picks.get(col) for col in ROLE_COLS}
    row.update(
        {
            "patch": patch,
            "league": league,
            "blue_team": blue_team,
            "red_team": red_team,
        }
    )

    df = pd.DataFrame([row])
    df = build_features(df, artifacts)

    proba = float(artifacts.model.predict_proba(df[artifacts.feature_cols].astype(float))[:, 1][0])
    proba = max(0.25, min(0.75, proba))

    filled = int(df["picks_filled"].iloc[0])
    confidence_weight = filled / 10.0
    proba = 0.5 + (proba - 0.5) * confidence_weight

    if filled <= 3:
        confidence = "low"
    elif filled <= 7:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "blue_win_prob": proba,
        "picks_filled": filled,
        "confidence": confidence,
    }
