"""
Demonstrates the phantom champion + synergy_diff cascade that causes wild win rate swings.
Run from: c:\Dev\group-projects\WIF3009-project\backend
"""
import os, sys, pickle, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from itertools import combinations

BASE_DIR = r"c:\Dev\group-projects\WIF3009-project\backend\data"

with open(os.path.join(BASE_DIR, "calibrated_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(BASE_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(BASE_DIR, "feature_cols.pkl"), "rb") as f:
    feature_cols = pickle.load(f)

import polars as pl
wr_df = pl.read_parquet(os.path.join(BASE_DIR, "Parquets", "champ_winrates.parquet"))
syn_df = pl.read_parquet(os.path.join(BASE_DIR, "Parquets", "champ_synergies.parquet"))

wr_map = dict(zip(wr_df["champion"].to_list(), wr_df["win_rate"].to_list()))
global_avg_wr = wr_df["win_rate"].mean()
pair_map = {}
for row in syn_df.to_dicts():
    key = tuple(sorted([row["champion"], row["champ2"]]))
    pair_map[key] = row["pair_win_rate"]

role_cols = ["blue_top","blue_jng","blue_mid","blue_bot","blue_sup",
             "red_top","red_jng","red_mid","red_bot","red_sup"]

def team_synergy_score(champs):
    valid = [c for c in champs if pd.notna(c) and c != ""]
    scores = []
    for c1, c2 in combinations(valid, 2):
        key = tuple(sorted([c1, c2]))
        if key in pair_map:
            scores.append(pair_map[key])
    return np.mean(scores) if scores else 0.5

def build_row_current(blue_slots_champs, red_slots_champs):
    """Current (buggy) behaviour: empty slots use classes_[0] as phantom."""
    role_keys = ["top","jng","mid","bot","sup"]
    row = {}
    for i, rk in enumerate(role_keys):
        col = f"blue_{rk}"
        row[col] = blue_slots_champs[i] if blue_slots_champs[i] else encoders[col].classes_[0]
    for i, rk in enumerate(role_keys):
        col = f"red_{rk}"
        row[col] = red_slots_champs[i] if red_slots_champs[i] else encoders[col].classes_[0]

    row["patch"] = "16.01"
    row["league"] = "LCK"
    row["blue_team"] = encoders["blue_team"].classes_[0]
    row["red_team"] = encoders["red_team"].classes_[0]

    for col in role_cols:
        row[f"{col}_wr"] = wr_map.get(row[col], global_avg_wr)

    blue_roles = [f"blue_{r}" for r in role_keys]
    red_roles = [f"red_{r}" for r in role_keys]
    row["blue_synergy"] = team_synergy_score([row[r] for r in blue_roles])
    row["red_synergy"] = team_synergy_score([row[r] for r in red_roles])
    row["blue_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in blue_roles])
    row["red_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in red_roles])
    row["wr_diff"] = row["blue_team_avg_wr"] - row["red_team_avg_wr"]
    row["synergy_diff"] = row["blue_synergy"] - row["red_synergy"]

    def encode_val(enc_name, val):
        le = encoders[enc_name]
        if val in le.classes_: return le.transform([val])[0]
        return le.transform([le.classes_[0]])[0]

    for col in role_cols:
        row[f"{col}_enc"] = encode_val(col, row[col])
    row["patch_enc"] = encode_val("patch", row["patch"])
    row["league_enc"] = encode_val("league", row["league"])
    row["blue_team_enc"] = encode_val("blue_team", row["blue_team"])
    row["red_team_enc"] = encode_val("red_team", row["red_team"])
    return row

def build_row_fixed(blue_slots_champs, red_slots_champs):
    """Fixed behaviour: empty slots use global_avg_wr and median enc, excluded from synergy."""
    role_keys = ["top","jng","mid","bot","sup"]
    EMPTY = "__EMPTY__"
    row = {}
    for i, rk in enumerate(role_keys):
        col = f"blue_{rk}"
        row[col] = blue_slots_champs[i] if blue_slots_champs[i] else EMPTY
    for i, rk in enumerate(role_keys):
        col = f"red_{rk}"
        row[col] = red_slots_champs[i] if red_slots_champs[i] else EMPTY

    row["patch"] = "16.01"
    row["league"] = "LCK"

    for col in role_cols:
        c = row[col]
        row[f"{col}_wr"] = wr_map.get(c, global_avg_wr)  # EMPTY -> global_avg_wr

    blue_roles = [f"blue_{r}" for r in role_keys]
    red_roles = [f"red_{r}" for r in role_keys]
    # EMPTY is excluded from synergy because it won't be in pair_map
    row["blue_synergy"] = team_synergy_score([row[r] for r in blue_roles if row[r] != EMPTY])
    row["red_synergy"] = team_synergy_score([row[r] for r in red_roles if row[r] != EMPTY])
    row["blue_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in blue_roles])
    row["red_team_avg_wr"] = np.mean([row[f"{r}_wr"] for r in red_roles])
    row["wr_diff"] = row["blue_team_avg_wr"] - row["red_team_avg_wr"]
    row["synergy_diff"] = row["blue_synergy"] - row["red_synergy"]

    def encode_val_fixed(enc_name, val):
        if val == EMPTY:
            return len(encoders[enc_name].classes_) // 2  # median
        le = encoders[enc_name]
        if val in le.classes_: return le.transform([val])[0]
        return le.transform([le.classes_[0]])[0]

    for col in role_cols:
        row[f"{col}_enc"] = encode_val_fixed(col, row[col])
    row["patch_enc"] = encode_val_fixed("patch", row["patch"])
    row["league_enc"] = encode_val_fixed("league", row["league"])
    row["blue_team_enc"] = len(encoders["blue_team"].classes_) // 2
    row["red_team_enc"] = len(encoders["red_team"].classes_) // 2
    return row

def predict(row):
    X = pd.DataFrame([row])
    X_feat = X[feature_cols].copy()
    cat_cols = [c + "_enc" for c in role_cols] + ["patch_enc", "league_enc"]
    for c in cat_cols:
        X_feat[c] = X_feat[c].astype("category")
    probs = model.predict_proba(X_feat)
    return float(probs[0][1]) * 100

print("=" * 70)
print("DEMONSTRATING WIN RATE FLUCTUATION BUG")
print("=" * 70)
print("\nScenario: Blue picks Ahri for MID, one pick at a time.")
print("Red has no picks yet. Simulating what happens across 5 blue picks.\n")

# Simulate: picks come in one at a time
pick_sequence = [
    (["Ahri",   None,       None,       None,     None],   [None, None, None, None, None]),
    (["Ahri",   "Jinx",     None,       None,     None],   [None, None, None, None, None]),
    (["Ahri",   "Jinx",     "Leona",    None,     None],   [None, None, None, None, None]),
    (["Ahri",   "Jinx",     "Leona",    "Vi",     None],   [None, None, None, None, None]),
    (["Ahri",   "Jinx",     "Leona",    "Vi",     "Ornn"], [None, None, None, None, None]),
]

print(f"{'Blue picks':<45} {'CURRENT (buggy)':>16} {'FIXED':>10} {'Synergy_diff (curr)':>20}")
print("-" * 95)

for blue, red in pick_sequence:
    filled = [c for c in blue if c]
    label = ", ".join(filled) if filled else "(none)"
    
    row_curr = build_row_current(blue, red)
    row_fixed = build_row_fixed(blue, red)
    
    wr_curr = predict(row_curr)
    wr_fixed = predict(row_fixed)
    syn_diff = row_curr["synergy_diff"]
    
    print(f"  Blue: {label:<40} {wr_curr:>14.1f}%  {wr_fixed:>8.1f}%  {syn_diff:>18.4f}")

print("\n")
print("=" * 70)
print("SHOWING SYNERGY POLLUTION FROM PHANTOM CHAMPIONS")
print("=" * 70)
print("\nPhantom champions injected for empty slots (classes_[0] per encoder):")
for rk in ["top","jng","mid","bot","sup"]:
    b = encoders[f"blue_{rk}"].classes_[0]
    r = encoders[f"red_{rk}"].classes_[0]
    print(f"  blue_{rk} -> '{b}'   |   red_{rk} -> '{r}'")

print("\nSynergy between blue phantom champions (these pollute synergy_diff):")
phantoms = [encoders[f"blue_{rk}"].classes_[0] for rk in ["top","jng","mid","bot","sup"]]
for c1, c2 in combinations(phantoms, 2):
    key = tuple(sorted([c1, c2]))
    s = pair_map.get(key, None)
    if s is not None:
        print(f"  {c1} + {c2}: pair_win_rate = {s:.4f}")
    else:
        print(f"  {c1} + {c2}: NOT IN pair_map")

print("\nDone.")
