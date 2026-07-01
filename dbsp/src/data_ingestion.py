"""
data_ingestion.py
=================
PHASE 2: Data Ingestion & Core Pipeline Engineering.

Builds the raw dataset of dead-ball / restart events. Two modes:

1. REAL MODE     -> pulls events from StatsBomb Open Data via `statsbombpy`.
2. SYNTHETIC MODE -> generates a realistic, signal-rich dataset locally with
                    no internet needed (so the app always runs).

Both modes return a DataFrame with IDENTICAL columns.

This version models a FULL set of dead-ball situations:
    corner, free_kick_wide, free_kick_central, penalty, throw_in,
    goal_kick, kick_off, play_restart
and a richer set of attacking tactical triggers (including a dedicated
second-ball setup).

Author: Project Team  |  Module: MIS41420 Sports & Performance Analytics
"""

from __future__ import annotations
import numpy as np
import pandas as pd

# All eight dead-ball situation types the tool supports.
SET_PIECE_TYPES = [
    "corner", "free_kick_wide", "free_kick_central", "penalty",
    "throw_in", "goal_kick", "kick_off", "play_restart",
]

# Nine attacking tactical triggers (boolean 0/1).
TRIGGER_COLS = [
    "near_post_decoy", "gk_screen", "blocker_action", "taker_foot_matches",
    "runner_from_deep", "overload_far_post", "second_wave_runner",
    "quick_delivery", "dummy_run",
]

RAW_COLUMNS = [
    "set_piece_type",       # one of SET_PIECE_TYPES
    "delivery_type",        # inswinger | outswinger | straight | short | long | driven | lofted
    "delivery_zone",        # near_post | central | far_post | penalty_spot | edge_box | six_yard | wide_channel
    "defensive_scheme",     # High Zonal | Deep Zonal | Man-to-Man | Hybrid
    "num_defenders_box",    # int
    "num_attackers_box",    # int
    "aerial_threat",        # 1-10
    "second_ball_runners",  # 0-4 attackers stationed at the edge for second balls
    *TRIGGER_COLS,
    "xg",                   # TARGET A (regression)
    "second_ball_won",      # TARGET B (classification)
]

# Type-specific baseline xG (everything else nudges around this).
TYPE_BASE_XG = {
    "penalty": 0.76, "free_kick_central": 0.07, "corner": 0.04,
    "free_kick_wide": 0.045, "throw_in": 0.02, "goal_kick": 0.012,
    "kick_off": 0.010, "play_restart": 0.02,
}
# Restart types where the attacking team naturally retains possession more often.
SHORT_RESTART_TYPES = {"throw_in", "kick_off", "goal_kick", "play_restart"}


# ---------------------------------------------------------------------------
# MODE 1 — REAL STATSBOMB DATA (optional; needs internet + statsbombpy)
# ---------------------------------------------------------------------------
def load_statsbomb_set_pieces(max_matches: int | None = 60) -> pd.DataFrame:
    from statsbombpy import sb  # lazy import

    comps = sb.competitions()
    frames, used = [], 0
    pattern_map = {
        "From Corner": "corner", "From Free Kick": "free_kick_wide",
        "From Throw In": "throw_in", "From Goal Kick": "goal_kick",
        "From Kick Off": "kick_off", "From Keeper": "play_restart",
    }
    for _, comp in comps.iterrows():
        try:
            matches = sb.matches(competition_id=comp["competition_id"],
                                 season_id=comp["season_id"])
        except Exception:
            continue
        for match_id in matches["match_id"].tolist():
            try:
                events = sb.events(match_id=match_id)
            except Exception:
                continue
            if "play_pattern" not in events.columns:
                continue
            sp = events[events["play_pattern"].isin(pattern_map.keys())]
            if sp.empty:
                continue
            frames.append(_statsbomb_rows_to_raw(sp, pattern_map))
            used += 1
            if max_matches and used >= max_matches:
                return _finalise(pd.concat(frames, ignore_index=True))
    if not frames:
        raise RuntimeError("No StatsBomb set-piece events retrieved.")
    return _finalise(pd.concat(frames, ignore_index=True))


def _statsbomb_rows_to_raw(sp: pd.DataFrame, pattern_map: dict) -> pd.DataFrame:
    rows = []
    for _, e in sp.iterrows():
        loc = e.get("location") or [None, None]
        y = loc[1] if isinstance(loc, (list, tuple)) and len(loc) > 1 else np.nan
        zone = ("near_post" if (pd.notna(y) and y < 30)
                else "far_post" if (pd.notna(y) and y > 50) else "central")
        shot = e.get("shot")
        xg = shot.get("statsbomb_xg", np.nan) if isinstance(shot, dict) else np.nan
        row = {
            "set_piece_type": pattern_map.get(str(e.get("play_pattern")), "play_restart"),
            "delivery_type": "inswinger", "delivery_zone": zone,
            "defensive_scheme": "Hybrid", "num_defenders_box": 8,
            "num_attackers_box": 5, "aerial_threat": 6, "second_ball_runners": 2,
            "xg": xg, "second_ball_won": np.nan,
        }
        for t in TRIGGER_COLS:
            row[t] = 0
        rows.append(row)
    df = pd.DataFrame(rows)
    df["xg"] = df["xg"].fillna(0.04)
    df["second_ball_won"] = np.random.binomial(1, 0.45, len(df))
    return df


# ---------------------------------------------------------------------------
# MODE 2 — SYNTHETIC DATA (always available)
# ---------------------------------------------------------------------------
def generate_synthetic_set_pieces(n: int = 9000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    set_piece_type = rng.choice(
        SET_PIECE_TYPES, n,
        p=[0.30, 0.14, 0.10, 0.04, 0.16, 0.12, 0.06, 0.08])
    delivery_type = rng.choice(
        ["inswinger", "outswinger", "straight", "short", "long", "driven", "lofted"],
        n, p=[0.20, 0.15, 0.15, 0.15, 0.12, 0.13, 0.10])
    delivery_zone = rng.choice(
        ["near_post", "central", "far_post", "penalty_spot", "edge_box",
         "six_yard", "wide_channel"],
        n, p=[0.18, 0.20, 0.15, 0.12, 0.15, 0.10, 0.10])
    defensive_scheme = rng.choice(
        ["High Zonal", "Deep Zonal", "Man-to-Man", "Hybrid"], n)
    num_defenders_box = rng.integers(3, 12, n)
    num_attackers_box = rng.integers(1, 8, n)
    aerial_threat = rng.integers(1, 11, n)
    second_ball_runners = rng.integers(0, 5, n)

    trig = {c: rng.binomial(1, p, n) for c, p in zip(
        TRIGGER_COLS,
        [0.35, 0.25, 0.30, 0.60, 0.40, 0.30, 0.35, 0.30, 0.25])}

    # ---- xG target ----
    zone_xg = pd.Series(delivery_zone).map({
        "near_post": 0.030, "central": 0.050, "far_post": 0.025,
        "penalty_spot": 0.060, "edge_box": 0.015, "six_yard": 0.055,
        "wide_channel": 0.005}).to_numpy()
    scheme_xg = pd.Series(defensive_scheme).map({
        "High Zonal": 0.012, "Deep Zonal": -0.004,
        "Man-to-Man": 0.018, "Hybrid": 0.006}).to_numpy()
    delivery_xg = pd.Series(delivery_type).map({
        "inswinger": 0.015, "outswinger": 0.008, "straight": 0.004,
        "short": -0.006, "long": 0.002, "driven": 0.012, "lofted": 0.006}).to_numpy()
    base = pd.Series(set_piece_type).map(TYPE_BASE_XG).to_numpy()

    xg = (
        base + zone_xg + scheme_xg + delivery_xg
        + 0.004 * aerial_threat
        + 0.015 * trig["near_post_decoy"] + 0.020 * trig["gk_screen"]
        + 0.012 * trig["blocker_action"] + 0.010 * trig["taker_foot_matches"]
        + 0.012 * trig["runner_from_deep"] + 0.010 * trig["overload_far_post"]
        + 0.008 * trig["dummy_run"] + 0.006 * trig["quick_delivery"]
        - 0.0045 * (num_defenders_box - num_attackers_box)
        + rng.normal(0, 0.02, n)
    )
    # Penalties are dominated by their base conversion rate.
    is_pen = set_piece_type == "penalty"
    xg = np.where(is_pen,
                  np.clip(0.76 + rng.normal(0, 0.03, n), 0.60, 0.92),
                  np.clip(xg, 0.005, 0.50))

    # ---- second-ball retention target ----
    z = (
        -0.40
        + 0.22 * num_attackers_box - 0.16 * num_defenders_box
        + 0.50 * second_ball_runners
        + 0.45 * trig["second_wave_runner"]
        + 0.40 * trig["blocker_action"] + 0.35 * trig["quick_delivery"]
        + 0.30 * trig["runner_from_deep"]
        + 0.50 * (pd.Series(delivery_type) == "short").to_numpy()
        + 0.30 * (pd.Series(delivery_type) == "long").to_numpy()
        + 0.40 * (pd.Series(delivery_zone) == "edge_box").to_numpy()
        + 0.25 * (pd.Series(delivery_zone) == "wide_channel").to_numpy()
        + 0.05 * aerial_threat
        + 0.30 * pd.Series(set_piece_type).isin(SHORT_RESTART_TYPES).to_numpy()
        - 0.50 * is_pen
        + rng.normal(0, 0.6, n)
    )
    second_ball_won = rng.binomial(1, 1 / (1 + np.exp(-z)))

    data = {
        "set_piece_type": set_piece_type, "delivery_type": delivery_type,
        "delivery_zone": delivery_zone, "defensive_scheme": defensive_scheme,
        "num_defenders_box": num_defenders_box, "num_attackers_box": num_attackers_box,
        "aerial_threat": aerial_threat, "second_ball_runners": second_ball_runners,
        "xg": np.round(xg, 4), "second_ball_won": second_ball_won,
    }
    data.update(trig)
    return _finalise(pd.DataFrame(data))


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[RAW_COLUMNS].dropna(subset=["xg", "second_ball_won"]).reset_index(drop=True)
    df["second_ball_won"] = df["second_ball_won"].astype(int)
    for t in TRIGGER_COLS:
        df[t] = df[t].astype(int)
    return df


def load_dataset(prefer_real: bool = False, max_matches: int | None = 60) -> pd.DataFrame:
    if prefer_real:
        try:
            return load_statsbomb_set_pieces(max_matches=max_matches)
        except Exception as exc:  # pragma: no cover
            print(f"[data_ingestion] StatsBomb load failed ({exc}); using synthetic.")
    return generate_synthetic_set_pieces()


if __name__ == "__main__":
    d = load_dataset()
    print(d.head())
    print("\nRows:", len(d), "| Types:", d.set_piece_type.nunique())
    print(d.groupby("set_piece_type")["xg"].mean().round(3))
