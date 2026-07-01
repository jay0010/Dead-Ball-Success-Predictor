"""
scenario.py
===========
Pure-Python helpers used by the Streamlit app (no Streamlit import here, so it
stays unit-testable):

  * random_scenario()        -> a realistic, randomly populated routine so the
                                app opens straight onto live data (no start
                                screen, never empty).
  * second_ball_drop_zone()  -> where the second ball is most likely to fall.
  * retention_levers()       -> which attacking second-ball levers are active in
                                this scenario, for the detailed retention panel.
  * routine_label()          -> a short human label for the current routine.

Author: Project Team  |  Module: MIS41420 Sports & Performance Analytics
"""

from __future__ import annotations
import random

from data_ingestion import SET_PIECE_TYPES, TRIGGER_COLS

DELIVERY_TYPES = ["inswinger", "outswinger", "straight", "short",
                  "long", "driven", "lofted"]
DELIVERY_ZONES = ["near_post", "central", "far_post", "penalty_spot",
                  "edge_box", "six_yard", "wide_channel"]
DEFENSIVE_SCHEMES = ["High Zonal", "Deep Zonal", "Man-to-Man", "Hybrid"]

# Friendly labels for the UI dropdowns.
PRETTY = {
    "free_kick_wide": "Free kick (wide)", "free_kick_central": "Free kick (central)",
    "throw_in": "Throw-in", "goal_kick": "Goal kick", "kick_off": "Kick-off",
    "play_restart": "Play restart", "near_post": "Near post", "far_post": "Far post",
    "penalty_spot": "Penalty spot", "edge_box": "Edge of box", "six_yard": "Six-yard box",
    "wide_channel": "Wide channel",
}
def pretty(v: str) -> str:
    return PRETTY.get(v, v.replace("_", " ").capitalize())


def random_scenario(seed: int | None = None) -> dict:
    """Return a complete, realistic routine as a dict of raw input fields."""
    rng = random.Random(seed)
    sp = rng.choices(
        SET_PIECE_TYPES,
        weights=[30, 14, 10, 6, 16, 12, 6, 8])[0]

    scenario = {
        "set_piece_type": sp,
        "delivery_type": rng.choice(DELIVERY_TYPES),
        "delivery_zone": rng.choice(DELIVERY_ZONES),
        "defensive_scheme": rng.choice(DEFENSIVE_SCHEMES),
        "num_attackers_box": rng.randint(2, 7),
        "num_defenders_box": rng.randint(4, 11),
        "aerial_threat": rng.randint(3, 9),
        "second_ball_runners": rng.randint(0, 4),
    }
    for t in TRIGGER_COLS:
        scenario[t] = int(rng.random() < 0.4)
    scenario["taker_foot_matches"] = int(rng.random() < 0.6)

    # Make penalties look like penalties.
    if sp == "penalty":
        scenario["delivery_zone"] = "penalty_spot"
        scenario["delivery_type"] = "driven"
    return scenario


def second_ball_drop_zone(delivery_zone: str) -> str:
    """Heuristic: where a cleared/loose ball is most likely to drop."""
    return {
        "near_post": "Central / penalty-spot area",
        "central": "Top of the box (edge)",
        "far_post": "Far-side edge of the box",
        "penalty_spot": "Edge of the box, centrally",
        "edge_box": "Top of the box — recycle wide",
        "six_yard": "Penalty spot / cleared long",
        "wide_channel": "Near touchline — recycle and switch",
    }.get(delivery_zone, "Edge of the box")


def retention_levers(scenario: dict) -> list[dict]:
    """
    Active vs inactive attacking second-ball levers, each with a rough weight
    (used to draw the retention-driver panel and to coach the recommendation).
    """
    s = scenario
    levers = [
        ("Edge runners stationed", s["second_ball_runners"] > 0,
         f"{s['second_ball_runners']} runner(s) at the edge", 0.50),
        ("Designated second-wave runner", bool(s["second_wave_runner"]),
         "A player crashes the box for loose balls", 0.45),
        ("Off-the-ball blocker", bool(s["blocker_action"]),
         "Frees a runner / wins the first contact", 0.40),
        ("Quick delivery", bool(s["quick_delivery"]),
         "Catches the defence before it is set", 0.35),
        ("Runner from deep", bool(s["runner_from_deep"]),
         "Late arrival attacks the second phase", 0.30),
        ("Short/long routine", s["delivery_type"] in ("short", "long"),
         "Manipulates distances to keep possession", 0.40),
        ("Edge / wide target zone", s["delivery_zone"] in ("edge_box", "wide_channel"),
         "Delivery lands where recovery is easier", 0.35),
    ]
    return [{"label": l, "active": bool(a), "detail": d, "weight": w}
            for (l, a, d, w) in levers]


def retention_advice(prob: float, scenario: dict) -> list[str]:
    """Concrete, coach-facing recommendations to improve second-ball control."""
    tips = []
    if scenario["second_ball_runners"] < 2:
        tips.append("Station at least 2 runners at the edge of the box to win second balls.")
    if not scenario["second_wave_runner"]:
        tips.append("Add a designated second-wave runner to crash loose deliveries.")
    if not scenario["blocker_action"]:
        tips.append("Use an off-the-ball block to free your first-contact runner.")
    if prob < 0.45 and scenario["delivery_type"] not in ("short", "long"):
        tips.append("Consider a worked short routine to keep possession if the box is crowded.")
    if not tips:
        tips.append("Strong second-ball setup — keep your rest-defence shape to counter the break.")
    return tips


def routine_label(scenario: dict) -> str:
    sp = pretty(scenario["set_piece_type"])
    return f"{sp} · {pretty(scenario['delivery_type'])} to {pretty(scenario['delivery_zone'])}"


if __name__ == "__main__":
    sc = random_scenario(7)
    print("Random scenario:", routine_label(sc))
    print("Drop zone:", second_ball_drop_zone(sc["delivery_zone"]))
    for lev in retention_levers(sc):
        print(("✓" if lev["active"] else "·"), lev["label"])
