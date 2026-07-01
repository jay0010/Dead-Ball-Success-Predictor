"""
feature_engineering.py
======================
PHASE 3: Feature Engineering & Preprocessing.

Turns raw set-piece rows into a numeric feature matrix. The SAME function is
used at training time AND inside the Streamlit app, guaranteeing the coach's
live inputs are encoded exactly like the training data.

Engineered features:
  * box_overload        = attackers in box - defenders in box
  * routine_complexity  = number of attacking movement triggers combined
  * second_ball_index   = strength of the attacking team's second-ball setup
                          (edge runners + a dedicated second-wave crasher)

Author: Project Team  |  Module: MIS41420 Sports & Performance Analytics
"""

from __future__ import annotations
import pandas as pd
from data_ingestion import TRIGGER_COLS  # single source of truth

CATEGORICAL_LEVELS = {
    "set_piece_type": ["corner", "free_kick_wide", "free_kick_central", "penalty",
                       "throw_in", "goal_kick", "kick_off", "play_restart"],
    "delivery_type": ["inswinger", "outswinger", "straight", "short",
                      "long", "driven", "lofted"],
    "delivery_zone": ["near_post", "central", "far_post", "penalty_spot",
                      "edge_box", "six_yard", "wide_channel"],
    "defensive_scheme": ["High Zonal", "Deep Zonal", "Man-to-Man", "Hybrid"],
}

NUMERIC_COLS = ["num_defenders_box", "num_attackers_box", "aerial_threat",
                "second_ball_runners", *TRIGGER_COLS]

ENGINEERED_COLS = ["box_overload", "routine_complexity", "second_ball_index"]

# Triggers that represent attacking movement (used for routine_complexity).
MOVEMENT_TRIGGERS = ["near_post_decoy", "blocker_action", "runner_from_deep",
                     "overload_far_post", "dummy_run", "gk_screen"]

TARGET_REG = "xg"
TARGET_CLF = "second_ball_won"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["box_overload"] = out["num_attackers_box"] - out["num_defenders_box"]
    out["routine_complexity"] = out[MOVEMENT_TRIGGERS].sum(axis=1)
    out["second_ball_index"] = out["second_ball_runners"] + out["second_wave_runner"]
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_engineered_features(df)
    cat_frames = []
    for col, levels in CATEGORICAL_LEVELS.items():
        s = pd.Categorical(df[col], categories=levels)
        cat_frames.append(pd.get_dummies(s, prefix=col).astype(int))
    numeric = df[NUMERIC_COLS + ENGINEERED_COLS].reset_index(drop=True)
    return pd.concat([numeric] + [f.reset_index(drop=True) for f in cat_frames], axis=1)


def feature_columns(df_sample: pd.DataFrame) -> list[str]:
    return build_feature_matrix(df_sample).columns.tolist()


if __name__ == "__main__":
    from data_ingestion import load_dataset
    X = build_feature_matrix(load_dataset())
    print("Feature matrix shape:", X.shape)
    print("Columns:", list(X.columns))
