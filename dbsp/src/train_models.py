"""
train_models.py
===============
PHASE 4: Advanced Predictive Modeling & Training.

Trains the two engines that power the Dead Ball Success Predictor:

  ENGINE A - Chance Quality Regressor   -> continuous xG. Metric: MSE/RMSE/R2.
  ENGINE B - Second-Ball Retention Clf  -> retention probability. Metric: ROC-AUC.

Both use a small cross-validated GridSearchCV to reduce overfitting. Trained
models + the exact feature-column order + a metrics summary are saved to
../models/ with joblib so the app loads them instantly.

Run:  python src/train_models.py
Author: Project Team  |  Module: MIS41420 Sports & Performance Analytics
"""

from __future__ import annotations
import json, os, sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score

sys.path.append(os.path.dirname(__file__))
from data_ingestion import load_dataset
from feature_engineering import build_feature_matrix, TARGET_REG, TARGET_CLF

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


def train_engine_a(Xtr, ytr, Xte, yte):
    grid = {"n_estimators": [150], "max_depth": [14, 18], "min_samples_leaf": [8]}
    search = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1),
                          grid, cv=4, scoring="neg_mean_squared_error", n_jobs=-1)
    search.fit(Xtr, ytr)
    model = search.best_estimator_
    preds = model.predict(Xte)
    mse = mean_squared_error(yte, preds)
    return model, {"best_params": search.best_params_, "mse": round(float(mse), 5),
                   "rmse": round(float(np.sqrt(mse)), 5),
                   "r2": round(float(r2_score(yte, preds)), 4)}


def train_engine_b(Xtr, ytr, Xte, yte):
    grid = {"n_estimators": [150], "max_depth": [12, 16], "min_samples_leaf": [8]}
    search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced"),
        grid, cv=4, scoring="roc_auc", n_jobs=-1)
    search.fit(Xtr, ytr)
    model = search.best_estimator_
    proba = model.predict_proba(Xte)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return model, {"best_params": search.best_params_,
                   "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
                   "accuracy": round(float(accuracy_score(yte, preds)), 4)}


def main(prefer_real: bool = False) -> None:
    print(">> Loading dataset ...")
    raw = load_dataset(prefer_real=prefer_real)
    raw.to_csv(DATA_DIR / "set_pieces.csv", index=False)
    print(f"   {len(raw)} rows, {raw.set_piece_type.nunique()} set-piece types")

    X = build_feature_matrix(raw)
    feature_cols = X.columns.tolist()
    Xtr, Xte, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, raw[TARGET_REG], raw[TARGET_CLF], test_size=0.2, random_state=42)

    print(">> Training Engine A (xG regressor) ...")
    engine_a, m_a = train_engine_a(Xtr, yr_tr, Xte, yr_te); print("   ", m_a)
    print(">> Training Engine B (retention classifier) ...")
    engine_b, m_b = train_engine_b(Xtr, yc_tr, Xte, yc_te); print("   ", m_b)

    joblib.dump(engine_a, MODELS_DIR / "engine_a_xg.joblib")
    joblib.dump(engine_b, MODELS_DIR / "engine_b_retention.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_columns.joblib")
    summary = {"engine_a_xg": m_a, "engine_b_retention": m_b,
               "n_rows": len(raw), "n_features": len(feature_cols),
               "data_source": "StatsBomb" if prefer_real else "synthetic"}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(">> Saved models + metrics to models/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(prefer_real=os.environ.get("PREFER_REAL") == "1")
