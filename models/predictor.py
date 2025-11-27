import os
import numpy as np
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_svm_model.joblib"
SCALER_PATH = BASE_DIR / "scaler (2).pkl"

# Expected feature order used by the SVM model
FEATURE_ORDER = [
    "age_at_heart_attack",
    "fractional_shortening",
    "epss",
    "lvdd",
    "wall_motion_index",
    "pericardial_effusion",
]

try:
    _model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"Could not load model from {MODEL_PATH}: {exc}")

try:
    _scaler = joblib.load(SCALER_PATH)
except Exception as exc:
    raise RuntimeError(f"Could not load scaler from {SCALER_PATH}: {exc}")


def predict_from_dict(values_dict):
    """Return prediction and probability from a dict of feature values."""
    try:
        values = [float(values_dict[name]) for name in FEATURE_ORDER]
    except KeyError as missing:
        raise ValueError(f"Missing required field: {missing}")
    except ValueError:
        raise ValueError("All fields must be numeric")

    X = np.array(values, dtype=float).reshape(1, -1)
    try:
        # Only apply scaler if it expects the same number of features
        if hasattr(_scaler, "n_features_in_") and _scaler.n_features_in_ != X.shape[1]:
            # Fall back to raw values when the scaler was trained with a different shape
            X_scaled = X
        else:
            X_scaled = _scaler.transform(X)
    except Exception as exc:
        raise ValueError(f"Error applying scaler: {exc}")

    pred = _model.predict(X_scaled)[0]

    proba = None
    if hasattr(_model, "predict_proba"):
        try:
            proba = float(_model.predict_proba(X_scaled)[0][1])
        except Exception:
            proba = None

    return {"prediction": int(pred), "probability": proba}
