import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_pipeline.joblib"
FEATURES_PATH = BASE_DIR / "feature_names.pkl"

print("\n=== LOADING PREDICTOR.PY ===")
print("Model path:", MODEL_PATH)
print("Feature names path:", FEATURES_PATH)

# Load pipeline (scaler + modelo)
try:
    _pipeline = joblib.load(MODEL_PATH)
    print("✔ Pipeline cargado correctamente.")
except Exception as exc:
    raise RuntimeError(f"❌ No se pudo cargar el pipeline desde {MODEL_PATH}: {exc}")

# Load correct feature order
try:
    FEATURE_ORDER = joblib.load(FEATURES_PATH)
    print("✔ FEATURES:", FEATURE_ORDER)
except Exception as exc:
    raise RuntimeError(f"❌ No se pudo cargar feature_names.pkl: {exc}")

print("=== PREDICTOR CARGADO COMPLETAMENTE ===\n")


def predict_from_dict(values_dict):
    """
    Recibe un dict con las 6 características del ecocardiograma.
    Usa el PIPELINE (scaler + modelo) y devuelve predicción + probabilidad.
    """

    # Convert input dict → ordered list
    try:
        values = [float(values_dict[name]) for name in FEATURE_ORDER]
    except KeyError as missing:
        raise ValueError(f"Missing required field: {missing}")
    except ValueError:
        raise ValueError("All fields must be numeric")

    # Convert to DataFrame with correct columns
    X = pd.DataFrame([values], columns=FEATURE_ORDER)

    # Predict directly using pipeline
    try:
        pred = int(_pipeline.predict(X)[0])

        # Probability (si el modelo lo permite)
        try:
            proba = float(_pipeline.predict_proba(X)[0][1])
        except:
            proba = None

        return {
            "prediction": pred,
            "probability": proba
        }

    except Exception as exc:
        raise ValueError(f"Error during prediction: {exc}")
