
import joblib
import pandas as pd

from src.features.engineering import engineer_features
from src.utils.config import settings


def risk_level(p: float) -> str:
    if p < settings.low_risk_threshold:
        return "Low"
    if p < settings.high_risk_threshold:
        return "Medium"
    return "High"


def load_model():
    return joblib.load(settings.model_dir / "churn_model.joblib")


def predict(payload: dict) -> dict:
    pipe = load_model()

    x = pd.DataFrame([payload])

    # Apply the same feature engineering used during training
    x = engineer_features(x)

    p = float(pipe.predict_proba(x)[:, 1][0])

    return {
        "prediction": int(p >= 0.5),
        "probability": p,
        "risk_level": risk_level(p),
        "model_version": settings.model_version,
    }
