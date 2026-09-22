import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    model_dir: Path = ROOT / os.getenv("MODEL_DIR", "models")
    database_path: Path = ROOT / os.getenv("DATABASE_PATH", "models/predictions.db")
    model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
    low_risk_threshold: float = float(os.getenv("LOW_RISK_THRESHOLD", "0.35"))
    high_risk_threshold: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.65"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    data_url: str = os.getenv("DATA_URL", "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
settings = Settings()
settings.model_dir.mkdir(parents=True, exist_ok=True)
