import sqlite3
from pathlib import Path
from datetime import datetime,timezone
from src.utils.config import settings

def connect():
    settings.database_path.parent.mkdir(parents=True,exist_ok=True); return sqlite3.connect(settings.database_path)

def init_db():
    with connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS prediction_history(prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT NOT NULL,customer_id TEXT,prediction INTEGER NOT NULL,probability REAL NOT NULL,risk_level TEXT NOT NULL,model_version TEXT NOT NULL)""")

def save_prediction(customer_id,prediction,probability,risk_level,model_version):
    with connect() as c:
        c.execute("INSERT INTO prediction_history(timestamp,customer_id,prediction,probability,risk_level,model_version) VALUES(?,?,?,?,?,?)",(datetime.now(timezone.utc).isoformat(),customer_id,prediction,probability,risk_level,model_version))

def stats():
    init_db()
    with connect() as c:
        count=c.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
        avg=c.execute("SELECT AVG(probability) FROM prediction_history").fetchone()[0]
    return {"prediction_count":count,"average_churn_probability":avg}
