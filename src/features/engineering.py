import numpy as np
import pandas as pd

def engineer_features(df:pd.DataFrame)->pd.DataFrame:
    x=df.copy()
    if "TotalCharges" in x and "tenure" in x:
        x["charges_per_tenure_month"]=x["TotalCharges"]/(x["tenure"].clip(lower=1))
    if "MonthlyCharges" in x and "tenure" in x:
        x["estimated_tenure_value"]=x["MonthlyCharges"]*x["tenure"]
    if "tenure" in x:
        x["tenure_bucket"]=pd.cut(x["tenure"],[-1,6,12,24,48,72],labels=["0-6","7-12","13-24","25-48","49-72"])
    service_cols=[c for c in ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"] if c in x]
    if service_cols:
        x["active_service_count"]=sum((x[c].astype(str)=="Yes").astype(int) for c in service_cols)
    if "PaymentMethod" in x:
        x["electronic_payment"]=x["PaymentMethod"].astype(str).str.contains("Electronic",case=False,na=False).astype(int)
    return x.replace([np.inf,-np.inf],np.nan)
