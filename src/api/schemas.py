REQUIRED_FIELDS=["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"]

def validate_payload(payload):
    missing=[f for f in REQUIRED_FIELDS if f not in payload]
    if missing: raise ValueError(f"Missing required fields: {missing}")
    return payload
