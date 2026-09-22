import pandas as pd
from src.data.preprocessing import clean_frame
from src.features.engineering import engineer_features
from src.models.predict import risk_level

def test_clean_and_engineer():
    df=pd.DataFrame({"customerID":["a","a"],"tenure":[1,2],"TotalCharges":["10","20"],"MonthlyCharges":[10,10],"Churn":["Yes","No"]})
    out=engineer_features(clean_frame(df))
    assert len(out)==1 and "charges_per_tenure_month" in out.columns

def test_risk_thresholds():
    assert risk_level(0.1)=="Low" and risk_level(0.5)=="Medium" and risk_level(0.9)=="High"
