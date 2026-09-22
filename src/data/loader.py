from pathlib import Path
from urllib.request import urlopen
import pandas as pd
from src.utils.config import settings
from src.utils.logging import get_logger
logger=get_logger(__name__)
RAW=settings.root/"data/raw/Telco-Customer-Churn.csv"

def download_dataset(force=False)->Path:
    RAW.parent.mkdir(parents=True,exist_ok=True)
    if RAW.exists() and not force:
        return RAW
    logger.info("Downloading public IBM Telco churn dataset")
    with urlopen(settings.data_url, timeout=60) as r:
        RAW.write_bytes(r.read())
    return RAW

def load_raw()->pd.DataFrame:
    path=download_dataset()
    df=pd.read_csv(path)
    if "TotalCharges" in df: df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
    return df

def quality_report(df:pd.DataFrame)->pd.DataFrame:
    return pd.DataFrame({"dtype":df.dtypes.astype(str),"missing":df.isna().sum(),"unique":df.nunique(),"missing_pct":df.isna().mean()*100})

def main():
    df=load_raw(); out=settings.root/"data/processed/data_quality_report.csv"; out.parent.mkdir(parents=True,exist_ok=True); quality_report(df).to_csv(out)
    print(f"rows={len(df)}, columns={len(df.columns)}; report={out}")
if __name__=="__main__": main()
