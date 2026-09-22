import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame, split_xy
from src.features.engineering import engineer_features
from src.utils.config import settings

def run_selection():
    df=engineer_features(clean_frame(load_raw()))
    X,y=split_xy(df)
    Xn=pd.get_dummies(X,drop_first=False).apply(pd.to_numeric,errors="coerce").fillna(0)
    Xtr,_,ytr,_=train_test_split(Xn,y,test_size=.2,stratify=y,random_state=settings.random_state)
    scores=mutual_info_classif(Xtr,ytr,random_state=settings.random_state)
    return pd.DataFrame({"feature":Xn.columns,"mutual_information":scores}).sort_values("mutual_information",ascending=False)

def main():
    out=settings.root/"data/processed/feature_selection.csv"; out.parent.mkdir(parents=True,exist_ok=True); run_selection().to_csv(out,index=False); print(out)
if __name__=="__main__": main()
