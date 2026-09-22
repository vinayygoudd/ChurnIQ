import pandas as pd
from sklearn.model_selection import train_test_split,cross_validate,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame,split_xy,build_preprocessor
from src.features.engineering import engineer_features
from src.utils.config import settings

def compare():
    df=engineer_features(clean_frame(load_raw())); X,y=split_xy(df); Xtr,_,ytr,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=settings.random_state); pre,_,_=build_preprocessor(X); cv=StratifiedKFold(5,shuffle=True,random_state=settings.random_state)
    rows=[]
    for label,weight in [("unweighted",None),("class_weight_balanced","balanced")]:
        pipe=Pipeline([("prep",pre),("model",LogisticRegression(max_iter=2000,class_weight=weight,random_state=settings.random_state))]); s=cross_validate(pipe,Xtr,ytr,cv=cv,scoring=["precision","recall","f1","roc_auc"]); rows.append({"strategy":label,"precision":s["test_precision"].mean(),"recall":s["test_recall"].mean(),"f1":s["test_f1"].mean(),"roc_auc":s["test_roc_auc"].mean()})
    return pd.DataFrame(rows)

def main():
    out=settings.root/"experiments/imbalance_comparison.csv"; out.parent.mkdir(parents=True,exist_ok=True); compare().to_csv(out,index=False); print(out)
if __name__=="__main__": main()
