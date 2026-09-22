import pandas as pd
from sklearn.model_selection import train_test_split,RandomizedSearchCV,StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame,split_xy,build_preprocessor
from src.features.engineering import engineer_features
from src.utils.config import settings

def main():
    df=engineer_features(clean_frame(load_raw())); X,y=split_xy(df); Xtr,_,ytr,_=train_test_split(X,y,test_size=.2,stratify=y,random_state=settings.random_state); pre,_,_=build_preprocessor(X)
    pipe=Pipeline([("prep",pre),("model",LogisticRegression(max_iter=3000,class_weight="balanced",random_state=settings.random_state))])
    params={"model__C":[.01,.05,.1,.25,.5,1,2,5,10],"model__solver":["liblinear","lbfgs"]}
    search=RandomizedSearchCV(pipe,params,n_iter=10,scoring="f1",cv=StratifiedKFold(5,shuffle=True,random_state=42),random_state=42,n_jobs=-1,refit=True); search.fit(Xtr,ytr)
    out=settings.root/"experiments/tuning_results.csv"; pd.DataFrame(search.cv_results_).to_csv(out,index=False); print(search.best_params_,search.best_score_)
if __name__=="__main__": main()
