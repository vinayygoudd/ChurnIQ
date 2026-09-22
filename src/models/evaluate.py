import json
import joblib,pandas as pd,matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay,roc_curve,precision_recall_curve,auc
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame,split_xy
from src.features.engineering import engineer_features
from src.utils.config import settings

def main():
    pipe=joblib.load(settings.model_dir/"churn_model.joblib"); df=engineer_features(clean_frame(load_raw())); X,y=split_xy(df); _,Xte,_,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=settings.random_state); prob=pipe.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int); out=settings.root/"artifacts/evaluation"; out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(); ConfusionMatrixDisplay.from_predictions(yte,pred,ax=ax); fig.tight_layout(); fig.savefig(out/"confusion_matrix.png",dpi=160); plt.close(fig)
    fpr,tpr,_=roc_curve(yte,prob); fig,ax=plt.subplots(); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],linestyle="--"); ax.set(xlabel="False Positive Rate",ylabel="True Positive Rate",title="ROC Curve"); fig.tight_layout(); fig.savefig(out/"roc_curve.png",dpi=160); plt.close(fig)
    p,r,_=precision_recall_curve(yte,prob); fig,ax=plt.subplots(); ax.plot(r,p); ax.set(xlabel="Recall",ylabel="Precision",title="Precision-Recall Curve"); fig.tight_layout(); fig.savefig(out/"precision_recall_curve.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
