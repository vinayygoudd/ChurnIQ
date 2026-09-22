import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame
from src.features.engineering import engineer_features
from src.utils.config import settings

def main():
    df=engineer_features(clean_frame(load_raw())); out=settings.root/"artifacts/eda"; out.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="whitegrid")
    plots=[("churn_distribution",lambda ax:sns.countplot(data=df,x="Churn",ax=ax)),("contract_churn",lambda ax:sns.countplot(data=df,x="Contract",hue="Churn",ax=ax)),("tenure_churn",lambda ax:sns.boxplot(data=df,x="Churn",y="tenure",ax=ax)),("charges_churn",lambda ax:sns.boxplot(data=df,x="Churn",y="MonthlyCharges",ax=ax)),("payment_churn",lambda ax:sns.countplot(data=df,x="PaymentMethod",hue="Churn",ax=ax)),("support_churn",lambda ax:sns.boxplot(data=df,x="Churn",y="MonthlyCharges",ax=ax))]
    for name,fn in plots:
        fig,ax=plt.subplots(figsize=(9,5)); fn(ax); ax.set_title(name.replace("_"," ").title()); fig.tight_layout(); fig.savefig(out/f"{name}.png",dpi=160); plt.close(fig)
    num=df.select_dtypes("number"); fig,ax=plt.subplots(figsize=(10,7)); sns.heatmap(num.corr(),cmap="crest",ax=ax); fig.tight_layout(); fig.savefig(out/"correlation.png",dpi=160); plt.close(fig)
if __name__=="__main__": main()
