import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame
from src.utils.config import settings

def run_tests(df):
    rows=[]
    for col in ["MonthlyCharges","tenure","TotalCharges"]:
        a=df.loc[df.Churn==0,col].dropna(); b=df.loc[df.Churn==1,col].dropna()
        stat,p=mannwhitneyu(a,b,alternative="two-sided")
        rows.append({"variable":col,"test":"Mann-Whitney U","statistic":stat,"p_value":p})
    for col in ["Contract","PaymentMethod","InternetService"]:
        if col in df:
            table=pd.crosstab(df[col],df.Churn); stat,p,dof,_=chi2_contingency(table)
            rows.append({"variable":col,"test":"Chi-square","statistic":stat,"p_value":p,"dof":dof})
    return pd.DataFrame(rows)

def main():
    df=clean_frame(load_raw()); out=settings.root/"data/processed/statistical_tests.csv"; out.parent.mkdir(parents=True,exist_ok=True); run_tests(df).to_csv(out,index=False); print(out)
if __name__=="__main__": main()
