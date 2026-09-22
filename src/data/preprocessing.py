import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    # Normalize column names
    x.columns = [str(col).strip() for col in x.columns]

    # Clean customer identifier before deduplication
    if "customerID" in x.columns:
        x["customerID"] = (
            x["customerID"]
            .astype(str)
            .str.strip()
        )

        # One row per customer
        x = x.drop_duplicates(
            subset=["customerID"],
            keep="first"
        )
    else:
        # Fallback for datasets without a customer identifier
        x = x.drop_duplicates()

    x = x.reset_index(drop=True)

    # Convert TotalCharges to numeric
    if "TotalCharges" in x.columns:
        x["TotalCharges"] = pd.to_numeric(
            x["TotalCharges"],
            errors="coerce"
        )

    # Convert target
    if "Churn" in x.columns:
        x["Churn"] = x["Churn"].map({
            "Yes": 1,
            "No": 0
        })

    return x


def split_xy(df: pd.DataFrame):
    y = df["Churn"].astype(int)

    X = df.drop(
        columns=["Churn", "customerID"],
        errors="ignore"
    )

    return X, y


def build_preprocessor(X: pd.DataFrame):

    num = X.select_dtypes(
        include="number"
    ).columns.tolist()

    cat = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="median")
                        ),
                        (
                            "scale",
                            StandardScaler()
                        ),
                    ]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False
                            )
                        ),
                    ]
                ),
                cat,
            ),
        ],
        remainder="drop",
    )

    return preprocessor, num, cat