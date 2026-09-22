import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.data.loader import load_raw
from src.data.preprocessing import clean_frame, split_xy
from src.features.engineering import engineer_features
from src.utils.config import settings


def run_error_analysis():
    df = engineer_features(clean_frame(load_raw()))

    X, y = split_xy(df)

    _, Xte, _, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=settings.random_state,
    )

    model = joblib.load(settings.model_dir / "churn_model.joblib")

    # Model predictions
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)

    # Build error-analysis dataframe
    result = Xte.copy()

    result["actual_churn"] = yte.values
    result["predicted_churn"] = pred
    result["probability"] = p

    # Default classification
    result["error_type"] = "correct"

    # False positives
    result.loc[
        (result["actual_churn"] == 0)
        & (result["predicted_churn"] == 1),
        "error_type",
    ] = "false_positive"

    # False negatives
    result.loc[
        (result["actual_churn"] == 1)
        & (result["predicted_churn"] == 0),
        "error_type",
    ] = "false_negative"

    # Borderline probability cases
    result["borderline_probability"] = (p >= 0.4) & (p <= 0.6)

    return result


def main():
    out = settings.root / "data/processed/error_analysis.csv"

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_error_analysis().to_csv(
        out,
        index=False,
    )

    print(out)


if __name__ == "__main__":
    main()