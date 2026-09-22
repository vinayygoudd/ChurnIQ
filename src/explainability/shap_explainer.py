
import json
import joblib
import pandas as pd

from src.data.loader import load_raw
from src.data.preprocessing import clean_frame, split_xy
from src.features.engineering import engineer_features
from src.utils.config import settings


def explain_row(payload: dict):
    try:
        import shap
    except ImportError:
        return {
            "available": False,
            "message": "Install SHAP to enable model explanations.",
        }

    try:
        # ---------------------------------------------------------
        # Load trained pipeline
        # ---------------------------------------------------------

        pipe = joblib.load(
            settings.model_dir / "churn_model.joblib"
        )

        prep = pipe.named_steps["prep"]
        model = pipe.named_steps["model"]

        # ---------------------------------------------------------
        # Prepare the customer row exactly as during training
        # ---------------------------------------------------------

        x = pd.DataFrame([payload])
        x = engineer_features(x)

        # Transform the customer row using the fitted preprocessor
        Xt = prep.transform(x)

        feature_names = prep.get_feature_names_out()

        # ---------------------------------------------------------
        # Build a representative background dataset
        # ---------------------------------------------------------

        raw = load_raw()
        clean = clean_frame(raw)
        engineered = engineer_features(clean)

        X_background, _ = split_xy(engineered)

        # Use the fitted preprocessing pipeline
        X_background_transformed = prep.transform(
            X_background
        )

        # Keep the background reasonably small
        # to avoid unnecessary SHAP computation.
        background_size = min(
            200,
            X_background_transformed.shape[0],
        )

        background = X_background_transformed[
            :background_size
        ]

        # ---------------------------------------------------------
        # Logistic Regression SHAP explanation
        # ---------------------------------------------------------

        if hasattr(model, "coef_"):

            explainer = shap.LinearExplainer(
                model,
                background,
            )

            shap_values = explainer(Xt)

            values = shap_values.values[0]

            # Binary Logistic Regression normally produces
            # one contribution vector.
            if values.ndim > 1:
                values = values[:, 0]

            # Get the strongest contributors
            order = sorted(
                range(len(values)),
                key=lambda i: abs(float(values[i])),
                reverse=True,
            )[:10]

            base_value = shap_values.base_values[0]

            if hasattr(base_value, "__len__"):
                base_value = base_value[0]

            return {
                "available": True,
                "method": "LinearExplainer",
                "base_value": float(base_value),
                "features": [
                    {
                        "feature": str(feature_names[i]),
                        "contribution": float(values[i]),
                    }
                    for i in order
                ],
            }

        # ---------------------------------------------------------
        # Fallback for compatible probability estimators
        # ---------------------------------------------------------

        if hasattr(model, "predict_proba"):

            explainer = shap.Explainer(
                model,
                background,
                feature_names=feature_names,
            )

            shap_values = explainer(Xt)
            values = shap_values.values[0]

            if values.ndim > 1:
                values = values[:, 0]

            order = sorted(
                range(len(values)),
                key=lambda i: abs(float(values[i])),
                reverse=True,
            )[:10]

            base_value = shap_values.base_values[0]

            if hasattr(base_value, "__len__"):
                base_value = base_value[0]

            return {
                "available": True,
                "method": "SHAP",
                "base_value": float(base_value),
                "features": [
                    {
                        "feature": str(feature_names[i]),
                        "contribution": float(values[i]),
                    }
                    for i in order
                ],
            }

        return {
            "available": False,
            "message": (
                "Estimator does not expose a compatible "
                "SHAP explanation interface."
            ),
        }

    except Exception as exc:

        return {
            "available": False,
            "message": (
                "SHAP explanation failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        }


def main():

    df = engineer_features(
        clean_frame(
            load_raw()
        )
    )

    X, _ = split_xy(df)

    result = explain_row(
        X.iloc[0].to_dict()
    )

    output_path = (
        settings.root
        / "artifacts"
        / "shap_example.json"
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )

    print(result)


if __name__ == "__main__":
    main()

