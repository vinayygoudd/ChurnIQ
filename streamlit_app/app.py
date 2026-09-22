
import sys
import json
from pathlib import Path

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.models.predict import predict
from src.data.loader import load_raw
from src.data.preprocessing import clean_frame
from src.database.database import init_db, stats
from src.explainability.shap_explainer import explain_row
from src.utils.config import settings


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="ChurnIQ",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Load external stylesheet
# ------------------------------------------------------------

def load_css():
    css_path = Path(__file__).parent / "style.css"

    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css()


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def metric_value(value, decimals=3):
    if value is None:
        return "—"

    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return "—"


def format_probability(value):
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def render_explanation(explanation):
    if not explanation.get("available"):
        st.info(
            explanation.get(
                "message",
                "Model explanation is currently unavailable.",
            )
        )
        return

    features = explanation.get("features", [])

    if not features:
        st.info("No feature-level explanation was returned.")
        return

    rows = []

    for item in features:
        contribution = item.get("contribution", 0)

        rows.append(
            {
                "Feature": item.get("feature", "Unknown"),
                "Contribution": float(contribution),
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "The values above are the contributions returned by the current SHAP implementation."
    )

    with st.expander("View raw explanation"):
        st.json(explanation)


# ------------------------------------------------------------
# Database initialization
# ------------------------------------------------------------

init_db()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            <div class="brand-name">ChurnIQ</div>
            <div class="brand-subtitle">
                Customer churn intelligence
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Workspace",
        [
            "Prediction",
            "Model Information",
            "Dataset Analysis",
        ],
    )

    st.markdown("---")

    st.caption(
        "Model-backed decision support. "
        "Use predictions alongside business context."
    )


# ============================================================
# PREDICTION
# ============================================================

if page == "Prediction":

    st.title("Customer churn prediction")

    st.markdown(
        """
        <div class="page-description">
            Estimate the likelihood of churn using customer account,
            service and billing information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Customer profile</div>',
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):

        # ----------------------------------------------------
        # Account
        # ----------------------------------------------------

        st.markdown(
            '<div class="form-section-title">Account</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )

            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
            )

        with col2:
            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"],
            )

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12,
                step=1,
            )

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year",
                ],
            )

        with col3:
            paperless = st.selectbox(
                "Paperless billing",
                ["Yes", "No"],
            )

            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

            customer_id = st.text_input(
                "Customer ID",
                placeholder="Optional",
            )

        st.markdown(
            '<div class="form-section-title">Services</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            phone = st.selectbox(
                "Phone service",
                ["Yes", "No"],
            )

            multiple = st.selectbox(
                "Multiple lines",
                [
                    "Yes",
                    "No",
                    "No phone service",
                ],
            )

            internet = st.selectbox(
                "Internet service",
                [
                    "DSL",
                    "Fiber optic",
                    "No",
                ],
            )

        with col2:
            security = st.selectbox(
                "Online security",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

            backup = st.selectbox(
                "Online backup",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

            device = st.selectbox(
                "Device protection",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

        with col3:
            tech = st.selectbox(
                "Tech support",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

            streaming_movies = st.selectbox(
                "Streaming movies",
                [
                    "Yes",
                    "No",
                    "No internet service",
                ],
            )

        st.markdown(
            '<div class="form-section-title">Billing</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            monthly = st.number_input(
                "Monthly charges",
                min_value=0.0,
                max_value=200.0,
                value=70.0,
                step=1.0,
            )

        with col2:
            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=840.0,
                step=10.0,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "Run churn assessment",
            type="primary",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    if submitted:

        payload = {
            "customer_id": customer_id or "UI-ANONYMOUS",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": tech,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        }

        try:

            result = predict(payload)

            st.markdown(
                '<div class="result-heading">Assessment</div>',
                unsafe_allow_html=True,
            )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:
                st.metric(
                    "Prediction",
                    "Churn" if result["prediction"] else "No churn",
                )

            with result_col2:
                st.metric(
                    "Churn probability",
                    format_probability(result["probability"]),
                )

            with result_col3:
                st.metric(
                    "Risk level",
                    result["risk_level"],
                )

            st.markdown(
                """
                <div class="assessment-note">
                    This result represents the model's estimated churn risk
                    for the supplied customer profile.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="section-title">Model explanation</div>',
                unsafe_allow_html=True,
            )

            explanation = explain_row(payload)

            render_explanation(explanation)

        except FileNotFoundError:

            st.error(
                "Model artifact not found. "
                "Run `python -m src.models.train` first."
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {type(exc).__name__}: {exc}"
            )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.title("Model information")

    st.markdown(
        """
        <div class="page-description">
            Training configuration, evaluation results and model activity.
        </div>
        """,
        unsafe_allow_html=True,
    )

    metadata_path = settings.model_dir / "metadata.json"

    if not metadata_path.exists():

        st.warning(
            "Model metadata was not found. "
            "Run `python -m src.models.train` first."
        )

    else:

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        metrics = metadata.get("test_metrics", {})

        # ----------------------------------------------------
        # Performance
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Evaluation</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)

        c1.metric(
            "ROC-AUC",
            metric_value(metrics.get("roc_auc")),
        )

        c2.metric(
            "PR-AUC",
            metric_value(metrics.get("pr_auc")),
        )

        c3.metric(
            "F1",
            metric_value(metrics.get("f1")),
        )

        c4.metric(
            "Recall",
            metric_value(metrics.get("recall")),
        )

        c5.metric(
            "Precision",
            metric_value(metrics.get("precision")),
        )

        c6.metric(
            "Accuracy",
            metric_value(metrics.get("accuracy")),
        )

        # ----------------------------------------------------
        # Model details
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Configuration</div>',
            unsafe_allow_html=True,
        )

        config_rows = [
            {
                "Property": "Model",
                "Value": metadata.get(
                    "model_type",
                    "—",
                ),
            },
            {
                "Property": "Version",
                "Value": metadata.get(
                    "model_version",
                    "—",
                ),
            },
            {
                "Property": "Selection criterion",
                "Value": metadata.get(
                    "selection_criterion",
                    "—",
                ),
            },
            {
                "Property": "Experiment",
                "Value": metrics.get(
                    "experiment_id",
                    "—",
                ),
            },
            {
                "Property": "Training time",
                "Value": (
                    f"{metrics.get('training_time_seconds', 0):.2f} seconds"
                ),
            },
        ]

        st.dataframe(
            config_rows,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Features
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Model features</div>',
            unsafe_allow_html=True,
        )

        features = metadata.get("features", [])

        if features:

            feature_rows = [
                {"Feature": feature}
                for feature in features
            ]

            st.dataframe(
                feature_rows,
                use_container_width=True,
                hide_index=True,
            )

        # ----------------------------------------------------
        # Prediction activity
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Prediction activity</div>',
            unsafe_allow_html=True,
        )

        current_stats = stats()

        c1, c2 = st.columns(2)

        c1.metric(
            "Stored predictions",
            current_stats.get(
                "prediction_count",
                0,
            ),
        )

        c2.metric(
            "Average churn probability",
            format_probability(
                current_stats.get(
                    "average_churn_probability",
                    0,
                )
            ),
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

else:

    st.title("Dataset analysis")

    st.markdown(
        """
        <div class="page-description">
            Overview of the customer churn dataset used to train ChurnIQ.
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:

        df = clean_frame(load_raw())

        customer_count = len(df)
        column_count = len(df.columns)

        if "Churn" in df.columns:
            churn_rate = (
                df["Churn"].astype(str).eq("Yes").mean()
            )
        else:
            churn_rate = 0.0

        st.markdown(
            '<div class="section-title">Dataset overview</div>',
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Customers",
            f"{customer_count:,}",
        )

        c2.metric(
            "Columns",
            column_count,
        )

        c3.metric(
            "Observed churn rate",
            f"{churn_rate:.1%}",
        )

        st.markdown(
            '<div class="section-title">Sample records</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df.head(25),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:

        st.error(
            f"Dataset analysis failed: {type(exc).__name__}: {exc}"
        )

