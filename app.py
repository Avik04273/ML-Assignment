"""
Streamlit App - AI Adoption: High Productivity Classifier
-----------------------------------------------------------
Lets a user upload test data (CSV), pick one of 5 trained classification
models, and view evaluation metrics, confusion matrix and classification
report on the uploaded data.

Run locally:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="AI Adoption - Productivity Classifier", layout="wide")

MODEL_DIR = "model"
MODELS = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

REQUIRED_FEATURE_COLS = [
    "Industry", "Job_Role", "Location", "Primary_AI_Tool",
    "Experience_Years", "Daily_Token_Usage", "Tasks_Automated_Per_Week",
    "Adoption_Year", "Adoption_Month", "Adoption_DayOfWeek",
]
TARGET_COL = "High_Productivity"


@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    st.title("🤖 AI Adoption — High Productivity User Classifier")
    st.markdown(
        """
        This app demonstrates **5 classification models** trained to predict
        whether an AI-tool user is a *High Productivity Gain* adopter
        (`High_Productivity = 1`) based on their industry, role, location,
        experience and AI-tool usage behaviour.
        """
    )

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Controls")
    model_name = st.sidebar.selectbox("Select a Model", list(MODELS.keys()))

    uploaded_file = st.sidebar.file_uploader(
        "Upload Test Data (CSV)", type=["csv"],
        help="Upload the provided test_data.csv (or a similarly formatted file)."
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Required columns:\n" + ", ".join(REQUIRED_FEATURE_COLS + [TARGET_COL])
    )

    if uploaded_file is None:
        st.info("👈 Upload `test_data.csv` from the sidebar to see predictions and metrics.")
        return

    # ---------------- Load data ----------------
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read CSV file: {e}")
        return

    missing = [c for c in REQUIRED_FEATURE_COLS if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {missing}")
        return

    has_labels = TARGET_COL in df.columns

    X = df[REQUIRED_FEATURE_COLS]

    # ---------------- Load model & predict ----------------
    model = load_model(MODELS[model_name])
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    st.subheader(f"Results — {model_name}")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Prediction Preview**")
        preview = X.copy()
        preview["Predicted_High_Productivity"] = y_pred
        preview["Predicted_Probability"] = np.round(y_proba, 3)
        st.dataframe(preview.head(20), use_container_width=True)

    if has_labels:
        y_true = df[TARGET_COL]
        metrics = compute_metrics(y_true, y_pred, y_proba)

        with col2:
            st.markdown("**Evaluation Metrics**")
            metrics_df = pd.DataFrame(
                {"Metric": list(metrics.keys()), "Value": [round(v, 4) for v in metrics.values()]}
            )
            st.table(metrics_df)

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Low (0)", "High (1)"],
                        yticklabels=["Low (0)", "High (1)"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with c2:
            st.markdown("**Classification Report**")
            report = classification_report(y_true, y_pred, target_names=["Low (0)", "High (1)"])
            st.text(report)
    else:
        with col2:
            st.warning(
                "Uploaded file has no `High_Productivity` column, so evaluation "
                "metrics / confusion matrix cannot be computed — showing predictions only."
            )

    # ---------------- Model comparison ----------------
    st.markdown("---")
    st.subheader("📊 Compare All Models on This Data")
    if has_labels and st.button("Run all 5 models on uploaded data"):
        rows = []
        for name, fname in MODELS.items():
            m = load_model(fname)
            yp = m.predict(X)
            ypr = m.predict_proba(X)[:, 1]
            rows.append({"Model": name, **compute_metrics(df[TARGET_COL], yp, ypr)})
        comp_df = pd.DataFrame(rows).set_index("Model").round(4)
        st.dataframe(comp_df, use_container_width=True)


if __name__ == "__main__":
    main()
