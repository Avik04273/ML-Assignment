"""
train_models.py
----------------
Trains 5 classification models on the AI Adoption dataset and saves:
  - a trained scikit-learn Pipeline (preprocessing + model) for each algorithm
  - a metrics comparison table (metrics.csv)
  - a held-out test_data.csv (raw, unprocessed rows + true label) used by the
    Streamlit app for demonstration / evaluation.

Dataset : user_level_ai_adoption.csv  (15,000 rows, 10 raw columns)
Problem : Binary classification -> predict whether a user is a
          "High Productivity Gain" adopter of AI tools (1) or not (0),
          based on their industry, role, location, experience, tool usage
          and adoption behaviour.

Target engineering:
    High_Productivity = 1 if Productivity_Gain_Percent >= median(Productivity_Gain_Percent)
                        else 0

Run:
    python train_models.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
DATA_PATH = "../user_level_ai_adoption.csv"   # relative to model/ folder
OUT_DIR = "."


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df):
    df = df.copy()

    # Target: binarize Productivity_Gain_Percent at its median
    median_gain = df["Productivity_Gain_Percent"].median()
    df["High_Productivity"] = (df["Productivity_Gain_Percent"] >= median_gain).astype(int)

    # Date-derived features
    df["Adoption_Date"] = pd.to_datetime(df["Adoption_Date"])
    df["Adoption_Year"] = df["Adoption_Date"].dt.year
    df["Adoption_Month"] = df["Adoption_Date"].dt.month
    df["Adoption_DayOfWeek"] = df["Adoption_Date"].dt.dayofweek

    # Drop columns not used as model inputs
    df = df.drop(columns=["User_ID", "Adoption_Date", "Productivity_Gain_Percent"])

    return df, median_gain


CATEGORICAL_COLS = ["Industry", "Job_Role", "Location", "Primary_AI_Tool"]
NUMERIC_COLS = ["Experience_Years", "Daily_Token_Usage", "Tasks_Automated_Per_Week",
                 "Adoption_Year", "Adoption_Month", "Adoption_DayOfWeek"]
TARGET_COL = "High_Productivity"


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )


# --------------------------------------------------------------------------
# 3. Models
# --------------------------------------------------------------------------
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=150, max_depth=12, random_state=RANDOM_STATE
        ),
    }


def evaluate(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main():
    df = load_data(DATA_PATH)
    df, median_gain = engineer_features(df)
    print(f"Median Productivity_Gain_Percent used as split threshold: {median_gain}")

    X = df[CATEGORICAL_COLS + NUMERIC_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, model in get_models().items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", model),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = evaluate(y_test, y_pred, y_proba)
        results[name] = metrics

        # Save trained pipeline
        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(pipe, os.path.join(OUT_DIR, f"{fname}.pkl"))
        print(f"Saved model/{fname}.pkl  ->  {metrics}")

    # Save comparison table
    metrics_df = pd.DataFrame(results).T
    metrics_df.index.name = "ML Model Name"
    metrics_df = metrics_df.round(4)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"))
    print("\nComparison table:\n", metrics_df)

    # Save held-out RAW test set (with true label) for the Streamlit app
    test_out = X_test.copy()
    test_out[TARGET_COL] = y_test.values
    test_out.to_csv("../test_data.csv", index=False)
    print(f"\nSaved ../test_data.csv with {len(test_out)} rows for app demo/testing.")

    with open(os.path.join(OUT_DIR, "median_gain.json"), "w") as f:
        json.dump({"median_productivity_gain": median_gain}, f)


if __name__ == "__main__":
    main()
