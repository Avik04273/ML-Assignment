# AI Adoption — High Productivity User Classifier

## a. Problem Statement
Organizations across industries are adopting AI tools (ChatGPT, Claude, Gemini,
Copilot, Perplexity, Midjourney, DeepSeek, etc.) at different rates, and the
productivity benefit users report varies widely. This project builds a
**binary classification** model that predicts whether a user is a
**"High Productivity Gain"** adopter of AI tools (`1`) or not (`0`), based on
their industry, job role, location, years of experience, primary AI tool,
daily token usage, number of tasks automated per week, and when they adopted
the tool.

The target label `High_Productivity` is engineered from the continuous
`Productivity_Gain_Percent` column: a user is labeled `1` if their reported
productivity gain is **at or above the dataset median (8.1%)**, and `0`
otherwise. This produces a balanced binary classification problem suitable
for comparing multiple ML algorithms.

## b. Dataset Description
- **Source:** `user_level_ai_adoption.csv`
- **Instances:** 15,000 user records
- **Raw columns (10):** `User_ID`, `Industry`, `Job_Role`, `Location`,
  `Experience_Years`, `Primary_AI_Tool`, `Daily_Token_Usage`,
  `Tasks_Automated_Per_Week`, `Productivity_Gain_Percent`, `Adoption_Date`
- **Engineered features used for modeling (10 raw → 40+ after encoding):**
  - Categorical (one-hot encoded): `Industry` (6), `Job_Role` (22),
    `Location` (5), `Primary_AI_Tool` (7)
  - Numeric (standardized): `Experience_Years`, `Daily_Token_Usage`,
    `Tasks_Automated_Per_Week`, `Adoption_Year`, `Adoption_Month`,
    `Adoption_DayOfWeek` (the last three derived from `Adoption_Date`)
  - **Target:** `High_Productivity` (binary, derived from
    `Productivity_Gain_Percent`)
- `User_ID`, `Adoption_Date` (raw) and `Productivity_Gain_Percent` are
  dropped from the model inputs (identifier / leakage / source-of-target
  columns).
- Feature count after preprocessing (one-hot + numeric) comfortably exceeds
  the minimum of 12; instance count (15,000) exceeds the minimum of 500.
- Train/test split: 80% / 20%, stratified on the target (12,000 train /
  3,000 test rows). The 3,000-row held-out test set is saved as
  `test_data.csv` and used both for the metrics below and for the Streamlit
  app demo.

## c. GitHub Repository Link
https://github.com/Avik04273/ML-Assignment

## d. Models Used

### Comparison Table (on the 3,000-row held-out test set)

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.8913   | 0.9513 | 0.8821    | 0.9054 | 0.8936 | 0.7829 |
| Decision Tree               | 0.8323   | 0.8323 | 0.8334    | 0.8340 | 0.8337 | 0.6646 |
| kNN                         | 0.8200   | 0.9023 | 0.7859    | 0.8836 | 0.8319 | 0.6447 |
| Naive Bayes                 | 0.5957   | 0.8381 | 0.7774    | 0.2771 | 0.4086 | 0.2559 |
| Random Forest (Ensemble)    | 0.8847   | 0.9446 | 0.8466    | 0.9418 | 0.8917 | 0.7741 |

*(AUC = ROC-AUC on the positive class probability; Precision/Recall/F1 computed for the "High Productivity" (1) class; MCC = Matthews Correlation Coefficient.)*

### Observations

| ML Model Name             | Observation about model performance |
|-----------------------------|--------------------------------------|
| Logistic Regression         | Best overall performer. The relationship between the one-hot encoded categorical features (Industry, Job Role, AI Tool) and productivity gain is largely linear/additive, which suits logistic regression well. Highest AUC (0.9513) and strong balance between precision and recall. |
| Decision Tree                | Moderate performance with visible overfitting tendency (single tree, no depth limit) — accuracy and AUC nearly identical (~0.83) since a single tree gives hard 0/1-like probability estimates, limiting AUC. Captures non-linear splits but is less stable than the ensemble. |
| kNN                           | Reasonable performance but sensitive to the high-dimensional, mostly-categorical (one-hot encoded) feature space — distance metrics get diluted with many sparse dummy variables, lowering precision (0.7859) despite a decent AUC. |
| Naive Bayes                  | Weakest performer (Accuracy 0.60, Recall 0.28). The Gaussian Naive Bayes independence assumption is violated by the correlated one-hot encoded categorical features, and it strongly under-predicts the positive ("High Productivity") class, hurting recall and F1 the most. |
| Random Forest (Ensemble)     | Second-best overall and most robust — highest Recall (0.9418), strong AUC (0.9446), and much less overfitting than the single Decision Tree thanks to bagging across 150 trees. Handles the mixed categorical/numeric feature space well. |
| **Overall Winner for your dataset?** | **Logistic Regression** — highest AUC (0.9513) and MCC (0.7829), with Random Forest a very close second. Given the near-linear separability of this engineered feature set, the simpler linear model edges out the ensemble while also being far cheaper to train/serve. |

## How to Run Locally
```bash
pip install -r requirements.txt
python model/train_models.py   # optional: retrain models from scratch
streamlit run app.py
```

## Streamlit App Features
- CSV upload of test data (sidebar)
- Model selection dropdown (5 models)
- Evaluation metrics table (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix heatmap + classification report
- One-click comparison of all 5 models on the uploaded data

## Live App Link
https://aokxnzyrayped6p88rr3zq.streamlit.app/
