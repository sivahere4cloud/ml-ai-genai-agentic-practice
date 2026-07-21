# Titanic Survival Prediction — Logistic Regression

End-to-end binary classification project predicting passenger survival on the Titanic.

## Dataset
Titanic dataset (891 rows) — Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked → Survived

## Workflow
1. EDA — survival distribution, gender/class survival patterns, missing value analysis
2. Data Cleaning:
   - `Cabin` (77% missing) → converted to binary `HasCabin` feature instead of imputing
   - `Age` (177 missing) → imputed using median grouped by Pclass + Sex (informed imputation, not blind mean/median)
   - `Embarked` (2 missing) → mode imputation (safe given tiny missing count)
3. Feature Engineering — dropped non-predictive columns (PassengerId, Name, Ticket); encoded `Sex` (binary map) and `Embarked` (one-hot, drop_first)
4. Train-Test Split — stratified split to preserve class balance
5. Model Training — `sklearn.linear_model.LogisticRegression`
6. Evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC
7. Model Persistence — saved with `pickle`
8. Deployment — local Streamlit app

## Key Findings
- Strongest positive predictor: `Sex` (being female significantly increased survival odds)
- Strongest negative predictor: `Pclass` (higher class number = lower survival odds)
- Findings align with the historical "women and children first" evacuation pattern and first-class lifeboat access

## Results
- Accuracy: [insert your value]
- Precision: [insert your value]
- Recall: [insert your value]
- F1 Score: [insert your value]
- ROC-AUC: [insert your value]

## Tech Stack
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, Streamlit

## How to Run
\`\`\`
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd src
streamlit run app.py
\`\`\`

## Next Steps
Docker, Kubernetes, CI/CD, AWS/Azure deployment — deferred until multiple model cycles are complete.

## Hyperparameter Tuning
Applied GridSearchCV (C: [0.01, 0.1, 1, 10, 100], penalty: [l1, l2]) with 5-fold CV, optimizing for F1 score.
Result: tuned model achieved identical F1 (0.7424) to the default LogisticRegression configuration —
indicating the default settings were already near-optimal for this dataset/feature set.

## Hyperparameter Tuning Comparison
Both GridSearchCV (exhaustive, 10 combinations) and RandomizedSearchCV (10 of 14 sampled) 
converged to F1 = 0.7424 — identical to the untuned default model. This indicates the 
model's performance ceiling here is governed by feature quality/engineering rather than 
hyperparameter choice. Further improvement would likely require additional feature 
engineering (e.g., extracting titles from Name, family size from SibSp+Parch) rather 
than further tuning.