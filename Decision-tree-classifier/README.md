# Iris Species Classification — Decision Tree

Multi-class classification project predicting Iris species using a Decision Tree Classifier.

## Dataset
Iris (built-in via sklearn.datasets) — 150 rows, 4 features (sepal/petal length & width), 3 balanced classes

## Workflow
1. Loaded data directly via `load_iris()`, built X (features) and y (target)
2. Train-test split (80/20)
3. Baseline `DecisionTreeClassifier()` trained
4. Hyperparameter tuning via GridSearchCV — tested `criterion` (gini, entropy, log_loss), 
   `splitter` (best, random), `max_depth` (1-5), `max_features` (auto, sqrt, log2)
5. Evaluation — Confusion Matrix, Classification Report, Accuracy

## Key Findings
- Best parameters: `criterion='gini'`, `max_depth=5`, `max_features='sqrt'`, `splitter='random'`
- Best cross-validated accuracy: 95.83%
- Test set accuracy: 86.67%
- Confusion matrix showed perfect classification for class 0 (setosa), but some 
  misclassification between classes 1 and 2 (versicolor/virginica) — consistent with 
  known overlap between these two species in petal measurements

## Results
- Cross-validated Accuracy: 95.83%
- Test Accuracy: 86.67%
- Precision/Recall/F1: 1.00 (class 0), 1.00/0.69/0.82 (class 1), 0.64/1.00/0.78 (class 2)

## Tech Stack
Python, pandas, numpy, matplotlib, scikit-learn

## How to Run
\`\`\`
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
\`\`\`

## Next Steps
Streamlit deployment, Docker, Kubernetes, CI/CD — deferred until multiple model cycles are complete.