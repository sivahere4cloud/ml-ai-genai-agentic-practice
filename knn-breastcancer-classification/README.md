# KNN Classifier & Regressor — Breast Cancer & California Housing

Two related KNN projects in one repo: classification (Breast Cancer) and regression (California Housing).

## Part 1: KNN Classifier — Breast Cancer Wisconsin
- Dataset: built-in via sklearn, 569 rows, 30 features, binary target (malignant/benign)
- Feature scaling applied (StandardScaler) — essential for KNN's distance-based calculations
- Baseline (k=5): 96.28% accuracy
- GridSearchCV best k=10, improved test accuracy to ~97.3%
- Noted: accuracy-vs-k plot was noisier than textbook-smooth, likely due to limited test set size (188 rows)

## Part 2: KNN Regressor — California Housing
- Dataset: built-in via sklearn, ~20,640 rows, 8 features, continuous target (median house value)
- Baseline (k=6): R²=0.674, MAE=0.448 (~$44,800 average error)
- GridSearchCV best k=10, improved cross-validated R² to 0.683
- Larger dataset produced a smoother, more stable K-vs-R² curve than the classifier's noisier plot

## Key Cross-Project Finding
Both KNN Classifier and Regressor found **k=10** as optimal via GridSearchCV — despite very 
different domains (medical vs housing data) — suggesting a slightly larger neighborhood size 
generalized better than the smaller reference defaults (k=5, k=6) for these specific datasets.

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
Use the sidebar to switch between Classifier and Regressor demos.

## Next Steps
Docker, Kubernetes, CI/CD, AWS/Azure deployment — deferred until multiple model cycles are complete.