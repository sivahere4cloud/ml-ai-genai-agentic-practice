# Iris Species Classification — Gaussian Naive Bayes

Multi-class classification project predicting Iris species using GaussianNB.

## Dataset
Iris (built-in via sklearn.datasets) — 150 rows, 4 features, 3 balanced classes

## Workflow
1. EDA — reused insights from prior SVC project (setosa separable, versicolor/virginica overlap)
2. No scaling required — Naive Bayes uses per-feature probability distributions, not distance
3. Model Training — `GaussianNB()`, default parameters (minimal tuning needed)
4. Evaluation — Accuracy, Confusion Matrix, Classification Report
5. Comparison — benchmarked directly against SVC (linear kernel) on the same dataset

## Key Findings
- GaussianNB achieved 97.78% accuracy vs SVC's 100% — the small gap reflects Naive Bayes' 
  feature-independence assumption, which doesn't fully hold for petal measurements (correlated in reality)
- Despite slightly lower accuracy, GaussianNB trained instantly with zero hyperparameter tuning — 
  a strong baseline choice when speed/simplicity matters more than squeezing out the last 2% accuracy
- Setosa was perfectly classified by both algorithms; the accuracy gap came entirely from 
  versicolor/virginica overlap

## Results
- Accuracy: 97.78%
- Precision/Recall/F1: 1.00 (setosa), ~0.94-0.97 (versicolor/virginica)

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