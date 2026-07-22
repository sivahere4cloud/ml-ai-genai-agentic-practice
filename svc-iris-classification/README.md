# Iris Species Classification — Support Vector Classifier (SVC)

Multi-class classification project predicting Iris species from flower measurements.

## Dataset
Iris (built-in via sklearn.datasets) — 150 rows, 4 features (sepal/petal length & width), 3 balanced classes

## Workflow
1. EDA — pairplot revealed setosa is linearly separable; versicolor/virginica overlap somewhat
2. Feature Engineering — StandardScaler applied (SVM is distance-based, scale-sensitive)
3. Model Training — compared kernels (linear, poly, rbf, sigmoid) and C values
4. Evaluation — classification report, confusion matrix

## Key Findings
- **Linear kernel achieved 100% accuracy**, outperforming rbf (96.67%), poly (90%), sigmoid (90%)
- Confirms EDA insight: Iris classes are largely linearly separable, so the simpler linear kernel was the best (not just adequate) choice — more complex kernels aren't automatically better
- For rbf kernel, accuracy plateaued at C=1 and above (96.67%), showing diminishing returns from hardening the margin further

## Results
- Accuracy: 100%
- Precision/Recall/F1: 1.00 across all three classes

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