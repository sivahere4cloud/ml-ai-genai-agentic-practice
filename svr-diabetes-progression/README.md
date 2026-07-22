# Diabetes Progression Prediction — Support Vector Regression (SVR)

Regression project predicting disease progression score from 10 patient measurements.

## Dataset
Diabetes dataset (built-in via sklearn.datasets) — 442 rows, 10 pre-standardized features, continuous target

## Workflow
1. EDA — correlation analysis (bmi, s5, bp strongest predictors; sex weakest)
2. Feature Engineering — features already pre-standardized by sklearn; StandardScaler reapplied for pipeline consistency
3. Model Training — compared kernels (linear, poly, rbf) and epsilon values
4. Evaluation — R², RMSE; benchmarked against plain Linear Regression

## Key Findings
- **Linear kernel outperformed rbf and poly** (R²=0.445 vs 0.182 and 0.283) — consistent with the largely linear correlations seen in EDA
- Tuning epsilon on the rbf kernel (0.01 to 5.0) barely changed performance (R² 0.18-0.19) — nowhere close to closing the gap with the linear kernel
- **Plain Linear Regression (R²=0.4526) slightly outperformed SVR with a linear kernel (R²=0.4451)** — the added complexity of SVR wasn't justified here; always benchmark against a simple baseline
- Overall R² (~0.45) reflects this dataset's known inherent noisiness, not a modeling shortfall

## Results
- Best model: SVR (linear kernel) — R²: 0.4451, RMSE: [insert]
- Baseline Linear Regression — R²: 0.4526

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