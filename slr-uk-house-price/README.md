# Simple Linear Regression — Salary Prediction

End-to-end ML project predicting salary based on years of experience, using Simple Linear Regression.

## Problem Statement
Predict an employee's salary based on their years of experience using a simple linear regression model.

## Dataset
Kaggle — Salary Data (Simple Linear Regression): `YearsExperience`, `Salary` (30 rows)

## Workflow
1. Exploratory Data Analysis (distribution, correlation, outlier checks)
2. Feature Engineering (train-test split; scaling not required for single-feature SLR)
3. Model Training — `sklearn.linear_model.LinearRegression` (OLS)
4. Evaluation — MAE, MSE, RMSE, R² Score, residual analysis
5. Model Persistence — saved with `pickle`
6. Deployment — local Streamlit app for interactive prediction

## Results
- R² Score: [insert your value]
- RMSE: £[insert your value]

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
Multi-cloud deployment (AWS, Azure), Docker, Kubernetes, CI/CD — planned once foundational model cycles are complete.