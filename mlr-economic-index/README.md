# Economic Index Prediction — Multiple Linear Regression

End-to-end MLR project predicting a market index price based on interest rate and unemployment rate.

## Dataset
Kaggle — Economic Index Dataset (`interest_rate`, `unemployment_rate` → `index_price`, 24 monthly rows)

## Workflow
1. EDA — distributions, bivariate analysis, correlation matrix
2. Feature Engineering — StandardScaler applied to both features, train-test split
3. Model Training — `sklearn.linear_model.LinearRegression`
4. Evaluation — MAE, MSE, RMSE, R² Score
5. Model Persistence — model and scaler saved with `pickle`
6. Deployment — local Streamlit app

## Key Finding: Multicollinearity
`interest_rate` and `unemployment_rate` are strongly correlated with each other (-0.93), confirmed via VIF. This was deliberately kept (not dropped) since both are meaningful economic indicators — the model still predicts well (high R²), but individual coefficient values should be interpreted cautiously due to this multicollinearity, rather than treated as stable standalone effects.

## Results
- R² Score: [insert your value]
- RMSE: [insert your value]

## Tech Stack
Python, pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, Streamlit

## How to Run
\`\`\`
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cd src
streamlit run app.py
\`\`\`

## Next Steps
Docker, Kubernetes, CI/CD, AWS/Azure deployment — deferred until multiple model cycles are complete (see portfolio roadmap).