# Position Salary Prediction — Polynomial Regression

End-to-end Polynomial Regression project predicting salary based on position level, using a deliberately non-linear dataset.

## Dataset
Position_Salaries.csv — `Level` (1-10) → `Salary`, 10 rows, classic non-linear teaching dataset

## Workflow
1. EDA — scatterplot showing a clear non-linear (curved) trend; correlation alone was high but misleading (same lesson as the UK House Price project)
2. Feature Engineering — `PolynomialFeatures` (degree=4) to transform `Level` into `Level, Level², Level³, Level⁴`
3. Model Training — `sklearn.linear_model.LinearRegression` fitted on the transformed polynomial features
4. Evaluation — MAE, MSE, RMSE, R² Score
5. Comparison — plotted Linear vs Polynomial fit side-by-side to visually demonstrate why polynomial regression fits this data better
6. Model Persistence — both the model and the `PolynomialFeatures` transformer saved with `pickle` (both required for correct predictions on new input)
7. Deployment — local Streamlit app

## Key Concept
Polynomial Regression is not a separate algorithm — it's Linear Regression applied to polynomial-transformed features. The curve comes entirely from the feature transformation, not the model itself.

## Results
- R² Score: [insert your value]
- RMSE: [insert your value]

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
Docker, Kubernetes, CI/CD, AWS/Azure deployment — deferred until multiple model cycles are complete (see portfolio roadmap).