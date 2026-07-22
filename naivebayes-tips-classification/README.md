# Smoker Prediction — Gaussian Naive Bayes (Tips Dataset)

Classification project attempting to predict smoker status from restaurant tipping behavior.

## Dataset
Seaborn built-in `tips` dataset — 244 rows: total_bill, tip, sex, day, time, size → smoker (Yes/No)

## Workflow
1. EDA — checked class balance, bivariate plots (bill/tip/day/time/sex vs smoker status)
2. Feature Engineering — encoded sex, time (binary), day (one-hot)
3. Model Training — `GaussianNB()`
4. Evaluation — Accuracy, Confusion Matrix, Classification Report
5. Benchmarked against majority-class baseline

## Key Finding: Model Underperforms Baseline
GaussianNB achieved **60.81% accuracy**, below the majority-class baseline of **62.16%** 
(always predicting "non-smoker"). Recall for the minority class (smoker) was particularly 
poor (0.21) — the model correctly identified only ~1 in 5 actual smokers.

**Root cause:** total_bill, tip, sex, day, time have little genuine predictive relationship 
with smoking status — there's no strong real-world link between someone's tipping/dining 
pattern and whether they smoke. Additionally, GaussianNB assumes every feature is normally 
distributed, which doesn't hold for the binary/dummy-encoded categorical columns used here 
(sex, time, day) — a mismatch that likely compounded the weak signal.

## Why This Project Is Still Valuable
This demonstrates a critical real-world lesson: **a trained model can perform worse than a 
trivial baseline** when the available features lack predictive power for the target. Always 
compare against a majority-class baseline before trusting a model's accuracy — a "working" 
model isn't automatically a *useful* one. No Streamlit deployment was built for this project, 
since deploying a model that underperforms a naive guess would be misleading.

## Results
- GaussianNB Accuracy: 60.81%
- Majority-class baseline: 62.16%
- Smoker class recall: 0.21 (poor)

## Tech Stack
Python, pandas, numpy, matplotlib, seaborn, scikit-learn

## Next Steps
None planned for this specific project — served its purpose as a diagnostic exercise in 
recognizing weak feature-target relationships.