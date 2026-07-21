# Ridge, Lasso & ElasticNet — Economic Index Prediction

## Key Findings
1. Manual alpha testing showed Lasso zeroing out `interest_rate` at alpha=170, and both coefficients at alpha=191 (model collapse).
2. Cross-validation (LassoCV, RidgeCV, ElasticNetCV) found much smaller optimal alphas (0.19, 1.0, 0.39), all converging close to plain Linear Regression's original coefficients.
3. Conclusion: despite strong multicollinearity (-0.93 correlation between features), this small (24-row) dataset did not benefit from heavy regularization — CV-based tuning is essential rather than assuming regularization strength.
4. Final model deployed: RidgeCV (alpha=1.0), chosen for its stability without risk of arbitrary feature elimination.