# AutoStack: Automated White-box Stacking

**Status**: Alpha Preview (v0.1.0-alpha)

AutoStack is a scikit-learn compatible library that automates Stacking Ensemble while providing **statistical inference** (summary tables with Bootstrap CIs) - something AutoML tools usually hide from you.

## Current Features (v0.1.0-alpha)
- **Automatic OOF (Out-of-Fold) matrix generation** with K-Fold CV.
- **Linear Meta-learners** (RidgeCV, LassoCV, LinearRegression) with **statsmodels-like summaries** (Coefficients, Bootstrap Standard Errors, Confidence Intervals, Sign Stability).
- **Multicollinearity diagnosis** (VIF & Correlation Matrix) to detect redundant base models.
- scikit-learn compatible API (`fit`, `predict`).

## Upcoming Roadmap
- `error_handling` & `verbose` controls for production robustness.
- Non-linear meta-learners (XGBoost, LightGBM as meta models) with Feature Importance summaries.
- `AutoStackRegressorSearch` for automatic meta-learner selection.
- `SuperStackRegressor` (concatenating original features to meta-features).
- Enable stacking classification algorithms.

## Example Usage

Here's a quickstart with the California Housing dataset to show how `AutoStackRegressor` works.

```python
# 1. Import relevant libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from autostack import AutoStackRegressor

# 2. Load data
data = fetch_california_housing()
X, y = data.data, data.target

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize model
model = AutoStackRegressor(
    base_learners=['lr', 'rf', 'xgb', 'mlp'], 
    meta_learner='ridge_cv',                 
    kf=5,
    bootstrap_iter=100,                       
    random_state=42,
    verbose=1                                
)

# 5. Train model
model.fit(X_train, y_train)

# 6. Test model
y_pred = model.predict(X_test)

# 7. Print Report
print("\n" + "="*70)
print("BASIC SUMMARY")
print("="*70)
model.summary(summary_level='basic')   # basic summary containing metrics for base learners and meta learner

print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
# a more detailed
summary_dict = model.summary(summary_level='diagnostic', return_table=True)
```

#### Example Output (Diagnostic Summary)

Running the example above produces the following beautifully formatted diagnostic tables:

```text
======================================================================
BASIC SUMMARY
======================================================================
======================================================================
                STACK MODEL METRICS REPORT                
======================================================================
======================================================================
╒══════════════╤══════════╤══════════╤══════════╤══════════╕
│              │     RMSE │      MSE │      MAE │       R2 │
╞══════════════╪══════════╪══════════╪══════════╪══════════╡
│ Meta Learner │ 0.485216 │ 0.235435 │ 0.323538 │ 0.823879 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ lr           │ 0.720645 │ 0.519329 │ 0.529061 │ 0.611507 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ rf           │ 0.511143 │ 0.261267 │ 0.334943 │ 0.804554 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ xgb          │ 0.495859 │ 0.245876 │ 0.339691 │ 0.816068 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ mlp          │ 0.95806  │ 0.917879 │ 0.725543 │ 0.313365 │
╘══════════════╧══════════╧══════════╧══════════╧══════════╛
======================================================================

======================================================================
DIAGNOSTIC SUMMARY
======================================================================
======================================================================
                STACKING MODEL DIAGNOSTICS REPORT                
======================================================================

[BASIC PERFORMANCE METRICS]
╒══════════════╤══════════╤══════════╤══════════╤══════════╕
│              │     RMSE │      MSE │      MAE │       R2 │
╞══════════════╪══════════╪══════════╪══════════╪══════════╡
│ Meta Learner │ 0.485216 │ 0.235435 │ 0.323538 │ 0.823879 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ lr           │ 0.720645 │ 0.519329 │ 0.529061 │ 0.611507 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ rf           │ 0.511143 │ 0.261267 │ 0.334943 │ 0.804554 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ xgb          │ 0.495859 │ 0.245876 │ 0.339691 │ 0.816068 │
├──────────────┼──────────┼──────────┼──────────┼──────────┤
│ mlp          │ 0.95806  │ 0.917879 │ 0.725543 │ 0.313365 │
╘══════════════╧══════════╧══════════╧══════════╧══════════╛
======================================================================

[BOOTSTRAP COEFFICIENT ANALYSIS]
╒═══════════╤════════════╤════════════╤═════════════════╤══════════════════╤══════════════════╕
│           │       mean │     median │   2.5% quantile │   97.5% quantile │   sign stability │
╞═══════════╪════════════╪════════════╪═════════════════╪══════════════════╪══════════════════╡
│ Intercept │ -0.0192309 │ -0.0197194 │      -0.0332745 │      -0.00176125 │             0.98 │
├───────────┼────────────┼────────────┼─────────────────┼──────────────────┼──────────────────┤
│ lr        │ -0.0306742 │ -0.0310612 │      -0.0541689 │      -0.00693166 │             1    │
├───────────┼────────────┼────────────┼─────────────────┼──────────────────┼──────────────────┤
│ rf        │  0.428255  │  0.428842  │       0.38686   │       0.466877   │             1    │
├───────────┼────────────┼────────────┼─────────────────┼──────────────────┼──────────────────┤
│ xgb       │  0.640332  │  0.63976   │       0.597914  │       0.679176   │             1    │
├───────────┼────────────┼────────────┼─────────────────┼──────────────────┼──────────────────┤
│ mlp       │ -0.031367  │ -0.0312339 │      -0.0474128 │      -0.0157623  │             1    │
╘═══════════╧════════════╧════════════╧═════════════════╧══════════════════╧══════════════════╛
======================================================================
Correlation Matrix
╒═════╤══════════╤══════════╤══════════╤══════════╕
│     │       lr │       rf │      xgb │      mlp │
╞═════╪══════════╪══════════╪══════════╪══════════╡
│ lr  │ 1        │ 0.865201 │ 0.8676   │ 0.761653 │
├─────┼──────────┼──────────┼──────────┼──────────┤
│ rf  │ 0.865201 │ 1        │ 0.970876 │ 0.685892 │
├─────┼──────────┼──────────┼──────────┼──────────┤
│ xgb │ 0.8676   │ 0.970876 │ 1        │ 0.679085 │
├─────┼──────────┼──────────┼──────────┼──────────┤
│ mlp │ 0.761653 │ 0.685892 │ 0.679085 │ 1        │
╘═════╧══════════╧══════════╧══════════╧══════════╛
======================================================================

[MULTICOLLINEARITY (VIF)]
╒═══════════╤══════════╕
│ Feature   │      VIF │
╞═══════════╪══════════╡
│ lr        │  5.30828 │
├───────────┼──────────┤
│ rf        │ 18.1768  │
├───────────┼──────────┤
│ xgb       │ 18.4148  │
├───────────┼──────────┤
│ mlp       │  2.40102 │
╘═══════════╧══════════╛
```


## Installation (Development)
```bash
git clone ...
cd autostack
pip install -e .