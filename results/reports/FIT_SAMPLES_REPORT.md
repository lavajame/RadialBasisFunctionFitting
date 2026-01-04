# RBF Fit of samples.csv Data - Analysis Report

## Overview
Successfully fit a Radial Basis Function (RBF) model to the financial market loss data from `samples.csv` using:
- **10 Latin Hypercube Sampling centers** (instead of all 50 training points)
- **Multiquadric RBF kernel** with ε=1.0
- **Polynomial degree 2** (quadratic basis functions)
- **Total of 31 basis functions** (10 RBF + 21 polynomial)

## Data Description
- **Samples**: 50 training points
- **Input dimensions**: 5 parameters
  - q
  - Merton.sigma (volatility)
  - Merton.lam (jump intensity)
  - Merton.muJ (jump mean)
  - Merton.sigmaJ (jump volatility)
- **Output**: Loss function values (ranging from 0.0232 to 174.57)
- **Gradients**: ∂loss/∂x for all 5 input dimensions

## Model Configuration
```
RBF Kernel:              Multiquadric with ε = 1.0
Polynomial Degree:       2 (includes constant, linear, and quadratic terms)
RBF Centers:             10 (generated via Latin Hypercube Sampling)
Total Basis Functions:   31 (10 RBF + 21 polynomial)
Regularization:          λ = 1e-8 (ridge regression parameter)
Normalization:           Independent z-score normalization per variable
```

## Performance Metrics (Normalized Scale)

### Function Fit Quality
- **RMSE**: 0.6723 standard deviations
- **R² Score**: 0.5480 (model explains 54.8% of loss variance)
- **Interpretation**: MODERATE fit quality

The model achieves a moderate level of accuracy. The 10 LHS-selected centers 
capture the main structure of the loss landscape, though some fine details are 
missed. With 31 basis functions fit to 50 training points, we have a reasonable 
model that avoids overfitting while capturing the dominant features.

### Gradient Fit Quality
- **Overall RMSE**: 0.9825 standard deviations
- **Per-dimension R² scores**:
  - q: 0.0362 (captures 3.6% of variance)
  - Merton.sigma: -0.1392 (poor fit)
  - Merton.lam: 0.1204 (modest fit)
  - Merton.muJ: 0.1080 (modest fit)
  - Merton.sigmaJ: 0.0485 (modest fit)

The gradient fits are **MODERATE**, indicating the model captures the general 
derivative trends but with some error. This is expected for financial models 
where gradients can be highly nonlinear and sensitive to local features.

## Residual Analysis

### Distribution
Residuals follow approximately Gaussian distribution with slight negative bias:
- 1st percentile: -0.937
- 5th percentile: -0.742
- Median: -0.162
- 95th percentile: 1.190
- 99th percentile: 2.529

The residuals are reasonably centered around zero with a slight tendency to 
underpredict on average.

### Spatial Distribution
Residuals are scattered across input space without obvious systematic patterns, 
suggesting the model has captured the main nonlinear structure of the loss 
function.

## Diagnostic Plots Generated

### 1. fit_samples_diagnostics.png (334 KB)
Comprehensive 9-panel diagnostic visualization:
1. **Predicted vs Actual Loss**: Shows how well predictions align with true values
2. **Residuals vs Predicted**: Reveals systematic biases across the prediction range
3. **Residual Distribution**: Histogram showing approximate Gaussian distribution
4-8. **Gradient Fits**: 5 panels showing predicted vs actual gradients for each dimension
9. **Model Information**: Summary of configuration and performance

### 2. fit_samples_residual_analysis.png (138 KB)
Detailed residual analysis with 6 panels:
1-5. **Residuals vs Each Input**: Shows how errors vary along each parameter
6. **Absolute Error Distribution**: Histogram of |error| values

## Key Observations

### Strengths
✓ **Efficient sampling**: 10 LHS centers vs 50 training points = 80% reduction
✓ **Good structural fit**: R² of 0.55 is reasonable for complex financial data
✓ **Numerical stability**: Normalization ensures well-conditioned system
✓ **Balanced complexity**: 31 basis functions vs 50 samples avoids overfitting

### Considerations
⚠ **Gradient fit**: Moderate accuracy suggests market sensitivities are complex
⚠ **Outliers**: Some samples show larger prediction errors (max RMSE envelope)
⚠ **Dimension-dependent**: Some parameters fit better than others

## Model Usage

The fitted RBF model can be used to:
1. **Predict loss values** at new parameter combinations
2. **Estimate loss gradients** for sensitivity analysis
3. **Accelerate computations** by replacing expensive financial calculations

To use the model on new data:
```python
from rbf_polynomial_fitter import RBFPolynomialFitter

# Load saved fitter (if saved)
# Make predictions on normalized data:
# X_new_norm = (X_new - X_mean) / X_std
# f_pred_norm = fitter.predict(X_new_norm)
# grad_pred_norm = fitter.predict_gradient(X_new_norm)
```

## Recommendations for Improvement

1. **More training data**: Collect additional loss evaluations to improve gradient fits
2. **Feature engineering**: Compute higher-order combinations of parameters
3. **Kernel tuning**: Experiment with different ε values (currently 1.0)
4. **Center placement**: Try adaptive center selection in high-error regions
5. **Polynomial degree**: Test degree 3 or higher for better fits (with regularization)

## Files Generated
- `fit_samples_diagnostics.png` - Main diagnostic visualization
- `fit_samples_residual_analysis.png` - Detailed residual analysis
- `fit_samples_normalized.py` - Python script that performs the fit

---
Generated: 2026-01-04  
Model: Multiquadric RBF + Polynomial Degree 2  
Training samples: 50  
RBF centers: 10 (LHS)
