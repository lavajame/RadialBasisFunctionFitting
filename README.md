# Radial Basis Function Polynomial Fitting

A comprehensive implementation of RBF-polynomial hybrid surrogate models with complexity analysis and optimization.

## Project Structure

```
RadialBasisFunctionFitting/
├── src/                              # Core implementation
│   └── rbf_polynomial_fitter.py      # Main RBF+Polynomial fitter class
├── scripts/                          # Analysis and utility scripts
│   ├── polynomial_degree_sweep.py    # Optimize polynomial degree (1-5)
│   ├── complexity_sweep.py           # Analyze complexity vs performance
│   ├── diagnostics.py                # Diagnostic visualization tools
│   ├── rbf_sweep_analysis.py         # RBF shape parameter optimization
│   ├── examples.py                   # Usage examples
│   └── [other analysis scripts]
├── data/                             # Data files
│   └── samples.csv                   # Training data (50 samples)
├── results/                          # Generated results
│   ├── plots/                        # Visualization PNG files
│   └── reports/                      # Analysis reports (Markdown/Text)
├── docs/                             # Documentation
└── README.md                         # This file
```

## Quick Start

### Installation

```bash
# Clone and navigate
git clone https://github.com/lavajame/RadialBasisFunctionFitting.git
cd RadialBasisFunctionFitting

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install numpy scipy matplotlib scikit-learn
```

### Basic Usage

```python
from src.rbf_polynomial_fitter import RBFPolynomialFitter
import numpy as np

# Initialize fitter
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.115,
    polynomial_degree=5,
    regularization_lambda=1e-8
)

# Fit model
fitter.fit(X_train, y_train, df=gradients_train, n_centers=5)

# Predict
y_pred = fitter.predict(X_test)
grad_pred = fitter.predict_gradient(X_test)
```

## Key Findings

### Minimum Complexity for Performance Targets

| Target RMSE | Min Bases | Config | Actual RMSE | R² |
|-------------|-----------|--------|-----------|-----|
| ≤ 0.50 | 253 | Deg 5, 1 center | 0.4945 | 0.7555 |
| ≤ 0.65 | 31 | Deg 2, 10 centers | 0.6498 | 0.5777 |
| ≤ 0.70 | 21 | Deg 1, 15 centers | 0.6998 | 0.5103 |

### Critical Insight: Polynomial Degree Dominates

**Polynomial degree is the primary lever for fit quality.** RBF centers contribute minimally:
- Degree 5 + 1 center: RMSE = 0.4945
- Degree 5 + 20 centers: RMSE = 0.4944
- **Difference: < 0.02% (negligible)**

## Analysis Scripts

### Polynomial Degree Optimization
```bash
python scripts/polynomial_degree_sweep.py
```
Tests polynomial degrees 1-5 with optimal RBF configuration. Output:
- `results/plots/polynomial_degree_analysis.png`
- `results/reports/ANALYSIS_RESULTS.md`

### Complexity Analysis
```bash
python scripts/complexity_sweep.py
```
Systematic evaluation of all polynomial degree × RBF center combinations. Output:
- `results/plots/complexity_analysis.png`
- `results/reports/COMPLEXITY_ANALYSIS.md`

### RBF Shape Parameter Optimization
```bash
python scripts/rbf_sweep_analysis.py
```
Optimizes the Gaussian RBF shape parameter (ε). Output:
- `results/plots/rbf_sweep_comparison.png`
- `results/reports/RBF_SWEEP_REPORT.md`

## Model Configuration

**Optimal for this problem:**
- RBF Kernel: Gaussian
- Shape Parameter: ε = 0.115
- Polynomial Degree: 5 (21 polynomial terms for 2D)
- LHS Centers: 1-5 (minimal variation in performance)
- Total Basis Functions: 253-257
- Training Samples: 50
- Regularization: λ = 1e-8

**Performance:**
- Function Fit: RMSE = 0.4945, R² = 0.7555
- Gradient Fit: RMSE = 0.3448
- Explains 75.5% of variance

## Reports

Detailed analysis reports are in `results/reports/`:
- `ANALYSIS_RESULTS.md` — Polynomial degree sweep findings
- `COMPLEXITY_ANALYSIS.md` — Full complexity trade-off analysis
- `RBF_SWEEP_REPORT.md` — RBF parameter optimization
- `CONDITION_NUMBER_ANALYSIS.md` — Numerical stability analysis
- `DIAGNOSTICS_REPORT.md` — Model diagnostic checks

## Features

✓ Hybrid RBF + polynomial basis functions  
✓ Gradient-aware fitting with L2 regularization  
✓ Latin Hypercube Sampling for RBF centers  
✓ Automatic basis function counting  
✓ Comprehensive diagnostic tools  
✓ Scalable to high dimensions  

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- scikit-learn (for LHS sampling)

## License

MIT

## Author

lavajame (james.lavan@gmail.com)
