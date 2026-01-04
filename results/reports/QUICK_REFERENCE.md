# Quick Reference Guide

## Installation & Setup

```python
from rbf_polynomial_fitter import RBFPolynomialFitter
import numpy as np

# Requires: numpy, scipy, matplotlib (optional for examples)
```

---

## Quick Examples

### 1. Basic Usage (Default Multiquadric)
```python
X_train = np.random.randn(20, 2)
y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])

fitter = RBFPolynomialFitter(polynomial_degree=1)
fitter.fit(X_train, y_train)

X_test = np.random.randn(10, 2)
y_pred = fitter.predict(X_test)
```

### 2. Using Gaussian Kernel with Shape Parameter
```python
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.5,      # Control width
    polynomial_degree=1,
    regularization_lambda=1e-6
)
fitter.fit(X_train, y_train)
```

### 3. Latin Hypercube Sampling for Centers
```python
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    polynomial_degree=1
)

fitter.fit(
    X_train, y_train,
    use_lhs_centers=True,
    n_centers=15,                          # Use 15 centers instead of all data
    lhs_bounds=(np.array([-5, -5]),        # Lower bound
                np.array([5, 5])),         # Upper bound
    random_state=42
)
```

### 4. RBF-Only Fitting (No Polynomials)
```python
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    polynomial_degree=None,        # No polynomial basis
    regularization_lambda=1e-4
)
fitter.fit(X_train, y_train)
```

### 5. Gradient-Informed Fitting
```python
# Gradient must be shape (n_samples, n_features)
dy_train = np.array([
    [df/dx_0, df/dx_1, ...],
    [...],
])

fitter = RBFPolynomialFitter(polynomial_degree=1)
fitter.fit(X_train, y_train, df=dy_train)  # Include gradients

# Predict gradients at new points
dy_pred = fitter.predict_gradient(X_test)
```

### 6. Available Kernels
```python
# Kernels WITH shape parameter (ε)
'multiquadric'           # sqrt(1 + (eps*r)^2)
'gaussian'               # exp(-(eps*r)^2)
'inverse_multiquadric'   # 1/sqrt(1 + (eps*r)^2)

# Kernels WITHOUT shape parameter
'thin_plate_spline'      # r^2 * log(r)
'cubic'                  # r^3
'linear'                 # r
```

### 7. Custom Kernel
```python
def my_kernel(r):
    return np.exp(-r**2)

def my_deriv(r):
    return -2*r * np.exp(-r**2)

fitter = RBFPolynomialFitter(
    rbf_kernel=my_kernel,
    rbf_kernel_derivative=my_deriv,
    polynomial_degree=1
)
```

---

## Parameter Tuning Guide

### Shape Parameter (ε)
| Value | Effect | Best For |
|-------|--------|----------|
| 0.1-0.3 | Very localized | Sparse, well-separated data |
| 0.5 | **Default, balanced** | Most cases |
| 1.0-2.0 | More global | Dense data, smooth functions |
| >2.0 | Very smooth | Highly smooth functions |

### Regularization (λ)
| Value | Effect | Use When |
|-------|--------|----------|
| <1e-6 | Very flexible, may overfit | Clean data, few samples |
| 1e-6 to 1e-4 | **Good default range** | Most cases |
| 1e-3 to 1e-1 | More regularization, underfitting | Noisy data |
| >0.1 | Severe smoothing | Heavy noise or overfitting |

### Polynomial Degree
| Value | Bases (n=2) | Use When |
|-------|------------|----------|
| None | RBF only | Highly oscillatory |
| 0 | 1 (constant) | Add mean + RBF |
| 1 | 4 (constant + linear) | **Most common** |
| 2 | 10 (+ quadratic) | Curved trends |
| ≥3 | Many bases | High polynomial degree needed |

---

## Common Workflows

### Workflow 1: Interpolation with Sparse Data
```python
# Use all data as centers, add polynomial trend
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    rbf_shape_parameter=0.5,
    polynomial_degree=1,
    regularization_lambda=1e-5
)
fitter.fit(X_train, y_train, df=df_train)
y_pred = fitter.predict(X_test)
```

### Workflow 2: High-Dimensional with Many Points
```python
# Use LHS centers to reduce basis functions
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.3,      # Smaller ε
    polynomial_degree=None,       # RBF-only
    regularization_lambda=1e-4
)
fitter.fit(
    X_train, y_train,
    use_lhs_centers=True,
    n_centers=min(50, len(X_train)),
    lhs_bounds=(lower_bounds, upper_bounds),
    random_state=42
)
```

### Workflow 3: Function with Oscillations
```python
# Use Thin Plate Spline for flexibility
fitter = RBFPolynomialFitter(
    rbf_name='thin_plate_spline',
    polynomial_degree=1,          # Capture trend
    regularization_lambda=1e-4    # Moderate regularization
)
fitter.fit(X_train, y_train, df=df_train)
```

### Workflow 4: Noisy Data
```python
# Use larger regularization
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=1.0,      # Smoother
    polynomial_degree=1,
    regularization_lambda=1e-3    # More regularization
)
fitter.fit(X_train, y_train)
```

---

## API Cheat Sheet

### Constructor
```python
RBFPolynomialFitter(
    rbf_name=None,                 # 'gaussian', 'multiquadric', etc.
    rbf_shape_parameter=None,      # ε (if kernel supports it)
    rbf_kernel=None,               # Custom kernel function
    rbf_kernel_derivative=None,    # Custom kernel derivative
    polynomial_degree=1,           # 0, 1, 2, ... or None
    regularization_lambda=1e-6
)
```

### Methods
```python
# Fit the model
fitter.fit(
    X, y,                          # Training data
    df=None,                       # Optional gradients
    centers=None,                  # Explicit centers
    n_centers=None,                # Number of centers
    use_lhs_centers=False,         # Use LHS sampling
    lhs_bounds=None,               # (lower, upper) bounds
    random_state=None              # Random seed
)

# Make predictions
y_pred = fitter.predict(X_test)

# Predict gradients
dy_pred = fitter.predict_gradient(X_test)
```

### Properties
```python
fitter.coefficients           # Fitted coefficients
fitter.centers               # RBF center points
fitter.n_rbf_bases          # Number of RBF bases
fitter.n_features           # Input dimension
fitter.fitted               # Boolean flag
```

---

## Troubleshooting

### Issue: "Design matrix is singular"
**Solution:** Increase `regularization_lambda` (e.g., 1e-5 → 1e-4)

### Issue: Predictions are too smooth
**Solution:** 
- Decrease `regularization_lambda`
- Increase `rbf_shape_parameter` for kernels with ε

### Issue: Overfitting to noise
**Solution:**
- Increase `regularization_lambda`
- Use `polynomial_degree=None` (RBF-only)
- Increase `rbf_shape_parameter`

### Issue: Slow computation with many data points
**Solution:** Use LHS centers with `n_centers < len(X_train)`

### Issue: Poor gradient fit
**Solution:** Increase weight of gradients by using smaller `regularization_lambda`

---

## Performance Tips

1. **Normalize data** to [-1, 1] or [0, 1] before fitting
2. **Scale regularization** with data scale
3. **Use LHS** for high dimensions or many points
4. **Start with ε=0.5**, adjust based on results
5. **Include gradients** if available (improves fit quality)
6. **Use RBF-only** for oscillatory functions
7. **Match kernel to function**: TPS for smooth, Gaussian for smooth+localized

---

## Documentation Files

| File | Purpose |
|------|---------|
| README.md | Complete API documentation |
| FEATURES.md | Detailed feature descriptions |
| IMPLEMENTATION_SUMMARY.md | Implementation details |
| **QUICK_REFERENCE.md** | **This file** |
| examples.py | Visual examples with plots |
| test_new_features.py | Automated tests |
| usage_examples.py | Comprehensive usage scenarios |
