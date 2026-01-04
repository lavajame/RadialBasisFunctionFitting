# RBF + Polynomial Fitter - Project Index

## Overview

A sophisticated Python implementation of a **Radial Basis Function + Polynomial Fitter** with L2 regularization for approximating functions f: ℝⁿ → ℝ from sampled data.

**Key Features:**
- ✓ 6 built-in RBF kernels with tunable shape parameters
- ✓ Latin Hypercube Sampling for intelligent center generation
- ✓ Gradient-informed fitting using both f(x) and ∇f(x)
- ✓ Flexible polynomial basis (constant/linear/quadratic/... or RBF-only)
- ✓ L2 regularization via matrix inversion with `np.linalg.inv()`
- ✓ Multi-dimensional support (works with any dimension)

---

## Quick Start

```python
from rbf_polynomial_fitter import RBFPolynomialFitter
import numpy as np

# Create and fit model
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.5,
    polynomial_degree=1,
    regularization_lambda=1e-6
)

# Fit to training data (with optional gradients)
fitter.fit(X_train, y_train, df=dy_train)

# Make predictions
y_pred = fitter.predict(X_test)
dy_pred = fitter.predict_gradient(X_test)
```

---

## Project Structure

```
RadialBasisFunctionFitting/
├── rbf_polynomial_fitter.py          # Core implementation
├── examples.py                       # Visual examples with plots
├── test_new_features.py             # Automated tests
├── usage_examples.py                # Comprehensive usage scenarios
│
├── README.md                        # Full API documentation
├── FEATURES.md                      # Detailed feature descriptions
├── QUICK_REFERENCE.md              # Quick reference guide
├── IMPLEMENTATION_SUMMARY.md        # Implementation details
└── PROJECT_INDEX.md                # This file
```

---

## Documentation Guide

### For First-Time Users
1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Start here!
   - Installation & setup
   - Common examples
   - Parameter tuning guide
   - Troubleshooting

2. **[examples.py](examples.py)** - Run to see visual examples
   ```bash
   python examples.py  # Generates plots
   ```

### For Detailed Reference
3. **[README.md](README.md)** - Complete documentation
   - Full API reference
   - Mathematical background
   - Available kernels
   - Tips for regularization and shape parameters

4. **[FEATURES.md](FEATURES.md)** - Feature deep dive
   - Shape parameter details
   - LHS center generation
   - RBF-only fitting
   - Use cases and workflows

### For Implementation Details
5. **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Technical details
   - File modifications
   - Code changes summary
   - Testing information
   - Performance considerations

---

## Core Functionality

### Available RBF Kernels

| Kernel | Formula | Shape Param | Best For |
|--------|---------|-------------|----------|
| Gaussian | exp(-(εr)²) | ✓ | Smooth, well-behaved functions |
| Multiquadric | √(1+(εr)²) | ✓ | General purpose (default) |
| Inverse Multiquadric | 1/√(1+(εr)²) | ✓ | Bounded, smooth functions |
| Thin Plate Spline | r² log(r) | ✗ | Scattered data, high dimensions |
| Cubic | r³ | ✗ | Smooth approximation |
| Linear | r | ✗ | Simple, fast |

### Design Matrix & Solving

The fitter constructs an augmented design matrix D that includes:
- RBF basis evaluations: φᵢ(x) = k(‖x - cᵢ‖)
- Polynomial basis: x₀, x₁, ..., x₀², ... (up to degree d)
- Gradient constraints: ∂φᵢ/∂xⱼ, ∂p/∂xⱼ (when df provided)

Solves via L2-penalized regression:
$$\mathbf{c} = (D^T D + \lambda I)^{-1} D^T \mathbf{y}$$

---

## Feature Comparison

### Feature 1: Shape Parameters (ε)

**Constructor:**
```python
RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.5
)
```

**Effect on fit:**
- ε=0.1: Highly localized, sharp features
- ε=0.5: Balanced (default)
- ε=2.0: Global, smoother curves

### Feature 2: Latin Hypercube Sampling

**Usage:**
```python
fitter.fit(
    X_train, y_train,
    use_lhs_centers=True,
    n_centers=15,
    lhs_bounds=(lower, upper),
    random_state=42
)
```

**Benefits:**
- ✓ Better domain coverage than data points
- ✓ Fewer RBF bases (15 centers vs 100 data points)
- ✓ Decouples from training data distribution
- ✓ Reproducible with random_state

### Feature 3: Flexible Polynomial Basis

**Options:**
```python
polynomial_degree=None  # RBF only, no polynomials
polynomial_degree=0     # Constant term
polynomial_degree=1     # Linear polynomial (default)
polynomial_degree=2     # Quadratic polynomial
```

**When to use:**
- RBF-only: Oscillatory functions
- Linear: Most common, captures trend
- Quadratic: Curved trends
- None: Space-saving, RBF is sufficient

---

## Example Workflows

### Workflow 1: Standard Interpolation
```python
fitter = RBFPolynomialFitter(polynomial_degree=1)
fitter.fit(X_train, y_train, df=df_train)
```
✓ Minimal setup, works for most problems

### Workflow 2: High-Dimensional Data
```python
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    polynomial_degree=None,
    regularization_lambda=1e-4
)
fitter.fit(
    X_train, y_train, df=df_train,
    use_lhs_centers=True,
    n_centers=30,
    lhs_bounds=(lower, upper)
)
```
✓ LHS reduces basis functions, RBF-only saves parameters

### Workflow 3: Noisy Data
```python
fitter = RBFPolynomialFitter(
    rbf_shape_parameter=1.0,      # Larger ε
    polynomial_degree=1,
    regularization_lambda=1e-2    # Strong regularization
)
fitter.fit(X_train, y_train)
```
✓ Larger regularization smooths noise

### Workflow 4: Custom Kernel
```python
fitter = RBFPolynomialFitter(
    rbf_kernel=my_kernel,
    rbf_kernel_derivative=my_deriv,
    polynomial_degree=2
)
fitter.fit(X_train, y_train, df=df_train)
```
✓ Full control over kernel design

---

## Testing & Validation

### Automated Tests
```bash
# Run comprehensive test suite
python test_new_features.py
```

Tests cover:
- ✓ All 6 built-in kernels
- ✓ Shape parameter effects
- ✓ LHS center generation
- ✓ RBF-only fitting
- ✓ Gradient constraints
- ✓ Backward compatibility

### Demonstration Scripts
```bash
# Run visual examples
python examples.py

# Run usage demonstrations
python usage_examples.py
```

---

## API Quick Reference

### Initialization
```python
fitter = RBFPolynomialFitter(
    rbf_name='gaussian'|'multiquadric'|'thin_plate_spline'|...,
    rbf_shape_parameter=0.5,              # For supported kernels
    polynomial_degree=None|0|1|2|...,    # None = RBF only
    regularization_lambda=1e-6
)
```

### Fitting
```python
fitter.fit(
    X, y,                                 # Training data
    df=None,                             # Optional gradients
    use_lhs_centers=False,               # Use LHS sampling
    n_centers=None,                      # Number of centers
    lhs_bounds=None,                     # LHS domain bounds
    random_state=None
)
```

### Prediction
```python
y_pred = fitter.predict(X_test)
dy_pred = fitter.predict_gradient(X_test)
```

---

## Performance

### Computational Complexity
- Design matrix construction: O(nm + n²d) where n=samples, m=features, d=basis functions
- Matrix inversion: O(d³)
- Prediction: O(nd)

### Memory Usage
- Design matrix: O(nm + md²)
- Gram matrix: O(d²)
- Total: O(d²) where d ≈ n_rbf + n_poly

### Optimization Tips
- Use LHS to reduce d
- Use polynomial_degree=None for fewer bases
- Use larger regularization_lambda (faster convergence)
- Normalize input data to [-1, 1]

---

## Mathematical Background

### RBF Interpolation
$$f(x) \approx \sum_{i=1}^n c_i k(\|x - x_i\|) + \sum_{j=1}^m w_j p_j(x)$$

### Design Matrix (with gradients)
$$D = \begin{bmatrix} \phi(X) & P(X) \\ \nabla_x\phi(X) & \nabla_x P(X) \end{bmatrix}$$

### L2-Penalized Solution
$$\hat{c} = (D^T D + \lambda I)^{-1} D^T y$$

---

## Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| Singular matrix | Ill-posed | Increase λ |
| Overfitting | Too many bases | Increase λ, use RBF-only |
| Underfitting | Over-regularized | Decrease λ |
| Slow computation | Too many bases | Use LHS, reduce polynomial degree |
| Oscillatory predictions | Poor parameter choice | Tune ε and λ |

---

## References

Useful papers and resources:
- Wendland, H. (2005). "Scattered Data Approximation"
- Fornberg, B., & Flyer, N. (2015). "A Primer on Radial Basis Functions"
- Buhmann, M. D. (2003). "Radial Basis Functions: Theory and Implementations"

---

## File Sizes & Structure

```
Main Code:
- rbf_polynomial_fitter.py     ~500 lines  (core implementation)
- examples.py                  ~300 lines  (demonstrations)
- test_new_features.py         ~150 lines  (tests)
- usage_examples.py            ~250 lines  (usage examples)

Documentation:
- README.md                    ~400 lines  (full reference)
- FEATURES.md                  ~300 lines  (feature details)
- QUICK_REFERENCE.md          ~200 lines  (quick guide)
- IMPLEMENTATION_SUMMARY.md    ~400 lines  (technical details)
- PROJECT_INDEX.md            ~300 lines  (this file)
```

---

## Getting Help

1. **Quick questions?** → Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **How do I...?** → See [examples.py](examples.py)
3. **API details?** → Read [README.md](README.md)
4. **Implementation?** → See [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
5. **Testing?** → Run [test_new_features.py](test_new_features.py)

---

## Summary

This project provides a **production-ready RBF + polynomial fitter** with:

✓ Multiple kernel choices  
✓ Tunable shape parameters  
✓ Intelligent center generation (LHS)  
✓ Gradient-informed fitting  
✓ Flexible polynomial basis  
✓ L2 regularization via matrix inversion  
✓ Full documentation & tests  
✓ Backward compatible API  

**Ready to use!** Start with [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
