# RBF + Polynomial Fitter - Feature Summary

## New Features Added

### 1. RBF Shape Parameters

All built-in RBF kernels with shape parameter support now include a tunable **epsilon (ε)** parameter that controls the "width" or "locality" of the kernel:

**Available Kernels:**

| Kernel | Formula | Has Shape Parameter | Default ε |
|--------|---------|---------------------|-----------|
| Multiquadric | √(1 + (εr)²) | ✓ Yes | 0.5 |
| Gaussian | exp(-(εr)²) | ✓ Yes | 0.5 |
| Inverse Multiquadric | 1/√(1 + (εr)²) | ✓ Yes | 0.5 |
| Thin Plate Spline | r² log(r) | ✗ No | N/A |
| Cubic | r³ | ✗ No | N/A |
| Linear | r | ✗ No | N/A |

**Usage:**
```python
# Use built-in kernel with shape parameter
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.8,  # Tune width
    polynomial_degree=1
)
```

### 2. Latin Hypercube Sampling (LHS) for Centers

Instead of using training data points as RBF centers, you can now generate centers using Latin Hypercube Sampling for better domain coverage.

**When to Use:**
- Training data is sparse or non-uniformly distributed
- Want to reduce number of RBF bases (computational savings)
- Need better coverage of the input domain

**Usage:**
```python
fitter.fit(
    X_train, y_train, df=df_train,
    use_lhs_centers=True,
    n_centers=15,
    lhs_bounds=(np.array([-5, -5]), np.array([5, 5])),
    random_state=42
)
```

### 3. RBF-Only Fitting (No Polynomials)

Disable polynomial basis by setting `polynomial_degree=None` for RBF-only fitting.

**When to Use:**
- Function is highly oscillatory (polynomials don't help)
- Want to reduce number of basis functions
- RBF alone is sufficient for the problem

**Usage:**
```python
# RBF-only, no polynomial basis
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    polynomial_degree=None,  # No polynomials
    regularization_lambda=1e-4
)
```

---

## Implementation Details

### Modified Files

**rbf_polynomial_fitter.py:**
- Added `BUILTIN_KERNELS` dictionary with 6 pre-configured kernels
- New `_setup_builtin_kernel()` method for kernel initialization
- Enhanced `__init__()` with `rbf_name` and `rbf_shape_parameter` parameters
- Extended `fit()` method with LHS center generation parameters
- Updated `_count_polynomial_bases()` to handle `None` case
- Updated polynomial basis methods to handle `None` polynomial_degree

**examples.py:**
- Test 5: Latin Hypercube Sampling comparison (LHS vs training data centers)
- Test 6: Shape parameter effects across different kernels
- Test 7: RBF-only vs RBF+polynomial comparison

**README.md:**
- Comprehensive kernel documentation with formulas and default parameters
- LHS and shape parameter usage examples
- Tips for tuning shape parameters and choosing between LHS/training data centers
- Updated API documentation for new parameters

---

## API Changes

### Constructor Parameters (New)

```python
RBFPolynomialFitter(
    ...
    rbf_name: Optional[str] = None,           # NEW
    rbf_shape_parameter: Optional[float] = None,  # NEW
    polynomial_degree: Optional[int] = 1,     # CHANGED: now accepts None
    ...
)
```

### Fit Method Parameters (New)

```python
def fit(
    self,
    X, f, df=None,
    centers=None,
    n_centers=None,                   # NEW
    use_lhs_centers: bool = False,    # NEW
    lhs_bounds=None,                  # NEW
    random_state: Optional[int] = None,  # NEW
):
    ...
```

---

## Backward Compatibility

✓ **Fully backward compatible** - all existing code continues to work without modification.

**Examples:**
```python
# Old code still works
fitter = RBFPolynomialFitter(polynomial_degree=1)
fitter.fit(X, y)

# New features are opt-in
fitter = RBFPolynomialFitter(rbf_name='gaussian', polynomial_degree=None)
fitter.fit(X, y, use_lhs_centers=True, n_centers=10, lhs_bounds=(lower, upper))
```

---

## Testing

All new features are tested in `test_new_features.py`:
- ✓ Shape parameters on all kernels
- ✓ LHS center generation with bounds
- ✓ RBF-only fitting (polynomial_degree=None)
- ✓ All built-in kernels work correctly

Run tests:
```bash
python test_new_features.py
```

---

## Example Use Cases

### Case 1: High-dimensional function with sparse data
```python
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.3,  # Smaller epsilon for localized fits
    polynomial_degree=None,    # RBF-only
    regularization_lambda=1e-4
)
fitter.fit(X_train, y_train, df=df_train,
          use_lhs_centers=True, n_centers=50,
          lhs_bounds=(lower, upper))
```

### Case 2: Smooth function with good data coverage
```python
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    rbf_shape_parameter=1.0,   # Larger epsilon for global fits
    polynomial_degree=1,        # Add linear polynomial
    regularization_lambda=1e-6
)
fitter.fit(X_train, y_train, df=df_train)  # Use training data as centers
```

### Case 3: Highly oscillatory function
```python
fitter = RBFPolynomialFitter(
    rbf_name='thin_plate_spline',
    polynomial_degree=1,        # Polynomial for trend
    regularization_lambda=1e-5
)
fitter.fit(X_train, y_train, df=df_train)  # Many centers needed
```
