# Radial Basis Function + Polynomial Fitter

A flexible and powerful Python implementation for fitting functions using RBF (Radial Basis Function) kernels combined with polynomial basis functions. Supports both function values and gradient information with L2 regularization.

## Features

- **RBF + Polynomial Basis**: Combines the flexibility of RBF kernels with polynomial basis functions
- **Gradient Support**: Can incorporate gradient information (∂f/∂xᵢ) as fitting constraints for improved accuracy
- **L2 Regularization**: Ridge regression with customizable regularization parameter to prevent overfitting
- **Matrix Inversion**: Uses `np.linalg.inv()` for solving the regularized least-squares problem
- **Built-in RBF Kernels**: Multiple kernels with shape parameter support:
  - Gaussian, Multiquadric, Inverse Multiquadric (with tunable shape parameter ε)
  - Thin Plate Spline, Cubic, Linear (without shape parameters)
- **Latin Hypercube Sampling**: Generate RBF centers using LHS for better coverage instead of using training data points
- **Flexible Polynomial Basis**: Specify polynomial degree (0, 1, 2, ...) or disable entirely with `polynomial_degree=None`
- **Multi-dimensional**: Works with functions f: ℝⁿ → ℝ of arbitrary dimension
- **Gradient Prediction**: Can predict gradients at new points, not just function values

## Installation

```bash
pip install numpy matplotlib scipy
```

## Quick Start

```python
import numpy as np
from rbf_polynomial_fitter import RBFPolynomialFitter

# Define your function and its gradient
def f(X):
    return np.sin(X[:, 0]) + 0.1 * X[:, 1]**2

def df(X):
    # Returns gradient: shape (n_samples, n_features)
    grad = np.zeros((X.shape[0], 2))
    grad[:, 0] = np.cos(X[:, 0])  # df/dx
    grad[:, 1] = 0.2 * X[:, 1]    # df/dy
    return grad

# Generate training data
X_train = np.random.randn(20, 2)
y_train = f(X_train)
dy_train = df(X_train)

# Create and fit the model with built-in Gaussian kernel
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',           # Built-in Gaussian RBF
    rbf_shape_parameter=0.5,       # Shape parameter ε
    polynomial_degree=1,           # Linear polynomial basis
    regularization_lambda=1e-6
)
fitter.fit(X_train, y_train, df=dy_train)

# Make predictions
X_test = np.random.randn(10, 2)
y_pred = fitter.predict(X_test)
dy_pred = fitter.predict_gradient(X_test)
```

### Using Latin Hypercube Sampling for Centers

```python
# Instead of fitting at training data points, use LHS-sampled centers
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    rbf_shape_parameter=0.8,
    polynomial_degree=1,
    regularization_lambda=1e-6
)

fitter.fit(
    X_train, y_train, df=dy_train,
    use_lhs_centers=True,          # Enable LHS center generation
    n_centers=15,                  # Number of centers to generate
    lhs_bounds=(                   # Specify domain bounds
        np.array([-5, -5]),
        np.array([5, 5])
    ),
    random_state=42
)
```

### RBF-Only Fitting (No Polynomials)

```python
# Use only RBF basis, disable polynomial basis
fitter = RBFPolynomialFitter(
    rbf_name='thin_plate_spline',  # TPS has no shape parameter
    polynomial_degree=None,        # No polynomial basis
    regularization_lambda=1e-4
)
fitter.fit(X_train, y_train, df=dy_train)
```

## API Reference

### RBFPolynomialFitter

#### Constructor

```python
RBFPolynomialFitter(
    rbf_kernel=None,
    rbf_kernel_derivative=None,
    rbf_name=None,
    rbf_shape_parameter=None,
    polynomial_degree=1,
    regularization_lambda=1e-6
)
```

**Parameters:**

- **rbf_name** (str, optional): Name of built-in RBF kernel. Options:
  - `'multiquadric'`: √(1 + (ε·r)²) - smooth, default choice
  - `'gaussian'`: exp(-(ε·r)²) - compact support approximation
  - `'inverse_multiquadric'`: 1/√(1 + (ε·r)²) - smooth, bounded
  - `'thin_plate_spline'`: r² log(r) - no shape parameter
  - `'cubic'`: r³ - no shape parameter
  - `'linear'`: r - no shape parameter
  
- **rbf_shape_parameter** (float, optional): Shape parameter (ε) for RBF kernels that support it. Default: 0.5. Ignored for kernels without shape parameter (TPS, cubic, linear).

- **rbf_kernel** (callable, optional): Custom RBF kernel function k(r). Ignored if rbf_name is provided.

- **rbf_kernel_derivative** (callable, optional): Derivative of custom kernel w.r.t. r. Ignored if rbf_name is provided.

- **polynomial_degree** (int or None): Degree of polynomial basis. Options:
  - `None`: No polynomial basis (RBF-only)
  - `0`: Constant term only
  - `1`: Linear polynomial basis (constant + linear terms)
  - `2`: Quadratic basis
  - etc.
  Default: 1

- **regularization_lambda** (float): L2 regularization parameter (λ in ridge regression). Default: 1e-6

#### Methods

**fit(X, f, df=None, centers=None, n_centers=None, use_lhs_centers=False, lhs_bounds=None, random_state=None)**

Fit the model to training data.

**Parameters:**
- **X** (ndarray, shape (n_samples, n_features)): Input training points
- **f** (ndarray, shape (n_samples,)): Function values at training points
- **df** (ndarray, optional, shape (n_samples, n_features)): Gradient values at training points. If provided, these are included as fitting constraints.
- **centers** (ndarray, optional, shape (n_centers, n_features)): Explicit RBF center points. If None, centers are generated based on other parameters.
- **n_centers** (int, optional): Number of centers to use or generate. If None and use_lhs_centers=False, uses all training data as centers.
- **use_lhs_centers** (bool, default=False): If True, generate center points using Latin Hypercube Sampling within lhs_bounds instead of using training data.
- **lhs_bounds** (tuple, optional): Bounds for LHS sampling: (lower, upper) each with shape (n_features,). Required if use_lhs_centers=True.
- **random_state** (int, optional): Random seed for LHS reproducibility.

**Returns:** self (for method chaining)

---

**predict(X)**

Predict function values at new points.

**Parameters:**
- **X** (ndarray, shape (n_eval, n_features)): Points for evaluation

**Returns:** predictions (ndarray, shape (n_eval,))

---

**predict_gradient(X)**

Predict gradients at new points.

**Parameters:**
- **X** (ndarray, shape (n_eval, n_features)): Points for gradient evaluation

**Returns:** gradient (ndarray, shape (n_eval, n_features))

## Mathematical Details

### Problem Formulation

The fitter approximates a function f: ℝⁿ → ℝ using a linear combination of basis functions:

$$\hat{f}(\mathbf{x}) = \sum_{i=1}^{N_{\text{rbf}}} c_i \phi_i(\mathbf{x}) + \sum_{j=1}^{N_{\text{poly}}} w_j p_j(\mathbf{x})$$

where:
- φᵢ are RBF basis functions: φᵢ(x) = k(||x - cᵢ||)
- pⱼ are polynomial basis functions
- c, w are coefficients to be fitted
- cᵢ are RBF centers

### RBF Kernels

**With Shape Parameter (ε):**
- **Multiquadric**: k(r) = √(1 + (εr)²) - smooth, positive definite, default
- **Gaussian**: k(r) = exp(-(εr)²) - infinitely smooth, common choice
- **Inverse Multiquadric**: k(r) = 1/√(1 + (εr)²) - smooth, bounded

**Without Shape Parameter:**
- **Thin Plate Spline**: k(r) = r² log(r) - common in 2D, conditionally positive definite
- **Cubic**: k(r) = r³ - conditionally positive definite
- **Linear**: k(r) = r - simple, linear growth

The shape parameter ε controls the "width" of the kernel:
- Small ε: more localized, sharper features
- Large ε: more global influence, smoother interpolation

### Polynomial Bases

All monomials up to degree d are included. For degree d in n dimensions:
- degree 0: 1 (constant)
- degree 1: x₀, x₁, ..., xₙ₋₁
- degree 2: x₀², x₀x₁, ..., xₙ₋₁²
- etc.

Number of polynomial bases for degree d in n dimensions: $\binom{n+d}{d}$

### Fitting with Gradients

When gradient data is provided, the system includes both function value constraints and gradient constraints:

$$\text{minimize} \; ||D\mathbf{c} - \mathbf{y}||^2 + \lambda||\mathbf{c}||^2$$

where:
- D is the design matrix containing both function and gradient basis evaluations
- y contains both function values and gradient components
- λ is the regularization parameter

### L2 Regularization

The regularized least-squares solution is computed as:

$$\mathbf{c} = (D^T D + \lambda I)^{-1} D^T \mathbf{y}$$

This is solved via matrix inversion using `np.linalg.inv()`.

## Custom RBF Kernels

You can use custom RBF kernels by providing both the kernel and its derivative:

```python
def custom_kernel(r):
    """Custom RBF kernel as function of distance r."""
    return np.exp(-r**2)  # Gaussian RBF

def custom_derivative(r):
    """Derivative of kernel w.r.t. r."""
    return -2*r * np.exp(-r**2)

fitter = RBFPolynomialFitter(
    rbf_kernel=custom_kernel,
    rbf_kernel_derivative=custom_derivative,
    polynomial_degree=1
)
```

### Using Built-in Kernels with Shape Parameters

```python
# Gaussian with custom shape parameter
fitter = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.8,  # Tune width
    polynomial_degree=1
)

# Thin Plate Spline (no shape parameter needed)
fitter = RBFPolynomialFitter(
    rbf_name='thin_plate_spline',
    polynomial_degree=1
)
```

## Regularization Tips

- **Too small λ** (e.g., 1e-10): May overfit, especially with gradients
- **Too large λ** (e.g., 1e-1): May underfit, loses information
- **Good starting point**: 1e-6 to 1e-4 depending on data scale and noise
- **Increase λ** if predictions are noisy or oscillatory
- **Decrease λ** if predictions are too smooth and miss important features

## Shape Parameter (ε) Tips

- **Small ε** (0.1-0.3): RBF becomes more localized, may need regularization
- **Medium ε** (0.5-1.0): Standard choice, good balance
- **Large ε** (2.0+): RBF becomes more global, may smooth out features
- **Tuning**: Start with ε=0.5, adjust based on prediction quality and data density

## Latin Hypercube Sampling (LHS) vs Training Data Centers

**Training Data as Centers:**
- Uses actual data points where function is evaluated
- Can interpolate training points exactly (with small λ)
- Number of RBF bases = number of training points
- Good when training points are well-distributed

**LHS-Generated Centers:**
- Generates centers on a uniform grid with randomization
- Decouples center locations from training data
- Can use fewer centers (computational savings)
- Better for sparse or non-uniform training data
- Can reduce overfitting with fewer parameters

```python
# Training data centers (default)
fitter.fit(X_train, y_train, df=df_train)

# LHS centers
fitter.fit(
    X_train, y_train, df=df_train,
    use_lhs_centers=True,
    n_centers=20,
    lhs_bounds=(lower_bound, upper_bound),
    random_state=42
)
```

## Examples

See `examples.py` for complete examples including:
1. 1D function fitting with gradients
2. 2D function fitting with gradients
3. Fitting without gradient information
4. Comparison of different RBF kernels
5. **Latin Hypercube Sampling for center generation** (new!)
6. **Effect of shape parameters on different kernels** (new!)
7. **RBF-only fitting without polynomials** (new!)

Run examples:
```bash
python examples.py
```

This generates plots: 
- `1d_fit.png`, `2d_fit.png`, `no_gradients.png` (original examples)
- `kernel_comparison.png` (custom kernels)
- `lhs_centers.png` (LHS center generation)
- `shape_parameters.png` (shape parameter effects)
- `rbf_only_comparison.png` (RBF-only vs RBF+polynomial)

## Performance Considerations

- **Computational Complexity**: O(n³) for n basis functions (matrix inversion)
- **Memory**: O(n²) for storing design and Gram matrices
- For large datasets, consider:
  - Using a subset of X as centers
  - Increasing regularization_lambda
  - Using lower polynomial degree

## References

- Buhmann, M. D. (2003). "Radial Basis Functions: Theory and Implementations"
- Fornberg, B., & Flyer, N. (2015). "A Primer on Radial Basis Functions"
- Wendland, H. (2005). "Scattered Data Approximation"

## License

MIT License - feel free to use and modify
