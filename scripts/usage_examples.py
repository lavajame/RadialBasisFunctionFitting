"""
Comprehensive Usage Examples - RBF + Polynomial Fitter

Demonstrates all key features:
- Multiple built-in RBF kernels
- Shape parameter tuning
- Latin Hypercube Sampling for centers
- RBF-only vs RBF+polynomial fitting
- Gradient-informed fitting
"""

import numpy as np
from rbf_polynomial_fitter import RBFPolynomialFitter


print("=" * 80)
print("RBF + Polynomial Fitter - Comprehensive Usage Examples")
print("=" * 80)


# ============================================================================
# Example 1: Using Built-in Kernels with Shape Parameters
# ============================================================================
print("\n1. Using Built-in Kernels with Shape Parameters")
print("-" * 80)

X_train = np.random.uniform(-2, 2, (20, 2))
y_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])

# Gaussian kernel with epsilon=0.5
fitter_gaussian = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.5,
    polynomial_degree=1,
    regularization_lambda=1e-5
)
fitter_gaussian.fit(X_train, y_train)
print(f"✓ Gaussian kernel with ε=0.5 fitted successfully")
print(f"  Centers shape: {fitter_gaussian.centers.shape}")
print(f"  Coefficients shape: {fitter_gaussian.coefficients.shape}")

# Thin Plate Spline (no shape parameter)
fitter_tps = RBFPolynomialFitter(
    rbf_name='thin_plate_spline',
    polynomial_degree=1,
    regularization_lambda=1e-4
)
fitter_tps.fit(X_train, y_train)
print(f"✓ Thin Plate Spline kernel fitted successfully")
print(f"  (Note: TPS has no shape parameter)")


# ============================================================================
# Example 2: Effect of Shape Parameter on Fit Quality
# ============================================================================
print("\n2. Effect of Shape Parameter (Epsilon)")
print("-" * 80)

X_test = np.random.uniform(-2, 2, (100, 2))
y_test = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1])

epsilons = [0.2, 0.5, 1.0, 2.0]
for eps in epsilons:
    fitter = RBFPolynomialFitter(
        rbf_name='multiquadric',
        rbf_shape_parameter=eps,
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter.fit(X_train, y_train)
    y_pred = fitter.predict(X_test)
    mse = np.mean((y_pred - y_test)**2)
    print(f"  ε = {eps:.1f}: MSE = {mse:.6e}")


# ============================================================================
# Example 3: Latin Hypercube Sampling for Center Generation
# ============================================================================
print("\n3. Latin Hypercube Sampling (LHS) for Centers")
print("-" * 80)

# Fit with LHS-generated centers
fitter_lhs = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.5,
    polynomial_degree=1,
    regularization_lambda=1e-5
)

# Generate 12 LHS centers instead of using all 20 training points
fitter_lhs.fit(
    X_train, y_train,
    use_lhs_centers=True,
    n_centers=12,
    lhs_bounds=(np.array([-2, -2]), np.array([2, 2])),
    random_state=42
)
print(f"✓ Fitted with LHS-generated centers")
print(f"  Training points: {X_train.shape[0]}")
print(f"  Generated centers: {fitter_lhs.centers.shape[0]}")
print(f"  Center ranges: x ∈ [{fitter_lhs.centers[:, 0].min():.2f}, {fitter_lhs.centers[:, 0].max():.2f}]")
print(f"                 y ∈ [{fitter_lhs.centers[:, 1].min():.2f}, {fitter_lhs.centers[:, 1].max():.2f}]")

# Compare with training data as centers
fitter_data = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.5,
    polynomial_degree=1,
    regularization_lambda=1e-5
)
fitter_data.fit(X_train, y_train)  # Uses all X_train as centers
print(f"\n  LHS centers used: {fitter_lhs.n_rbf_bases}")
print(f"  Data centers used: {fitter_data.n_rbf_bases}")


# ============================================================================
# Example 4: RBF-Only vs RBF+Polynomial Fitting
# ============================================================================
print("\n4. RBF-Only vs RBF+Polynomial Fitting")
print("-" * 80)

X = np.random.uniform(-3, 3, (25, 1))
y = np.sin(2*X).ravel() + 0.05*X.ravel()  # Oscillatory + linear trend

# RBF-only (no polynomials)
fitter_rbf_only = RBFPolynomialFitter(
    rbf_name='gaussian',
    polynomial_degree=None,  # No polynomial basis
    regularization_lambda=1e-4
)
fitter_rbf_only.fit(X, y)
y_pred_rbf = fitter_rbf_only.predict(X)
mse_rbf = np.mean((y_pred_rbf - y)**2)

# RBF + Linear polynomial
fitter_hybrid = RBFPolynomialFitter(
    rbf_name='gaussian',
    polynomial_degree=1,  # Linear trend
    regularization_lambda=1e-4
)
fitter_hybrid.fit(X, y)
y_pred_hybrid = fitter_hybrid.predict(X)
mse_hybrid = np.mean((y_pred_hybrid - y)**2)

# RBF + Quadratic polynomial
fitter_quad = RBFPolynomialFitter(
    rbf_name='gaussian',
    polynomial_degree=2,  # Quadratic trend
    regularization_lambda=1e-4
)
fitter_quad.fit(X, y)
y_pred_quad = fitter_quad.predict(X)
mse_quad = np.mean((y_pred_quad - y)**2)

print(f"✓ RBF-only (no polynomials): MSE = {mse_rbf:.6e}")
print(f"✓ RBF + Linear polynomial:   MSE = {mse_hybrid:.6e}")
print(f"✓ RBF + Quadratic polynomial: MSE = {mse_quad:.6e}")


# ============================================================================
# Example 5: Gradient-Informed Fitting with Custom Kernels
# ============================================================================
print("\n5. Gradient-Informed Fitting")
print("-" * 80)

# Define test function and gradient
def f_func(X):
    return np.sin(X[:, 0]) * np.cos(X[:, 1])

def df_func(X):
    grad = np.zeros_like(X)
    grad[:, 0] = np.cos(X[:, 0]) * np.cos(X[:, 1])
    grad[:, 1] = -np.sin(X[:, 0]) * np.sin(X[:, 1])
    return grad

X_train = np.random.uniform(-np.pi, np.pi, (15, 2))
y_train = f_func(X_train)
dy_train = df_func(X_train)

# Fit with gradient information
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    rbf_shape_parameter=0.8,
    polynomial_degree=1,
    regularization_lambda=1e-5
)
fitter.fit(X_train, y_train, df=dy_train)
print(f"✓ Fitted with gradient information")
print(f"  Total constraints used: {X_train.shape[0]} (values) + {X_train.shape[0]*2} (gradients)")

# Predict at new points
X_test = np.random.uniform(-np.pi, np.pi, (5, 2))
y_pred = fitter.predict(X_test)
dy_pred = fitter.predict_gradient(X_test)
print(f"✓ Predictions and gradients computed")
print(f"  Predicted function values shape: {y_pred.shape}")
print(f"  Predicted gradients shape: {dy_pred.shape}")


# ============================================================================
# Example 6: Comparing All Built-in Kernels
# ============================================================================
print("\n6. Comparison of All Built-in Kernels")
print("-" * 80)

X = np.random.uniform(-2, 2, (15, 1))
y = np.exp(-X**2).ravel()  # Smooth Gaussian function

kernels = {
    'multiquadric': {'shape': 0.5},
    'gaussian': {'shape': 0.5},
    'inverse_multiquadric': {'shape': 0.5},
    'thin_plate_spline': {},
    'cubic': {},
    'linear': {}
}

for kernel_name, params in kernels.items():
    fitter = RBFPolynomialFitter(
        rbf_name=kernel_name,
        rbf_shape_parameter=params.get('shape'),
        polynomial_degree=1,
        regularization_lambda=1e-4
    )
    fitter.fit(X, y)
    y_pred = fitter.predict(X)
    mse = np.mean((y_pred - y)**2)
    shape_str = f"ε={params['shape']}" if 'shape' in params else "(no shape)"
    print(f"  {kernel_name:20s} {shape_str:15s}: MSE = {mse:.6e}")


# ============================================================================
# Example 7: Tuning Regularization with LHS Centers
# ============================================================================
print("\n7. Regularization Tuning with LHS Centers")
print("-" * 80)

X_train = np.random.uniform(-1, 1, (30, 2))
y_train = np.sin(2*np.pi*X_train[:, 0]) + np.cos(2*np.pi*X_train[:, 1])

lambdas = [1e-7, 1e-5, 1e-3, 1e-1]
print(f"Regularization parameter (lambda) effect with LHS centers:")

for lam in lambdas:
    fitter = RBFPolynomialFitter(
        rbf_name='gaussian',
        rbf_shape_parameter=0.5,
        polynomial_degree=1,
        regularization_lambda=lam
    )
    fitter.fit(
        X_train, y_train,
        use_lhs_centers=True,
        n_centers=12,
        lhs_bounds=(np.array([-1, -1]), np.array([1, 1])),
        random_state=42
    )
    y_pred = fitter.predict(X_train)
    mse = np.mean((y_pred - y_train)**2)
    print(f"  λ = {lam:.0e}: MSE = {mse:.6e}")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary of Features Demonstrated:")
print("=" * 80)
print("""
✓ Built-in RBF kernels with shape parameters (Gaussian, Multiquadric, etc.)
✓ Kernels without shape parameters (Thin Plate Spline, Cubic, Linear)
✓ Shape parameter tuning for controlling fit properties
✓ Latin Hypercube Sampling (LHS) for intelligent center generation
✓ Comparison of LHS vs training data as centers
✓ RBF-only fitting (polynomial_degree=None)
✓ RBF + polynomial fitting with various degrees
✓ Gradient-informed fitting (incorporating df/dx)
✓ Gradient prediction at new points
✓ Regularization tuning
✓ Full API demonstrated across multiple scenarios

For full documentation, see:
- README.md: Complete API reference and background
- FEATURES.md: Detailed feature descriptions
- examples.py: Visual examples with plots
- test_new_features.py: Automated tests for all features
""")
print("=" * 80)
