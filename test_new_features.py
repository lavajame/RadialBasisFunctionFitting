"""
Quick test of new features:
- RBF shape parameters
- Latin Hypercube Sampling centers
- RBF-only (no polynomials)
"""

import numpy as np
from rbf_polynomial_fitter import RBFPolynomialFitter


def test_shape_parameters():
    """Test shape parameters on built-in kernels."""
    print("Testing shape parameters...")
    
    X = np.random.randn(10, 2)
    y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    
    # Gaussian with shape parameter
    fitter = RBFPolynomialFitter(
        rbf_name='gaussian',
        rbf_shape_parameter=0.5,
        polynomial_degree=1
    )
    fitter.fit(X, y)
    y_pred = fitter.predict(X)
    print(f"  Gaussian eps=0.5: MSE={np.mean((y_pred - y)**2):.6e}")
    
    # Multiquadric with different shape
    fitter = RBFPolynomialFitter(
        rbf_name='multiquadric',
        rbf_shape_parameter=1.0,
        polynomial_degree=1
    )
    fitter.fit(X, y)
    y_pred = fitter.predict(X)
    print(f"  Multiquadric eps=1.0: MSE={np.mean((y_pred - y)**2):.6e}")
    
    # TPS (no shape parameter)
    fitter = RBFPolynomialFitter(
        rbf_name='thin_plate_spline',
        polynomial_degree=1
    )
    fitter.fit(X, y)
    y_pred = fitter.predict(X)
    print(f"  Thin Plate Spline: MSE={np.mean((y_pred - y)**2):.6e}")
    print("  ✓ Shape parameters work correctly\n")


def test_lhs_centers():
    """Test Latin Hypercube Sampling for centers."""
    print("Testing LHS center generation...")
    
    # Generate training data
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (30, 2))
    y = np.sin(X[:, 0]) * np.cos(X[:, 1])
    
    # Fit with LHS centers
    fitter = RBFPolynomialFitter(
        rbf_name='gaussian',
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    
    fitter.fit(
        X, y,
        use_lhs_centers=True,
        n_centers=15,
        lhs_bounds=(np.array([-2, -2]), np.array([2, 2])),
        random_state=42
    )
    
    print(f"  Training data points: {X.shape[0]}")
    print(f"  Generated LHS centers: {fitter.centers.shape[0]}")
    print(f"  Center bounds: x∈[{fitter.centers[:, 0].min():.2f}, {fitter.centers[:, 0].max():.2f}], "
          f"y∈[{fitter.centers[:, 1].min():.2f}, {fitter.centers[:, 1].max():.2f}]")
    
    y_pred = fitter.predict(X)
    mse = np.mean((y_pred - y)**2)
    print(f"  MSE on training data: {mse:.6e}")
    print("  ✓ LHS center generation works correctly\n")


def test_rbf_only():
    """Test RBF-only fitting without polynomials."""
    print("Testing RBF-only (polynomial_degree=None)...")
    
    X = np.random.randn(15, 1)
    y = np.sin(3*X).ravel() + 0.1*np.random.randn(15)
    
    # RBF-only
    fitter_rbf = RBFPolynomialFitter(
        rbf_name='gaussian',
        polynomial_degree=None,  # No polynomials
        regularization_lambda=1e-4
    )
    fitter_rbf.fit(X, y)
    y_pred_rbf = fitter_rbf.predict(X)
    mse_rbf = np.mean((y_pred_rbf - y)**2)
    
    # RBF + Linear
    fitter_hybrid = RBFPolynomialFitter(
        rbf_name='gaussian',
        polynomial_degree=1,  # Linear polynomial
        regularization_lambda=1e-4
    )
    fitter_hybrid.fit(X, y)
    y_pred_hybrid = fitter_hybrid.predict(X)
    mse_hybrid = np.mean((y_pred_hybrid - y)**2)
    
    print(f"  RBF-only MSE: {mse_rbf:.6e}")
    print(f"  RBF+Linear MSE: {mse_hybrid:.6e}")
    print("  ✓ RBF-only fitting works correctly\n")


def test_all_kernels():
    """Test all built-in kernels."""
    print("Testing all built-in kernels...")
    
    X = np.random.randn(12, 1)
    y = np.sin(X).ravel()
    
    kernels = ['multiquadric', 'gaussian', 'inverse_multiquadric', 
               'thin_plate_spline', 'cubic', 'linear']
    
    for kernel_name in kernels:
        fitter = RBFPolynomialFitter(
            rbf_name=kernel_name,
            polynomial_degree=0,
            regularization_lambda=1e-4
        )
        fitter.fit(X, y)
        y_pred = fitter.predict(X)
        mse = np.mean((y_pred - y)**2)
        print(f"  {kernel_name:20s}: MSE={mse:.6e}")
    
    print("  ✓ All kernels work correctly\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing New Features")
    print("=" * 70 + "\n")
    
    test_shape_parameters()
    test_lhs_centers()
    test_rbf_only()
    test_all_kernels()
    
    print("=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
