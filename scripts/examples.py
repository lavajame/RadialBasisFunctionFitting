"""
Example usage and tests for the RBF + Polynomial Fitter

Demonstrates:
- Built-in RBF kernels with shape parameters
- Latin Hypercube Sampling for center generation
- Polynomial degree control (including None for RBF-only)
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_polynomial_fitter import RBFPolynomialFitter


def test_1d_function():
    """Test fitting a 1D function with function values and gradients."""
    print("=" * 60)
    print("Test 1: 1D Function with Values and Gradients")
    print("=" * 60)
    
    # Define test function: f(x) = sin(x) + 0.1*x^2
    def f(x):
        return np.sin(x) + 0.1 * x**2
    
    def df_dx(x):
        return np.cos(x) + 0.2 * x
    
    # Generate training data
    np.random.seed(42)
    x_train = np.linspace(-2*np.pi, 2*np.pi, 15).reshape(-1, 1)
    f_train = f(x_train).ravel()
    df_train = df_dx(x_train).reshape(-1, 1)
    
    # Fit model with gradients
    fitter = RBFPolynomialFitter(
        polynomial_degree=2,
        regularization_lambda=1e-5
    )
    fitter.fit(x_train, f_train, df=df_train)
    
    # Evaluate on test points
    x_test = np.linspace(-2*np.pi, 2*np.pi, 100).reshape(-1, 1)
    f_pred = fitter.predict(x_test)
    df_pred = fitter.predict_gradient(x_test)
    
    f_true = f(x_test).ravel()
    df_true = df_dx(x_test).ravel()
    
    # Compute errors
    func_error = np.mean((f_pred - f_true)**2)
    grad_error = np.mean((df_pred.ravel() - df_true)**2)
    
    print(f"Function MSE: {func_error:.6e}")
    print(f"Gradient MSE: {grad_error:.6e}")
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    axes[0].plot(x_test, f_true, 'b-', label='True function', linewidth=2)
    axes[0].plot(x_test, f_pred, 'r--', label='RBF+Poly prediction', linewidth=2)
    axes[0].scatter(x_train, f_train, color='green', s=50, label='Training points', zorder=5)
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Function Fitting: 1D Example')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_test, df_true, 'b-', label='True gradient', linewidth=2)
    axes[1].plot(x_test, df_pred, 'r--', label='RBF+Poly gradient', linewidth=2)
    axes[1].scatter(x_train, df_train, color='green', s=50, label='Training gradients', zorder=5)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('df/dx')
    axes[1].set_title('Gradient Fitting: 1D Example')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1d_fit.png', dpi=150)
    print("Saved plot to 1d_fit.png\n")


def test_2d_function():
    """Test fitting a 2D function with function values and gradients."""
    print("=" * 60)
    print("Test 2: 2D Function with Values and Gradients")
    print("=" * 60)
    
    # Define test function: f(x,y) = sin(x)*cos(y) + 0.1*x*y
    def f(X):
        x, y = X[:, 0], X[:, 1]
        return (np.sin(x) * np.cos(y) + 0.1 * x * y).reshape(-1)
    
    def df_dx(X):
        x, y = X[:, 0], X[:, 1]
        return (np.cos(x) * np.cos(y) + 0.1 * y).reshape(-1, 1)
    
    def df_dy(X):
        x, y = X[:, 0], X[:, 1]
        return (-np.sin(x) * np.sin(y) + 0.1 * x).reshape(-1, 1)
    
    # Generate training data
    np.random.seed(42)
    x_train = np.random.uniform(-np.pi, np.pi, (20, 2))
    f_train = f(x_train)
    df_train = np.hstack([df_dx(x_train), df_dy(x_train)])
    
    # Fit model with gradients
    fitter = RBFPolynomialFitter(
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter.fit(x_train, f_train, df=df_train)
    
    # Evaluate on grid
    x_grid = np.linspace(-np.pi, np.pi, 30)
    y_grid = np.linspace(-np.pi, np.pi, 30)
    xx, yy = np.meshgrid(x_grid, y_grid)
    x_test = np.column_stack([xx.ravel(), yy.ravel()])
    
    f_pred = fitter.predict(x_test).reshape(xx.shape)
    f_true = f(x_test).reshape(xx.shape)
    
    df_pred = fitter.predict_gradient(x_test)
    
    func_error = np.mean((f_pred - f_true)**2)
    print(f"Function MSE: {func_error:.6e}")
    
    # Plot results
    fig = plt.figure(figsize=(14, 5))
    
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(xx, yy, f_true, cmap='viridis', alpha=0.7)
    ax1.scatter(x_train[:, 0], x_train[:, 1], f_train, color='red', s=50, zorder=5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)')
    ax1.set_title('True Function')
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(xx, yy, f_pred, cmap='viridis', alpha=0.7)
    ax2.scatter(x_train[:, 0], x_train[:, 1], f_train, color='red', s=50, zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('f(x,y)')
    ax2.set_title('RBF+Poly Prediction')
    
    ax3 = fig.add_subplot(133)
    error = np.abs(f_pred - f_true)
    contour = ax3.contourf(xx, yy, error, levels=15, cmap='hot')
    ax3.scatter(x_train[:, 0], x_train[:, 1], color='blue', s=30, zorder=5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Absolute Error')
    plt.colorbar(contour, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('2d_fit.png', dpi=150)
    print("Saved plot to 2d_fit.png\n")


def test_without_gradients():
    """Test fitting without gradient information."""
    print("=" * 60)
    print("Test 3: Fitting Without Gradient Information")
    print("=" * 60)
    
    def f(X):
        x = X[:, 0] if X.ndim > 1 else X
        return (np.sin(3*x) * np.exp(-0.1*x**2)).reshape(-1)
    
    np.random.seed(42)
    x_train = np.random.uniform(-5, 5, (25, 1))
    f_train = f(x_train)
    
    # Fit without gradients
    fitter = RBFPolynomialFitter(
        polynomial_degree=1,
        regularization_lambda=1e-4
    )
    fitter.fit(x_train, f_train)  # No df parameter
    
    x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    f_pred = fitter.predict(x_test)
    f_true = f(x_test)
    
    mse = np.mean((f_pred - f_true)**2)
    print(f"MSE (without gradients): {mse:.6e}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_test, f_true, 'b-', label='True function', linewidth=2)
    ax.plot(x_test, f_pred, 'r--', label='RBF+Poly (no gradients)', linewidth=2)
    ax.scatter(x_train, f_train, color='green', s=50, label='Training points', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Fitting Without Gradient Information')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('no_gradients.png', dpi=150)
    print("Saved plot to no_gradients.png\n")


def test_different_kernels():
    """Test with different RBF kernels."""
    print("=" * 60)
    print("Test 4: Different RBF Kernels")
    print("=" * 60)
    
    def f(x):
        return np.sin(2*x) + 0.05*x
    
    def df_dx(x):
        return 2*np.cos(2*x) + 0.05
    
    np.random.seed(42)
    x_train = np.linspace(-np.pi, np.pi, 12).reshape(-1, 1)
    f_train = f(x_train).ravel()
    df_train = df_dx(x_train).reshape(-1, 1)
    
    x_test = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
    f_true = f(x_test).ravel()
    
    # Gaussian RBF
    def gaussian_rbf(r):
        epsilon = 0.5
        return np.exp(-(epsilon * r)**2)
    
    def gaussian_deriv(r):
        epsilon = 0.5
        return -2 * epsilon**2 * r * np.exp(-(epsilon * r)**2)
    
    fitter_gaussian = RBFPolynomialFitter(
        rbf_kernel=gaussian_rbf,
        rbf_kernel_derivative=gaussian_deriv,
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter_gaussian.fit(x_train, f_train, df=df_train)
    f_pred_gaussian = fitter_gaussian.predict(x_test)
    err_gaussian = np.mean((f_pred_gaussian - f_true)**2)
    print(f"Gaussian RBF MSE: {err_gaussian:.6e}")
    
    # Multiquadric (default)
    fitter_mq = RBFPolynomialFitter(polynomial_degree=1, regularization_lambda=1e-5)
    fitter_mq.fit(x_train, f_train, df=df_train)
    f_pred_mq = fitter_mq.predict(x_test)
    err_mq = np.mean((f_pred_mq - f_true)**2)
    print(f"Multiquadric RBF MSE: {err_mq:.6e}")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_test, f_true, 'k-', label='True function', linewidth=2)
    ax.plot(x_test, f_pred_gaussian, 'b--', label='Gaussian RBF', linewidth=2)
    ax.plot(x_test, f_pred_mq, 'r--', label='Multiquadric RBF', linewidth=2)
    ax.scatter(x_train, f_train, color='green', s=50, label='Training points', zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Comparison of RBF Kernels')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kernel_comparison.png', dpi=150)
    print("Saved plot to kernel_comparison.png\n")


def test_lhs_centers():
    """Test fitting with Latin Hypercube Sampling for center generation."""
    print("=" * 60)
    print("Test 5: Latin Hypercube Sampling for Centers")
    print("=" * 60)
    
    # Define 2D test function
    def f(X):
        x, y = X[:, 0], X[:, 1]
        return (np.sin(x) * np.cos(y) + 0.1 * x * y).reshape(-1)
    
    def df_dx(X):
        x, y = X[:, 0], X[:, 1]
        return (np.cos(x) * np.cos(y) + 0.1 * y).reshape(-1, 1)
    
    def df_dy(X):
        x, y = X[:, 0], X[:, 1]
        return (-np.sin(x) * np.sin(y) + 0.1 * x).reshape(-1, 1)
    
    # Generate training data (arbitrary points)
    np.random.seed(42)
    x_train = np.random.uniform(-np.pi, np.pi, (25, 2))
    f_train = f(x_train)
    df_train = np.hstack([df_dx(x_train), df_dy(x_train)])
    
    # Test grid
    x_grid = np.linspace(-np.pi, np.pi, 25)
    y_grid = np.linspace(-np.pi, np.pi, 25)
    xx, yy = np.meshgrid(x_grid, y_grid)
    x_test = np.column_stack([xx.ravel(), yy.ravel()])
    f_true = f(x_test).reshape(xx.shape)
    
    # Fit with LHS-generated centers
    fitter_lhs = RBFPolynomialFitter(
        rbf_name='gaussian',
        rbf_shape_parameter=0.5,
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter_lhs.fit(
        x_train, f_train, df=df_train,
        use_lhs_centers=True,
        n_centers=15,
        lhs_bounds=(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi])),
        random_state=42
    )
    f_pred_lhs = fitter_lhs.predict(x_test).reshape(xx.shape)
    err_lhs = np.mean((f_pred_lhs - f_true)**2)
    
    # Fit with training data as centers
    fitter_data = RBFPolynomialFitter(
        rbf_name='gaussian',
        rbf_shape_parameter=0.5,
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter_data.fit(x_train, f_train, df=df_train)
    f_pred_data = fitter_data.predict(x_test).reshape(xx.shape)
    err_data = np.mean((f_pred_data - f_true)**2)
    
    print(f"LHS Centers (n=15) MSE: {err_lhs:.6e}")
    print(f"Training Data Centers (n=25) MSE: {err_data:.6e}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot LHS centers
    ax = axes[0]
    contour = ax.contourf(xx, yy, f_pred_lhs, levels=15, cmap='viridis')
    ax.scatter(fitter_lhs.centers[:, 0], fitter_lhs.centers[:, 1], 
               color='red', s=50, marker='*', label='LHS Centers')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'LHS Centers (MSE={err_lhs:.3e})')
    plt.colorbar(contour, ax=ax)
    
    # Plot training data centers
    ax = axes[1]
    contour = ax.contourf(xx, yy, f_pred_data, levels=15, cmap='viridis')
    ax.scatter(x_train[:, 0], x_train[:, 1], 
               color='red', s=30, label='Training Centers')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Training Centers (MSE={err_data:.3e})')
    plt.colorbar(contour, ax=ax)
    
    # Plot true function
    ax = axes[2]
    contour = ax.contourf(xx, yy, f_true, levels=15, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('True Function')
    plt.colorbar(contour, ax=ax)
    
    plt.tight_layout()
    plt.savefig('lhs_centers.png', dpi=150)
    print("Saved plot to lhs_centers.png\n")


def test_rbf_with_shape_parameters():
    """Test built-in RBF kernels with different shape parameters."""
    print("=" * 60)
    print("Test 6: RBF Kernels with Shape Parameters")
    print("=" * 60)
    
    def f(x):
        return np.sin(3*x) * np.exp(-0.1*x**2)
    
    def df_dx(x):
        return 3*np.cos(3*x)*np.exp(-0.1*x**2) - 0.2*x*np.sin(3*x)*np.exp(-0.1*x**2)
    
    np.random.seed(42)
    x_train = np.linspace(-5, 5, 15).reshape(-1, 1)
    f_train = f(x_train).ravel()
    df_train = df_dx(x_train).reshape(-1, 1)
    
    x_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    f_true = f(x_test).ravel()
    
    # Test different shape parameters for Gaussian kernel
    epsilons = [0.1, 0.5, 1.0, 2.0]
    errors = []
    preds = []
    
    for eps in epsilons:
        fitter = RBFPolynomialFitter(
            rbf_name='gaussian',
            rbf_shape_parameter=eps,
            polynomial_degree=1,
            regularization_lambda=1e-5
        )
        fitter.fit(x_train, f_train, df=df_train)
        f_pred = fitter.predict(x_test)
        err = np.mean((f_pred - f_true)**2)
        errors.append(err)
        preds.append(f_pred)
        print(f"Gaussian with epsilon={eps}: MSE={err:.6e}")
    
    # Also test built-in kernels without shape parameters
    print("\nKernels without shape parameters:")
    fitter_tps = RBFPolynomialFitter(
        rbf_name='thin_plate_spline',
        polynomial_degree=1,
        regularization_lambda=1e-4
    )
    fitter_tps.fit(x_train, f_train, df=df_train)
    f_pred_tps = fitter_tps.predict(x_test)
    err_tps = np.mean((f_pred_tps - f_true)**2)
    print(f"Thin Plate Spline: MSE={err_tps:.6e}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_test, f_true, 'k-', linewidth=3, label='True function', zorder=5)
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (eps, pred) in enumerate(zip(epsilons, preds)):
        ax.plot(x_test, pred, '--', color=colors[i], linewidth=2, 
                label=f'Gaussian ε={eps} (MSE={errors[i]:.2e})')
    
    ax.scatter(x_train, f_train, color='black', s=50, zorder=5, label='Training points')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Effect of Shape Parameter (ε) on Gaussian RBF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('shape_parameters.png', dpi=150)
    print("Saved plot to shape_parameters.png\n")


def test_rbf_only_no_polynomials():
    """Test RBF-only fitting without polynomial basis (polynomial_degree=None)."""
    print("=" * 60)
    print("Test 7: RBF-Only Fitting (No Polynomials)")
    print("=" * 60)
    
    def f(x):
        return np.sin(2*x) * np.cos(x/2)
    
    def df_dx(x):
        return 2*np.cos(2*x)*np.cos(x/2) - 0.5*np.sin(2*x)*np.sin(x/2)
    
    np.random.seed(42)
    x_train = np.linspace(-2*np.pi, 2*np.pi, 20).reshape(-1, 1)
    f_train = f(x_train).ravel()
    df_train = df_dx(x_train).reshape(-1, 1)
    
    x_test = np.linspace(-2*np.pi, 2*np.pi, 150).reshape(-1, 1)
    f_true = f(x_test).ravel()
    
    # RBF-only fitter (no polynomials)
    fitter_rbf_only = RBFPolynomialFitter(
        rbf_name='multiquadric',
        rbf_shape_parameter=0.5,
        polynomial_degree=None,  # No polynomial basis
        regularization_lambda=1e-5
    )
    fitter_rbf_only.fit(x_train, f_train, df=df_train)
    f_pred_rbf = fitter_rbf_only.predict(x_test)
    err_rbf = np.mean((f_pred_rbf - f_true)**2)
    
    # RBF + Linear polynomial
    fitter_rbf_poly = RBFPolynomialFitter(
        rbf_name='multiquadric',
        rbf_shape_parameter=0.5,
        polynomial_degree=1,  # Linear polynomial basis
        regularization_lambda=1e-5
    )
    fitter_rbf_poly.fit(x_train, f_train, df=df_train)
    f_pred_rbf_poly = fitter_rbf_poly.predict(x_test)
    err_rbf_poly = np.mean((f_pred_rbf_poly - f_true)**2)
    
    print(f"RBF-only (no polynomials) MSE: {err_rbf:.6e}")
    print(f"RBF + Linear polynomial MSE: {err_rbf_poly:.6e}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.plot(x_test, f_true, 'k-', linewidth=2.5, label='True function')
    ax.plot(x_test, f_pred_rbf, 'r--', linewidth=2, label=f'RBF-only (MSE={err_rbf:.2e})')
    ax.scatter(x_train, f_train, color='green', s=40, zorder=5, label='Training points')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('RBF-Only Fitting (polynomial_degree=None)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    ax.plot(x_test, f_true, 'k-', linewidth=2.5, label='True function')
    ax.plot(x_test, f_pred_rbf_poly, 'b--', linewidth=2, label=f'RBF+Linear (MSE={err_rbf_poly:.2e})')
    ax.scatter(x_train, f_train, color='green', s=40, zorder=5, label='Training points')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('RBF + Polynomial Fitting (polynomial_degree=1)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rbf_only_comparison.png', dpi=150)
    print("Saved plot to rbf_only_comparison.png\n")



if __name__ == "__main__":
    test_1d_function()
    test_2d_function()
    test_without_gradients()
    test_different_kernels()
    test_lhs_centers()
    test_rbf_with_shape_parameters()
    test_rbf_only_no_polynomials()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
