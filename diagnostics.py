"""
Diagnostic plotting for RBF + Polynomial Fitter

Comprehensive visualizations to validate that the fitter is working correctly:
- Function value predictions vs true values
- Gradient predictions accuracy
- Center locations and coverage
- Shape parameter effects
- LHS vs training data centers comparison
- Residuals and error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from rbf_polynomial_fitter import RBFPolynomialFitter


def plot_1d_diagnostics():
    """1D diagnostic plot showing fit quality and gradients."""
    print("Generating 1D diagnostics...")
    
    # Define test function and its gradient
    def f(x):
        return np.sin(3*x) * np.exp(-0.1*x**2)
    
    def df_dx(x):
        return (3*np.cos(3*x) - 0.2*x*np.sin(3*x)) * np.exp(-0.1*x**2)
    
    # Training data
    np.random.seed(42)
    x_train = np.linspace(-5, 5, 12).reshape(-1, 1)
    f_train = f(x_train).ravel()
    df_train = df_dx(x_train).reshape(-1, 1)
    
    # Test data
    x_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    f_true = f(x_test).ravel()
    df_true = df_dx(x_test).ravel()
    
    # Fit model
    rbf_shape_parameter=2.0
    fitter = RBFPolynomialFitter(
        rbf_name='gaussian',
        rbf_shape_parameter=rbf_shape_parameter,
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter.fit(x_train, f_train, df=df_train)
    
    # Predictions
    f_pred = fitter.predict(x_test)
    df_pred = fitter.predict_gradient(x_test).ravel()
    
    # Calculate errors
    f_error = f_pred - f_true
    df_error = df_pred - df_true
    f_rmse = np.sqrt(np.mean(f_error**2))
    df_rmse = np.sqrt(np.mean(df_error**2))
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Function predictions
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(x_test, f_true, 'b-', linewidth=2.5, label='True function')
    ax1.plot(x_test, f_pred, 'r--', linewidth=2, label='RBF+Poly prediction')
    ax1.scatter(x_train, f_train, color='green', s=80, marker='o', 
                edgecolors='darkgreen', linewidth=1.5, label='Training points', zorder=5)
    ax1.set_ylabel('f(x)', fontsize=10, fontweight='bold')
    ax1.set_title('Function Fit', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Function residuals
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(x_test, f_error, 'purple', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.fill_between(x_test.ravel(), f_error, alpha=0.3, color='purple')
    ax2.set_ylabel('Error', fontsize=10, fontweight='bold')
    ax2.set_title(f'Function Residuals (RMSE={f_rmse:.3e})', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Function error histogram
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(f_error, bins=30, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax3.set_xlabel('Error', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title('Function Error Distribution', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Gradient predictions
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(x_test, df_true, 'b-', linewidth=2.5, label='True gradient')
    ax4.plot(x_test, df_pred, 'r--', linewidth=2, label='RBF+Poly gradient')
    ax4.scatter(x_train, df_train.ravel(), color='orange', s=80, marker='^',
                edgecolors='darkorange', linewidth=1.5, label='Training gradients', zorder=5)
    ax4.set_ylabel('df/dx', fontsize=10, fontweight='bold')
    ax4.set_title('Gradient Fit', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Gradient residuals
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(x_test, df_error, 'orange', linewidth=2)
    ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax5.fill_between(x_test.ravel(), df_error, alpha=0.3, color='orange')
    ax5.set_ylabel('Error', fontsize=10, fontweight='bold')
    ax5.set_title(f'Gradient Residuals (RMSE={df_rmse:.3e})', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Gradient error histogram
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df_error, bins=30, color='orange', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax6.set_xlabel('Error', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax6.set_title('Gradient Error Distribution', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Prediction vs True (scatter)
    ax7 = plt.subplot(3, 3, 7)
    ax7.scatter(f_true, f_pred, alpha=0.5, s=20, color='purple')
    lims = [min(f_true.min(), f_pred.min()), max(f_true.max(), f_pred.max())]
    ax7.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
    ax7.set_xlabel('True f(x)', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Predicted f(x)', fontsize=10, fontweight='bold')
    ax7.set_title('Function: Predicted vs True', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # Gradient prediction vs True (scatter)
    ax8 = plt.subplot(3, 3, 8)
    ax8.scatter(df_true, df_pred, alpha=0.5, s=20, color='orange')
    lims = [min(df_true.min(), df_pred.min()), max(df_true.max(), df_pred.max())]
    ax8.plot(lims, lims, 'r--', linewidth=2, label='Perfect fit')
    ax8.set_xlabel('True df/dx', fontsize=10, fontweight='bold')
    ax8.set_ylabel('Predicted df/dx', fontsize=10, fontweight='bold')
    ax8.set_title('Gradient: Predicted vs True', fontsize=11, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    
    # Model info
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    info_text = f"""
    Model Configuration:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RBF Kernel: Gaussian (ε={rbf_shape_parameter})
    Polynomial Degree: 1 (Linear)
    Regularization λ: 1e-5
    
    Fit Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Training Points: {len(x_train)}
    RBF Centers: {fitter.n_rbf_bases}
    
    Error Metrics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Function RMSE: {f_rmse:.4e}
    Gradient RMSE: {df_rmse:.4e}
    Max |Error|: {np.max(np.abs(f_error)):.4e}
    """
    ax9.text(0.1, 0.5, info_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('1D Function & Gradient Fitting Diagnostics', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('diagnostic_1d.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved to diagnostic_1d.png\n")


def plot_2d_diagnostics():
    """2D diagnostic plot with center locations and error maps."""
    print("Generating 2D diagnostics...")
    
    # Define test function
    def f(X):
        x, y = X[:, 0], X[:, 1]
        return np.sin(2*x) * np.cos(2*y) + 0.1*x*y
    
    # Training data
    np.random.seed(42)
    X_train = np.random.uniform(-np.pi, np.pi, (25, 2))
    f_train = f(X_train)
    
    # Test grid
    x_grid = np.linspace(-np.pi, np.pi, 40)
    y_grid = np.linspace(-np.pi, np.pi, 40)
    xx, yy = np.meshgrid(x_grid, y_grid)
    X_test = np.column_stack([xx.ravel(), yy.ravel()])
    f_true = f(X_test).reshape(xx.shape)
    
    # Fit model
    fitter = RBFPolynomialFitter(
        rbf_name='multiquadric',
        rbf_shape_parameter=0.5,
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter.fit(X_train, f_train)
    
    # Predictions
    f_pred = fitter.predict(X_test).reshape(xx.shape)
    error = f_pred - f_true
    
    # Calculate metrics
    rmse = np.sqrt(np.mean(error**2))
    r2 = 1 - (np.sum(error**2) / np.sum((f_true - np.mean(f_train))**2))
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # True function
    ax1 = plt.subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(xx, yy, f_true, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.scatter(X_train[:, 0], X_train[:, 1], f_train, color='red', s=50, zorder=5)
    ax1.set_xlabel('x', fontsize=9)
    ax1.set_ylabel('y', fontsize=9)
    ax1.set_zlabel('f(x,y)', fontsize=9)
    ax1.set_title('True Function', fontsize=11, fontweight='bold')
    plt.colorbar(surf1, ax=ax1, pad=0.1, shrink=0.8)
    
    # Predicted function
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(xx, yy, f_pred, cmap='viridis', alpha=0.8, edgecolor='none')
    ax2.scatter(X_train[:, 0], X_train[:, 1], f_train, color='red', s=50, zorder=5)
    ax2.set_xlabel('x', fontsize=9)
    ax2.set_ylabel('y', fontsize=9)
    ax2.set_zlabel('f(x,y)', fontsize=9)
    ax2.set_title('RBF+Poly Prediction', fontsize=11, fontweight='bold')
    plt.colorbar(surf2, ax=ax2, pad=0.1, shrink=0.8)
    
    # Error map
    ax3 = plt.subplot(2, 3, 3)
    contour = ax3.contourf(xx, yy, np.abs(error), levels=20, cmap='hot')
    ax3.scatter(X_train[:, 0], X_train[:, 1], color='cyan', s=30, marker='x', linewidth=2)
    ax3.scatter(fitter.centers[:, 0], fitter.centers[:, 1], color='blue', s=80, 
                marker='*', edgecolors='darkblue', linewidth=1, label='RBF Centers')
    ax3.set_xlabel('x', fontsize=10, fontweight='bold')
    ax3.set_ylabel('y', fontsize=10, fontweight='bold')
    ax3.set_title(f'Absolute Error (RMSE={rmse:.3e})', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    cbar = plt.colorbar(contour, ax=ax3)
    cbar.set_label('|Error|', fontsize=9)
    
    # Error distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(error.ravel(), bins=40, color='crimson', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='k', linestyle='--', linewidth=2)
    ax4.set_xlabel('Error', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Predicted vs True
    ax5 = plt.subplot(2, 3, 5)
    scatter = ax5.scatter(f_true.ravel(), f_pred.ravel(), c=error.ravel(), 
                         cmap='coolwarm', s=20, alpha=0.6)
    lims = [min(f_true.min(), f_pred.min()), max(f_true.max(), f_pred.max())]
    ax5.plot(lims, lims, 'r--', linewidth=2.5, label='Perfect fit')
    ax5.set_xlabel('True f(x,y)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Predicted f(x,y)', fontsize=10, fontweight='bold')
    ax5.set_title('Predicted vs True', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax5)
    cbar.set_label('Error', fontsize=9)
    
    # Model info
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    info_text = f"""
    Model Configuration:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RBF Kernel: Multiquadric (ε=0.5)
    Polynomial Degree: 1 (Linear)
    Regularization λ: 1e-5
    
    Fit Statistics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Training Points: {len(X_train)}
    RBF Centers: {fitter.n_rbf_bases}
    Test Grid: {len(X_test)} points
    
    Error Metrics:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    RMSE: {rmse:.4e}
    R² Score: {r2:.6f}
    Max |Error|: {np.max(np.abs(error)):.4e}
    Mean |Error|: {np.mean(np.abs(error)):.4e}
    """
    ax6.text(0.1, 0.5, info_text, fontsize=9, verticalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.suptitle('2D Function Fitting Diagnostics', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('diagnostic_2d.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved to diagnostic_2d.png\n")


def plot_lhs_vs_data_centers():
    """Compare LHS-generated centers vs training data centers."""
    print("Generating LHS vs Data Centers comparison...")
    
    # Function
    def f(X):
        x, y = X[:, 0], X[:, 1]
        return np.sin(3*x) * np.cos(2*y)
    
    # Generate sparse training data
    np.random.seed(42)
    X_train = np.random.uniform(-np.pi, np.pi, (30, 2))
    f_train = f(X_train)
    
    # Test grid
    x_grid = np.linspace(-np.pi, np.pi, 30)
    y_grid = np.linspace(-np.pi, np.pi, 30)
    xx, yy = np.meshgrid(x_grid, y_grid)
    X_test = np.column_stack([xx.ravel(), yy.ravel()])
    f_true = f(X_test).reshape(xx.shape)
    
    # Fit with training data as centers
    fitter_data = RBFPolynomialFitter(
        rbf_name='gaussian',
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter_data.fit(X_train, f_train)
    f_pred_data = fitter_data.predict(X_test).reshape(xx.shape)
    error_data = f_pred_data - f_true
    rmse_data = np.sqrt(np.mean(error_data**2))
    
    # Fit with LHS centers
    fitter_lhs = RBFPolynomialFitter(
        rbf_name='gaussian',
        polynomial_degree=1,
        regularization_lambda=1e-5
    )
    fitter_lhs.fit(
        X_train, f_train,
        use_lhs_centers=True,
        n_centers=15,
        lhs_bounds=(np.array([-np.pi, -np.pi]), np.array([np.pi, np.pi])),
        random_state=42
    )
    f_pred_lhs = fitter_lhs.predict(X_test).reshape(xx.shape)
    error_lhs = f_pred_lhs - f_true
    rmse_lhs = np.sqrt(np.mean(error_lhs**2))
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Data centers - prediction
    ax1 = plt.subplot(2, 3, 1)
    contour1 = ax1.contourf(xx, yy, f_pred_data, levels=20, cmap='viridis')
    ax1.scatter(X_train[:, 0], X_train[:, 1], color='red', s=30, marker='o',
                edgecolors='darkred', linewidth=1, label='Training Points', zorder=5)
    ax1.scatter(fitter_data.centers[:, 0], fitter_data.centers[:, 1], 
                color='yellow', s=100, marker='*', edgecolors='orange', linewidth=2,
                label='RBF Centers', zorder=6)
    ax1.set_xlabel('x', fontsize=10, fontweight='bold')
    ax1.set_ylabel('y', fontsize=10, fontweight='bold')
    ax1.set_title(f'Data Centers Prediction\n(n={fitter_data.n_rbf_bases}, RMSE={rmse_data:.3e})',
                  fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    plt.colorbar(contour1, ax=ax1)
    
    # Data centers - error
    ax2 = plt.subplot(2, 3, 2)
    contour2 = ax2.contourf(xx, yy, np.abs(error_data), levels=20, cmap='hot')
    ax2.scatter(X_train[:, 0], X_train[:, 1], color='cyan', s=20, marker='x', linewidth=1)
    ax2.scatter(fitter_data.centers[:, 0], fitter_data.centers[:, 1], 
                color='blue', s=80, marker='*', edgecolors='darkblue', linewidth=1, zorder=5)
    ax2.set_xlabel('x', fontsize=10, fontweight='bold')
    ax2.set_ylabel('y', fontsize=10, fontweight='bold')
    ax2.set_title('Data Centers Error', fontsize=11, fontweight='bold')
    plt.colorbar(contour2, ax=ax2)
    
    # Data centers - histogram
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(error_data.ravel(), bins=30, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Error', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title(f'Data Centers Error Dist.', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # LHS centers - prediction
    ax4 = plt.subplot(2, 3, 4)
    contour4 = ax4.contourf(xx, yy, f_pred_lhs, levels=20, cmap='viridis')
    ax4.scatter(X_train[:, 0], X_train[:, 1], color='red', s=30, marker='o',
                edgecolors='darkred', linewidth=1, label='Training Points', zorder=5)
    ax4.scatter(fitter_lhs.centers[:, 0], fitter_lhs.centers[:, 1], 
                color='yellow', s=100, marker='*', edgecolors='orange', linewidth=2,
                label='LHS Centers', zorder=6)
    ax4.set_xlabel('x', fontsize=10, fontweight='bold')
    ax4.set_ylabel('y', fontsize=10, fontweight='bold')
    ax4.set_title(f'LHS Centers Prediction\n(n={fitter_lhs.n_rbf_bases}, RMSE={rmse_lhs:.3e})',
                  fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    plt.colorbar(contour4, ax=ax4)
    
    # LHS centers - error
    ax5 = plt.subplot(2, 3, 5)
    contour5 = ax5.contourf(xx, yy, np.abs(error_lhs), levels=20, cmap='hot')
    ax5.scatter(X_train[:, 0], X_train[:, 1], color='cyan', s=20, marker='x', linewidth=1)
    ax5.scatter(fitter_lhs.centers[:, 0], fitter_lhs.centers[:, 1], 
                color='blue', s=80, marker='*', edgecolors='darkblue', linewidth=1, zorder=5)
    ax5.set_xlabel('x', fontsize=10, fontweight='bold')
    ax5.set_ylabel('y', fontsize=10, fontweight='bold')
    ax5.set_title('LHS Centers Error', fontsize=11, fontweight='bold')
    plt.colorbar(contour5, ax=ax5)
    
    # LHS centers - histogram
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(error_lhs.ravel(), bins=30, color='green', alpha=0.7, edgecolor='black')
    ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax6.set_xlabel('Error', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax6.set_title(f'LHS Centers Error Dist.', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Comparison: LHS vs Training Data Centers', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('diagnostic_lhs_vs_data.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved to diagnostic_lhs_vs_data.png\n")


def plot_kernel_comparison():
    """Compare different RBF kernels on same problem."""
    print("Generating kernel comparison...")
    
    # Function
    def f(x):
        return np.sin(2*x) * np.cos(x)
    
    # Training data
    np.random.seed(42)
    x_train = np.linspace(-3, 3, 15).reshape(-1, 1)
    f_train = f(x_train).ravel()
    
    # Test data
    x_test = np.linspace(-3, 3, 150).reshape(-1, 1)
    f_true = f(x_test).ravel()
    
    # Test different kernels
    kernels = [
        ('gaussian', 0.5),
        ('multiquadric', 0.5),
        ('inverse_multiquadric', 0.5),
        ('thin_plate_spline', None),
        ('cubic', None),
        ('linear', None),
    ]
    
    fig = plt.figure(figsize=(16, 12))
    
    results = []
    for idx, (kernel_name, shape_param) in enumerate(kernels):
        fitter = RBFPolynomialFitter(
            rbf_name=kernel_name,
            rbf_shape_parameter=shape_param,
            polynomial_degree=1,
            regularization_lambda=1e-4
        )
        fitter.fit(x_train, f_train)
        f_pred = fitter.predict(x_test)
        error = f_pred - f_true
        rmse = np.sqrt(np.mean(error**2))
        results.append((kernel_name, f_pred, error, rmse))
        
        # Prediction plot
        ax_pred = plt.subplot(3, 6, 2*idx + 1)
        ax_pred.plot(x_test, f_true, 'b-', linewidth=2.5, label='True')
        ax_pred.plot(x_test, f_pred, 'r--', linewidth=2, label='Prediction')
        ax_pred.scatter(x_train, f_train, color='green', s=60, zorder=5)
        ax_pred.set_ylabel('f(x)', fontsize=9)
        if kernel_name in ['gaussian', 'multiquadric', 'inverse_multiquadric']:
            title = f'{kernel_name} (ε={shape_param})'
        else:
            title = kernel_name
        ax_pred.set_title(title, fontsize=10, fontweight='bold')
        ax_pred.legend(fontsize=8)
        ax_pred.grid(True, alpha=0.3)
        
        # Error plot
        ax_err = plt.subplot(3, 6, 2*idx + 2)
        ax_err.plot(x_test, error, color='purple', linewidth=2)
        ax_err.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax_err.fill_between(x_test.ravel(), error, alpha=0.3, color='purple')
        ax_err.set_ylabel('Error', fontsize=9)
        ax_err.set_title(f'RMSE={rmse:.3e}', fontsize=10, fontweight='bold')
        ax_err.grid(True, alpha=0.3)
    
    plt.suptitle('Kernel Comparison on 1D Function', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('diagnostic_kernels.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved to diagnostic_kernels.png\n")
    
    # Print summary
    print("  Kernel Performance Summary:")
    print("  " + "─" * 50)
    for kernel_name, _, _, rmse in results:
        print(f"    {kernel_name:25s}: RMSE = {rmse:.6e}")
    print()


def main():
    """Run all diagnostic plots."""
    print("\n" + "=" * 70)
    print("RBF + POLYNOMIAL FITTER - DIAGNOSTIC VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    plot_1d_diagnostics()
    plot_2d_diagnostics()
    plot_lhs_vs_data_centers()
    plot_kernel_comparison()
    
    print("=" * 70)
    print("All diagnostic plots generated successfully!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • diagnostic_1d.png         - 1D function & gradient fitting")
    print("  • diagnostic_2d.png         - 2D function with error analysis")
    print("  • diagnostic_lhs_vs_data.png - LHS vs training data centers")
    print("  • diagnostic_kernels.png    - Comparison of RBF kernels")
    print()


if __name__ == "__main__":
    main()
