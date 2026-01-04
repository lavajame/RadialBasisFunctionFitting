"""
Complete Example: Loading CSV Data and Fitting

This example demonstrates the complete workflow:
1. Loading function values from CSV
2. Loading gradient data from CSV
3. Fitting the RBF + polynomial model
4. Making predictions
5. Computing performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_polynomial_fitter import RBFPolynomialFitter


def example_with_synthetic_csv_data():
    """
    Complete example using synthetic data that mimics CSV loading.
    Replace with actual CSV loading code for your data.
    """
    print("\n" + "=" * 70)
    print("EXAMPLE: Loading CSV Data and Fitting RBF + Polynomial Model")
    print("=" * 70 + "\n")
    
    # =========================================================================
    # STEP 1: LOAD DATA (simulating CSV loading)
    # =========================================================================
    print("Step 1: Loading Data")
    print("-" * 70)
    
    # For CSV data, you would do:
    # import pandas as pd
    # data = pd.read_csv('your_data.csv')
    # X_train = data[['x1', 'x2', ...]].values
    # f_train = data['function_value'].values
    # df_train = data[['df_dx1', 'df_dx2', ...]].values
    
    # Simulating CSV loading:
    np.random.seed(42)
    n_samples = 40
    n_features = 2
    
    # Simulate training data from CSV
    X_train = np.random.uniform(-2*np.pi, 2*np.pi, (n_samples, n_features))
    f_train = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1]) + 0.05*np.random.randn(n_samples)
    
    # Simulate gradient data from CSV
    df_dx0 = np.cos(X_train[:, 0]) * np.cos(X_train[:, 1])
    df_dx1 = -np.sin(X_train[:, 0]) * np.sin(X_train[:, 1])
    df_train = np.column_stack([df_dx0, df_dx1])
    
    print(f"  Training points: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  X shape: {X_train.shape}")
    print(f"  f shape: {f_train.shape}")
    print(f"  df shape: {df_train.shape}")
    print(f"  f range: [{f_train.min():.4f}, {f_train.max():.4f}]")
    print()
    
    # =========================================================================
    # STEP 2: CREATE FITTER WITH DESIRED CONFIGURATION
    # =========================================================================
    print("Step 2: Creating and Configuring Fitter")
    print("-" * 70)
    
    # Option A: RBF with linear polynomial
    fitter = RBFPolynomialFitter(
        rbf_name='gaussian',              # Gaussian RBF kernel
        rbf_shape_parameter=0.5,          # Shape parameter (epsilon)
        polynomial_degree=1,              # Linear polynomial basis
        regularization_lambda=1e-5        # L2 regularization
    )
    print("  Configuration:")
    print(f"    RBF Kernel: gaussian (ε=0.5)")
    print(f"    Polynomial Degree: 1 (linear)")
    print(f"    Regularization λ: 1e-5")
    print()
    
    # Alternative options:
    # Option B: RBF-only (no polynomial)
    # fitter = RBFPolynomialFitter(
    #     rbf_name='thin_plate_spline',
    #     polynomial_degree=None,          # No polynomial basis
    #     regularization_lambda=1e-4
    # )
    
    # Option C: With LHS centers
    # fitter = RBFPolynomialFitter(
    #     rbf_name='multiquadric',
    #     rbf_shape_parameter=0.8,
    #     polynomial_degree=2,
    #     regularization_lambda=1e-6
    # )
    
    # =========================================================================
    # STEP 3: FIT THE MODEL
    # =========================================================================
    print("Step 3: Fitting the Model")
    print("-" * 70)
    
    # Option A: Standard fit with training data as centers
    fitter.fit(X_train, f_train, df=df_train)
    print(f"  Model fitted successfully!")
    print(f"  RBF centers used: {fitter.n_rbf_bases}")
    print(f"  Total basis functions: {fitter.n_rbf_bases + fitter._count_polynomial_bases(n_features)}")
    print()
    
    # Option B: Fit with LHS-generated centers
    # fitter.fit(
    #     X_train, f_train, df=df_train,
    #     use_lhs_centers=True,
    #     n_centers=20,
    #     lhs_bounds=(
    #         np.array([-2*np.pi, -2*np.pi]),
    #         np.array([2*np.pi, 2*np.pi])
    #     ),
    #     random_state=42
    # )
    
    # =========================================================================
    # STEP 4: MAKE PREDICTIONS ON NEW DATA
    # =========================================================================
    print("Step 4: Making Predictions")
    print("-" * 70)
    
    # Generate test points
    x_grid = np.linspace(-2*np.pi, 2*np.pi, 20)
    y_grid = np.linspace(-2*np.pi, 2*np.pi, 20)
    xx, yy = np.meshgrid(x_grid, y_grid)
    X_test = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Predict function values
    f_pred = fitter.predict(X_test)
    print(f"  Function predictions: {f_pred.shape}")
    print(f"  f_pred range: [{f_pred.min():.4f}, {f_pred.max():.4f}]")
    
    # Predict gradients
    df_pred = fitter.predict_gradient(X_test)
    print(f"  Gradient predictions: {df_pred.shape}")
    print()
    
    # =========================================================================
    # STEP 5: EVALUATE MODEL PERFORMANCE
    # =========================================================================
    print("Step 5: Evaluating Performance")
    print("-" * 70)
    
    # Compute errors on training data
    f_train_pred = fitter.predict(X_train)
    f_train_error = f_train_pred - f_train
    f_train_rmse = np.sqrt(np.mean(f_train_error**2))
    f_train_mae = np.mean(np.abs(f_train_error))
    
    print(f"  Function Value Metrics (on training data):")
    print(f"    RMSE: {f_train_rmse:.6e}")
    print(f"    MAE:  {f_train_mae:.6e}")
    print(f"    Max Error: {np.max(np.abs(f_train_error)):.6e}")
    
    # Gradient errors
    df_train_pred = fitter.predict_gradient(X_train)
    df_train_error = df_train_pred - df_train
    df_train_rmse = np.sqrt(np.mean(df_train_error**2))
    
    print(f"  Gradient Metrics (on training data):")
    print(f"    RMSE: {df_train_rmse:.6e}")
    print()
    
    # =========================================================================
    # STEP 6: VISUALIZATION
    # =========================================================================
    print("Step 6: Creating Visualizations")
    print("-" * 70)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Function values
    ax1 = fig.add_subplot(131)
    scatter1 = ax1.scatter(X_test[:, 0], X_test[:, 1], c=f_pred, cmap='viridis', s=50)
    ax1.scatter(X_train[:, 0], X_train[:, 1], color='red', s=50, 
                marker='x', linewidth=2, label='Training points')
    ax1.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax1.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax1.set_title('Predicted Function Values', fontsize=12, fontweight='bold')
    ax1.legend()
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('f(x)', fontsize=10)
    
    # Plot 2: Gradient magnitude
    ax2 = fig.add_subplot(132)
    grad_magnitude = np.linalg.norm(df_pred, axis=1)
    scatter2 = ax2.scatter(X_test[:, 0], X_test[:, 1], c=grad_magnitude, cmap='hot', s=50)
    ax2.scatter(X_train[:, 0], X_train[:, 1], color='cyan', s=50, 
                marker='x', linewidth=2, label='Training points')
    ax2.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax2.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax2.set_title('Gradient Magnitude ||∇f||', fontsize=12, fontweight='bold')
    ax2.legend()
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('||∇f||', fontsize=10)
    
    # Plot 3: Residuals on training data
    ax3 = fig.add_subplot(133)
    scatter3 = ax3.scatter(X_train[:, 0], X_train[:, 1], c=np.abs(f_train_error), 
                          cmap='Reds', s=80, marker='o', edgecolors='darkred', linewidth=1)
    ax3.set_xlabel('x₁', fontsize=11, fontweight='bold')
    ax3.set_ylabel('x₂', fontsize=11, fontweight='bold')
    ax3.set_title('Training Error Magnitude |f_pred - f_train|', fontsize=12, fontweight='bold')
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('|Error|', fontsize=10)
    
    plt.suptitle('RBF + Polynomial Fitter: Complete Example', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('example_complete.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved visualization to 'example_complete.png'")
    print()
    
    # =========================================================================
    # STEP 7: ACCESSING INTERNAL DATA
    # =========================================================================
    print("Step 7: Accessing Fitter Internal Data")
    print("-" * 70)
    print(f"  RBF centers (first 3):")
    print(f"    {fitter.centers[:3]}")
    print(f"  Fitted coefficients (first 5):")
    print(f"    {fitter.coefficients[:5]}")
    print(f"  Fitted flag: {fitter.fitted}")
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print(f"""
Summary:
  ✓ Loaded {n_samples} training points with gradients
  ✓ Fitted RBF+polynomial model
  ✓ Made predictions on {len(X_test)} test points
  ✓ Computed performance metrics
  ✓ Generated visualizations
  
To use this with your CSV data:
  1. Replace synthetic data with pd.read_csv('your_file.csv')
  2. Extract X, f, and df columns
  3. Adjust fitter configuration as needed
  4. Follow steps 2-7 above
  
Performance Summary:
  Function RMSE: {f_train_rmse:.6e}
  Gradient RMSE: {df_train_rmse:.6e}
  
Next Steps:
  - Tune regularization_lambda if overfitting
  - Experiment with different rbf_kernels
  - Adjust polynomial_degree (or set to None)
  - Try LHS center generation for large datasets
""")


if __name__ == "__main__":
    example_with_synthetic_csv_data()
