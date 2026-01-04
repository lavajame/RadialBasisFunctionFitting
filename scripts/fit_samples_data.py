"""
Fit RBF model to samples.csv data with diagnostic output.
Uses data normalization for numerical stability.

- First 5 columns: training data locations (q, Merton.sigma, Merton.lam, Merton.muJ, Merton.sigmaJ)
- 6th column: loss (function values)
- Next 5 columns: gradients
- Model: 10 Latin Hypercube centers, Multiquadric RBF, polynomial degree 2
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_polynomial_fitter import RBFPolynomialFitter

# Load CSV data
data = np.genfromtxt('samples.csv', delimiter=',', skip_header=1)
print(f"Loaded {data.shape[0]} samples with {data.shape[1]} columns")

# Extract components
X = data[:, :5]  # First 5 columns: training locations
f = data[:, 5]   # 6th column: loss values
grads = data[:, 6:11]  # Next 5 columns: gradients

print(f"\nData shapes:")
print(f"  X (locations): {X.shape}")
print(f"  f (loss values): {f.shape}")
print(f"  gradients: {grads.shape}")
print(f"\nData ranges BEFORE normalization:")
print(f"  Loss: [{f.min():.4f}, {f.max():.4f}]")

# Normalize data for numerical stability
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-10  # Avoid division by zero
X_norm = (X - X_mean) / X_std

f_mean = f.mean()
f_std = f.std() + 1e-10
f_norm = (f - f_mean) / f_std

# Gradients need to be scaled appropriately: df_norm/dX_norm = df/dX * (dX_norm/dX) = df/dX * (1/X_std)
grads_norm = grads / X_std[np.newaxis, :]  # Broadcasting: divide each gradient by its dimension's std

print(f"\nData ranges AFTER normalization:")
print(f"  Loss: [{f_norm.min():.4f}, {f_norm.max():.4f}]")
print(f"  Gradient magnitudes per dimension: {[f'[{grads_norm[:,i].min():.2f}, {grads_norm[:,i].max():.2f}]' for i in range(5)]}")

# Fit RBF model on normalized data
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    rbf_shape_parameter=1.0,  # epsilon for multiquadric
    polynomial_degree=2,       # max polynomial order of 2
    regularization_lambda=1e-6
)

print("\n" + "="*70)
print("FITTING MODEL (on normalized data)")
print("="*70)

# Fit with 10 LHS centers
lhs_lower = X_norm.min(axis=0)
lhs_upper = X_norm.max(axis=0)

fitter.fit(
    X_norm, f_norm,
    df=grads_norm,
    n_centers=10,
    use_lhs_centers=True,
    lhs_bounds=(lhs_lower, lhs_upper),
    random_state=42
)

print(f"✓ Model fitted with {fitter.n_rbf_bases} RBF centers")
n_poly_bases = fitter._count_polynomial_bases(X_norm.shape[1])
print(f"  Polynomial bases: {n_poly_bases}")
print(f"  Total basis functions: {fitter.n_rbf_bases + n_poly_bases}")

# Make predictions on normalized training data
f_pred_norm = fitter.predict(X_norm)
grads_pred_norm = fitter.predict_gradient(X_norm)

# Denormalize predictions
f_pred = f_pred_norm * f_std + f_mean
grads_pred = grads_pred_norm / X_std[np.newaxis, :]

# Compute metrics in original scale
residuals = f - f_pred
rmse = np.sqrt(np.mean(residuals**2))
r2 = 1 - np.sum(residuals**2) / np.sum((f - f.mean())**2)
mae = np.abs(residuals).mean()

grad_residuals = grads - grads_pred
grad_rmse = np.sqrt(np.mean(grad_residuals**2))
grad_r2_per_dim = [
    1 - np.sum(grad_residuals[:, i]**2) / np.sum((grads[:, i] - grads[:, i].mean())**2)
    for i in range(5)
]

print(f"\n" + "="*70)
print("MODEL PERFORMANCE (original scale)")
print("="*70)
print(f"Function Fit:")
print(f"  RMSE: {rmse:.6e}")
print(f"  MAE:  {mae:.6e}")
print(f"  R²: {r2:.6f}")
print(f"  Max Error: {np.abs(residuals).max():.6e}")
print(f"\nGradient Fit (per dimension):")
print(f"  Overall RMSE: {grad_rmse:.6e}")
grad_dim_names = ['q', 'Merton.sigma', 'Merton.lam', 'Merton.muJ', 'Merton.sigmaJ']
for i, name in enumerate(grad_dim_names):
    rmse_i = np.sqrt(np.mean(grad_residuals[:, i]**2))
    mae_i = np.abs(grad_residuals[:, i]).mean()
    print(f"  {name:20s}: R² = {grad_r2_per_dim[i]:8.4f}, RMSE = {rmse_i:.6e}, MAE = {mae_i:.6e}")

# Create comprehensive diagnostic plots
fig = plt.figure(figsize=(16, 12))

# 1. Predicted vs Actual (Function)
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(f, f_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
lims = [min(f.min(), f_pred.min()), max(f.max(), f_pred.max())]
ax1.plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
ax1.set_xlabel('Actual Loss', fontsize=10)
ax1.set_ylabel('Predicted Loss', fontsize=10)
ax1.set_title(f'Predicted vs Actual Loss\nR² = {r2:.4f}', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Residuals vs Predicted
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(f_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Loss', fontsize=10)
ax2.set_ylabel('Residuals', fontsize=10)
ax2.set_title(f'Residuals vs Predicted\nRMSE = {rmse:.4e}', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Residual Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Residual Value', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title(f'Residual Distribution\nMean = {residuals.mean():.4e}, Std = {residuals.std():.4e}', 
              fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# 4-8. Gradient fits for each dimension
for dim in range(5):
    ax = plt.subplot(3, 3, 4 + dim)
    ax.scatter(grads[:, dim], grads_pred[:, dim], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    lims = [min(grads[:, dim].min(), grads_pred[:, dim].min()), 
            max(grads[:, dim].max(), grads_pred[:, dim].max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel(f'Actual ∂loss/∂{grad_dim_names[dim]}', fontsize=9)
    ax.set_ylabel(f'Predicted ∂loss/∂{grad_dim_names[dim]}', fontsize=9)
    ax.set_title(f'Gradient {grad_dim_names[dim]}\nR² = {grad_r2_per_dim[dim]:.4f}', 
                 fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

# 9. Model Info
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
info_text = f"""
MODEL CONFIGURATION
RBF Kernel: Multiquadric (ε=1.0)
Polynomial Degree: 2
LHS Centers: 10
RBF Bases: {fitter.n_rbf_bases}
Polynomial Bases: {n_poly_bases}
Total Basis Functions: {fitter.n_rbf_bases + n_poly_bases}

PERFORMANCE SUMMARY
Function RMSE: {rmse:.6e}
Function R²: {r2:.4f}
Gradient RMSE: {grad_rmse:.6e}
Max Error: {np.abs(residuals).max():.6e}

TRAINING DATA
Samples: {len(X)}
Features: {X.shape[1]}
"""
ax9.text(0.05, 0.95, info_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('fit_samples_diagnostics.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Diagnostic plot saved to: fit_samples_diagnostics.png")
plt.close()

# Create residual analysis plot
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Residuals vs each input dimension
for dim in range(5):
    ax = axes[dim // 3, dim % 3]
    ax.scatter(X[:, dim], residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(grad_dim_names[dim], fontsize=10)
    ax.set_ylabel('Residuals', fontsize=10)
    ax.set_title(f'Residuals vs {grad_dim_names[dim]}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Absolute errors histogram on last subplot
ax = axes[1, 2]
abs_errors = np.abs(residuals)
ax.hist(abs_errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Absolute Error', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title(f'Absolute Error Distribution\nMedian = {np.median(abs_errors):.4e}', 
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fit_samples_residual_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Residual analysis plot saved to: fit_samples_residual_analysis.png")
plt.close()

# Summary statistics
print(f"\n" + "="*70)
print("RESIDUAL STATISTICS")
print("="*70)
print(f"Percentiles (function residuals):")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  {p:2d}th: {np.percentile(residuals, p):12.6e}")

print(f"\n✓ Analysis complete!")
