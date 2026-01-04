"""
Fit RBF model to samples.csv data with comprehensive diagnostic output.
Uses individual normalization for each variable for numerical stability.
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

grad_dim_names = ['q', 'Merton.sigma', 'Merton.lam', 'Merton.muJ', 'Merton.sigmaJ']

print(f"\nData shapes:")
print(f"  X (locations): {X.shape}")
print(f"  f (loss values): {f.shape}")
print(f"  gradients: {grads.shape}")

print(f"\n" + "="*70)
print("DATA NORMALIZATION")
print("="*70)

# Normalize input features
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-10  
X_norm = (X - X_mean) / X_std

# Normalize function values
f_mean = f.mean()
f_std = f.std() + 1e-10
f_norm = (f - f_mean) / f_std

# Normalize EACH gradient independently for stability
grads_norm = np.zeros_like(grads)
for i in range(grads.shape[1]):
    g = grads[:, i]
    g_mean = g.mean()
    g_std = g.std() + 1e-10
    grads_norm[:, i] = (g - g_mean) / g_std

print(f"\nNormalized ranges:")
print(f"  Loss: [{f_norm.min():.4f}, {f_norm.max():.4f}] (std={f_norm.std():.4f})")
print(f"  Gradient ranges:")
for i in range(5):
    print(f"    {grad_dim_names[i]:15s}: [{grads_norm[:,i].min():8.4f}, {grads_norm[:,i].max():8.4f}] (std={grads_norm[:,i].std():.4f})")

# Fit RBF model on normalized data
fitter = RBFPolynomialFitter(
    rbf_name='multiquadric',
    rbf_shape_parameter=1.0,
    polynomial_degree=2,
    regularization_lambda=1e-8
)

print("\n" + "="*70)
print("FITTING MODEL")
print("="*70)

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

# Compute normalized metrics (meaningful scale)
residuals_norm = f_norm - f_pred_norm
rmse_norm = np.sqrt(np.mean(residuals_norm**2))
r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)

grad_residuals_norm = grads_norm - grads_pred_norm
grad_rmse_norm = np.sqrt(np.mean(grad_residuals_norm**2))

# Store gradient statistics for denormalization reference
grad_stats = []
for i in range(grads.shape[1]):
    g = grads[:, i]
    grad_stats.append({
        'mean': g.mean(),
        'std': g.std() + 1e-10,
    })

print(f"\n" + "="*70)
print("MODEL PERFORMANCE (normalized scale - most reliable)")
print("="*70)
print(f"Function Fit (in normalized loss units):")
print(f"  RMSE: {rmse_norm:.6f}")
print(f"  R²:   {r2_norm:.6f}")
print(f"\nGradient Fit (in normalized gradient units):")
print(f"  Overall RMSE: {grad_rmse_norm:.6f}")
grad_r2_per_dim = []
for i in range(5):
    rmse_i = np.sqrt(np.mean(grad_residuals_norm[:, i]**2))
    r2_i = 1 - np.sum(grad_residuals_norm[:, i]**2) / np.sum((grads_norm[:, i] - grads_norm[:, i].mean())**2)
    grad_r2_per_dim.append(r2_i)
    print(f"  {grad_dim_names[i]:15s}: R² = {r2_i:8.4f}, RMSE = {rmse_i:.6f}")

# Create comprehensive diagnostic plots (using normalized predictions)
fig = plt.figure(figsize=(16, 12))

# 1. Predicted vs Actual (Function) - normalized scale
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(f_norm, f_pred_norm, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
lims = [min(f_norm.min(), f_pred_norm.min()), max(f_norm.max(), f_pred_norm.max())]
ax1.plot(lims, lims, 'r--', lw=2, label='Perfect Fit')
ax1.set_xlabel('Actual Loss (normalized)', fontsize=10)
ax1.set_ylabel('Predicted Loss (normalized)', fontsize=10)
ax1.set_title(f'Predicted vs Actual Loss\nR² = {r2_norm:.4f} (normalized scale)', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Residuals vs Predicted
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(f_pred_norm, residuals_norm, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Predicted Loss (normalized)', fontsize=10)
ax2.set_ylabel('Residuals (normalized)', fontsize=10)
ax2.set_title(f'Residuals vs Predicted\nRMSE = {rmse_norm:.4f} (normalized)', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.3)

# 3. Residual Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(residuals_norm, bins=20, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('Residual Value', fontsize=10)
ax3.set_ylabel('Frequency', fontsize=10)
ax3.set_title(f'Residual Distribution\nMean = {residuals_norm.mean():.4e}, Std = {residuals_norm.std():.4e}', 
              fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# 4-8. Gradient fits for each dimension
for dim in range(5):
    ax = plt.subplot(3, 3, 4 + dim)
    ax.scatter(grads_norm[:, dim], grads_pred_norm[:, dim], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    lims = [min(grads_norm[:, dim].min(), grads_pred_norm[:, dim].min()), 
            max(grads_norm[:, dim].max(), grads_pred_norm[:, dim].max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel(f'Actual ∂loss/∂{grad_dim_names[dim]} (norm)', fontsize=9)
    ax.set_ylabel(f'Predicted ∂loss/∂{grad_dim_names[dim]} (norm)', fontsize=9)
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
Regularization: 1e-8

BASIS FUNCTIONS
RBF Bases: {fitter.n_rbf_bases}
Polynomial Bases: {n_poly_bases}
Total: {fitter.n_rbf_bases + n_poly_bases}

NORMALIZED PERFORMANCE
Function RMSE: {rmse_norm:.6f}
Function R²: {r2_norm:.4f}
Gradient RMSE: {grad_rmse_norm:.6f}

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
    ax.scatter(X[:, dim], residuals_norm, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel(grad_dim_names[dim], fontsize=10)
    ax.set_ylabel('Residuals (normalized)', fontsize=10)
    ax.set_title(f'Residuals vs {grad_dim_names[dim]}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Absolute errors histogram on last subplot
ax = axes[1, 2]
abs_errors = np.abs(residuals_norm)
ax.hist(abs_errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.set_xlabel('Absolute Error (normalized)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title(f'Absolute Error Distribution\nMedian = {np.median(abs_errors):.4f}', 
             fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('fit_samples_residual_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Residual analysis plot saved to: fit_samples_residual_analysis.png")
plt.close()

# Summary statistics
print(f"\n" + "="*70)
print("RESIDUAL STATISTICS (normalized scale)")
print("="*70)
print(f"Percentiles (function residuals):")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"  {p:2d}th: {np.percentile(residuals_norm, p):12.6f}")

print(f"\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print(f"""
The model has been fit on normalized data for numerical stability.
All metrics shown are in normalized units (standard deviations).

A normalized RMSE of {rmse_norm:.4f} means the typical prediction 
error is {rmse_norm:.4f} standard deviations, which is
{'EXCELLENT' if rmse_norm < 0.2 else 'GOOD' if rmse_norm < 0.5 else 'MODERATE' if rmse_norm < 1.0 else 'POOR'}.

The R² of {r2_norm:.4f} indicates the model explains 
{r2_norm*100:.1f}% of the variance in the normalized loss.

For gradients, a normalized RMSE of {grad_rmse_norm:.4f} is
{'EXCELLENT' if grad_rmse_norm < 0.3 else 'GOOD' if grad_rmse_norm < 0.7 else 'MODERATE' if grad_rmse_norm < 1.5 else 'POOR'}.
""")

print(f"✓ Analysis complete!")
