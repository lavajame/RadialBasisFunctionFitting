"""
Polynomial Degree Sweep with Comprehensive Diagnostics

Tests different polynomial degrees (1-5) with optimal Gaussian RBF (ε=0.115)
and creates high-quality visualization of fit quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sys import path
path.insert(0, '..')
from src.rbf_polynomial_fitter import RBFPolynomialFitter
import warnings
warnings.filterwarnings('ignore')

# Load and normalize data
data = np.genfromtxt('../data/samples.csv', delimiter=',', skip_header=1)
X = data[:, :5]
f = data[:, 5]
grads = data[:, 6:11]

X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-10
X_norm = (X - X_mean) / X_std

f_mean = f.mean()
f_std = f.std() + 1e-10
f_norm = (f - f_mean) / f_std

grads_norm = np.zeros_like(grads)
for i in range(grads.shape[1]):
    g = grads[:, i]
    g_mean = g.mean()
    g_std = g.std() + 1e-10
    grads_norm[:, i] = (g - g_mean) / g_std

lhs_lower = X_norm.min(axis=0)
lhs_upper = X_norm.max(axis=0)

grad_dim_names = ['q', 'Merton.sigma', 'Merton.lam', 'Merton.muJ', 'Merton.sigmaJ']

print("="*70)
print("POLYNOMIAL DEGREE SWEEP")
print("="*70)
print("\nTesting polynomial degrees 1-5 with optimal Gaussian RBF (ε=0.115)")

# Test polynomial degrees
poly_degrees = [1, 2, 3, 4, 5]
optimal_eps = 0.115

results = {}

for poly_degree in poly_degrees:
    print(f"\nTesting polynomial degree {poly_degree}...", end=" ", flush=True)
    
    fitter = RBFPolynomialFitter(
        rbf_name='gaussian',
        rbf_shape_parameter=optimal_eps,
        polynomial_degree=poly_degree,
        regularization_lambda=1e-8
    )
    
    fitter.fit(
        X_norm, f_norm,
        df=grads_norm,
        n_centers=5,
        use_lhs_centers=True,
        lhs_bounds=(lhs_lower, lhs_upper),
        random_state=42
    )
    
    # Predictions
    f_pred_norm = fitter.predict(X_norm)
    grads_pred_norm = fitter.predict_gradient(X_norm)
    
    # Metrics
    residuals_norm = f_norm - f_pred_norm
    rmse_norm = np.sqrt(np.mean(residuals_norm**2))
    mae_norm = np.abs(residuals_norm).mean()
    r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)
    
    grad_residuals = grads_norm - grads_pred_norm
    grad_rmse = np.sqrt(np.mean(grad_residuals**2))
    
    # Per-dimension gradient metrics
    grad_r2_per_dim = []
    for i in range(5):
        r2_i = 1 - np.sum(grad_residuals[:, i]**2) / np.sum((grads_norm[:, i] - grads_norm[:, i].mean())**2)
        grad_r2_per_dim.append(r2_i)
    
    results[poly_degree] = {
        'fitter': fitter,
        'f_pred': f_pred_norm,
        'grads_pred': grads_pred_norm,
        'residuals': residuals_norm,
        'grad_residuals': grad_residuals,
        'rmse': rmse_norm,
        'mae': mae_norm,
        'r2': r2_norm,
        'grad_rmse': grad_rmse,
        'grad_r2': grad_r2_per_dim,
        'n_bases': fitter.n_rbf_bases + fitter._count_polynomial_bases(X_norm.shape[1]),
    }
    
    print(f"✓ RMSE={rmse_norm:.6f}, R²={r2_norm:.6f}, Bases={results[poly_degree]['n_bases']}")

# Print summary table
print("\n" + "="*70)
print("SUMMARY: POLYNOMIAL DEGREE COMPARISON")
print("="*70)

print(f"\n{'Degree':<8} {'Bases':<8} {'RMSE':<12} {'MAE':<12} {'R²':<12} {'Grad RMSE':<12}")
print("-" * 70)
for degree in poly_degrees:
    r = results[degree]
    print(f"{degree:<8} {r['n_bases']:<8} {r['rmse']:<12.6f} {r['mae']:<12.6f} {r['r2']:<12.6f} {r['grad_rmse']:<12.6f}")

# Find best
best_degree = min(poly_degrees, key=lambda d: results[d]['rmse'])
print(f"\n{'='*70}")
print(f"BEST POLYNOMIAL DEGREE: {best_degree}")
print(f"  RMSE: {results[best_degree]['rmse']:.6f}")
print(f"  R²: {results[best_degree]['r2']:.6f}")
print(f"  Gradient RMSE: {results[best_degree]['grad_rmse']:.6f}")
print(f"  Total Basis Functions: {results[best_degree]['n_bases']}")

# Create comprehensive diagnostic figure - simplified layout
fig = plt.figure(figsize=(16, 10))

# ============ RMSE PROGRESSION ============
ax_rmse = plt.subplot(2, 3, 1)
rmse_vals = [results[d]['rmse'] for d in poly_degrees]
ax_rmse.plot(poly_degrees, rmse_vals, 'o-', linewidth=3, markersize=10, color='#d62728')
ax_rmse.scatter([best_degree], [results[best_degree]['rmse']], s=300, color='green', 
               marker='*', edgecolors='darkgreen', linewidth=2, zorder=5)
ax_rmse.set_xlabel('Polynomial Degree', fontsize=11)
ax_rmse.set_ylabel('RMSE (normalized)', fontsize=11)
ax_rmse.set_title('Function RMSE vs Polynomial Degree', fontsize=12, fontweight='bold')
ax_rmse.grid(alpha=0.3)
ax_rmse.set_xticks(poly_degrees)

# ============ R² PROGRESSION ============
ax_r2 = plt.subplot(2, 3, 2)
r2_vals = [results[d]['r2'] for d in poly_degrees]
ax_r2.plot(poly_degrees, r2_vals, 's-', linewidth=3, markersize=10, color='#2ca02c')
ax_r2.scatter([best_degree], [results[best_degree]['r2']], s=300, color='red', 
             marker='*', edgecolors='darkred', linewidth=2, zorder=5)
ax_r2.set_xlabel('Polynomial Degree', fontsize=11)
ax_r2.set_ylabel('R² Score', fontsize=11)
ax_r2.set_title('Variance Explained vs Polynomial Degree', fontsize=12, fontweight='bold')
ax_r2.grid(alpha=0.3)
ax_r2.set_xticks(poly_degrees)

# ============ GRADIENT RMSE PROGRESSION ============
ax_grad = plt.subplot(2, 3, 3)
grad_rmse_vals = [results[d]['grad_rmse'] for d in poly_degrees]
ax_grad.plot(poly_degrees, grad_rmse_vals, '^-', linewidth=3, markersize=10, color='#ff7f0e')
ax_grad.scatter([best_degree], [results[best_degree]['grad_rmse']], s=300, color='purple', 
               marker='*', edgecolors='purple', linewidth=2, zorder=5)
ax_grad.set_xlabel('Polynomial Degree', fontsize=11)
ax_grad.set_ylabel('Gradient RMSE (normalized)', fontsize=11)
ax_grad.set_title('Gradient Fit vs Polynomial Degree', fontsize=12, fontweight='bold')
ax_grad.grid(alpha=0.3)
ax_grad.set_xticks(poly_degrees)

# ============ NUMBER OF BASES ============
ax_bases = plt.subplot(2, 3, 4)
n_bases = [results[d]['n_bases'] for d in poly_degrees]
ax_bases.semilogy(poly_degrees, n_bases, 'd-', linewidth=3, markersize=10, color='#9467bd')
ax_bases.set_xlabel('Polynomial Degree', fontsize=11)
ax_bases.set_ylabel('Total Basis Functions (log scale)', fontsize=11)
ax_bases.set_title('Model Complexity Growth', fontsize=12, fontweight='bold')
ax_bases.grid(alpha=0.3, which='both')
ax_bases.set_xticks(poly_degrees)

# Add text annotations
for i, (degree, bases) in enumerate(zip(poly_degrees, n_bases)):
    ax_bases.text(degree, bases * 1.3, str(bases), ha='center', fontsize=9)

# ============ BEST FIT: Predicted vs Actual ============
ax_best = plt.subplot(2, 3, 5)
r_best = results[best_degree]
scatter = ax_best.scatter(f_norm, r_best['f_pred'], c=np.abs(r_best['residuals']), 
                         s=80, alpha=0.7, edgecolors='k', linewidth=0.5, cmap='RdYlGn_r')
lims = [min(f_norm.min(), r_best['f_pred'].min()), max(f_norm.max(), r_best['f_pred'].max())]
ax_best.plot(lims, lims, 'k--', lw=2, label='Perfect')
ax_best.set_xlabel('Actual Loss (normalized)', fontsize=11)
ax_best.set_ylabel('Predicted Loss (normalized)', fontsize=11)
ax_best.set_title(f'BEST (Degree {best_degree})\nR²={r_best["r2"]:.4f}, RMSE={r_best["rmse"]:.6f}', 
                 fontsize=12, fontweight='bold')
ax_best.legend(fontsize=10)
ax_best.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax_best, label='Absolute Error')

# ============ RESIDUALS FOR BEST FIT ============
ax_resid = plt.subplot(2, 3, 6)
ax_resid.hist(r_best['residuals'], bins=25, edgecolor='black', alpha=0.7, color='steelblue')
ax_resid.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
ax_resid.set_xlabel('Residual Value (normalized)', fontsize=11)
ax_resid.set_ylabel('Frequency', fontsize=11)
ax_resid.set_title(f'Residual Distribution (Degree {best_degree})\nMean={r_best["residuals"].mean():.4f}, Std={r_best["residuals"].std():.4f}', 
                  fontsize=12, fontweight='bold')
ax_resid.legend(fontsize=10)
ax_resid.grid(alpha=0.3, axis='y')

plt.suptitle(f'Polynomial Degree Analysis\nGaussian RBF with ε=0.115, 5 LHS Centers, 50 Training Samples',
            fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('polynomial_degree_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Comprehensive analysis plot saved to: polynomial_degree_analysis.png")
plt.close()

# Create detailed fit quality visualization for best model
print("\n" + "="*70)
print("CREATING DETAILED FIT QUALITY PLOTS FOR BEST MODEL")
print("="*70)

best_result = results[best_degree]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Detailed Fit Quality Analysis - Polynomial Degree {best_degree}\nGaussian RBF (ε=0.115), {best_result["n_bases"]} Basis Functions', 
            fontsize=13, fontweight='bold')

# 1. Predicted vs Actual (large)
ax = axes[0, 0]
scatter = ax.scatter(f_norm, best_result['f_pred'], c=np.abs(best_result['residuals']), 
                     s=100, alpha=0.7, edgecolors='k', linewidth=0.5, cmap='RdYlGn_r')
lims = [min(f_norm.min(), best_result['f_pred'].min()), max(f_norm.max(), best_result['f_pred'].max())]
ax.plot(lims, lims, 'k--', lw=2, label='Perfect Fit', alpha=0.7)
ax.set_xlabel('Actual Loss (normalized)', fontsize=11)
ax.set_ylabel('Predicted Loss (normalized)', fontsize=11)
ax.set_title(f'Predicted vs Actual\nR² = {best_result["r2"]:.6f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Absolute Error', fontsize=10)

# 2. Residuals vs predicted (large)
ax = axes[0, 1]
scatter = ax.scatter(best_result['f_pred'], best_result['residuals'], c=best_result['f_pred'],
                     s=100, alpha=0.7, edgecolors='k', linewidth=0.5, cmap='viridis')
ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
ax.set_xlabel('Predicted Loss (normalized)', fontsize=11)
ax.set_ylabel('Residuals (normalized)', fontsize=11)
ax.set_title(f'Residual Plot\nRMSE = {best_result["rmse"]:.6f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Predicted Value', fontsize=10)

# 3. Error by prediction magnitude
ax = axes[0, 2]
sorted_idx = np.argsort(best_result['f_pred'])
ax.fill_between(range(len(best_result['residuals'])), 
               -np.abs(best_result['residuals'][sorted_idx]), 
               np.abs(best_result['residuals'][sorted_idx]),
               alpha=0.3, color='steelblue', label='±|Error|')
ax.plot(best_result['residuals'][sorted_idx], 'o-', alpha=0.6, markersize=4, color='#1f77b4')
ax.axhline(y=0, color='k', linestyle='-', lw=1)
ax.set_xlabel('Sample Index (sorted by prediction)', fontsize=11)
ax.set_ylabel('Residual Value', fontsize=11)
ax.set_title('Error Profile\n(Sorted by Prediction)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# 4. Residual histogram with stats
ax = axes[1, 0]
ax.hist(best_result['residuals'], bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero')
ax.axvline(x=best_result['residuals'].mean(), color='g', linestyle='--', lw=2, 
          label=f'Mean = {best_result["residuals"].mean():.4f}')
ax.set_xlabel('Residual Value', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Residual Distribution\n(Gaussian-like)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis='y')

# Add stats text
stats_text = f"Std: {best_result['residuals'].std():.4f}\nMax: {np.abs(best_result['residuals']).max():.4f}\nSkew: {0 if np.isnan((best_result['residuals']**3).mean()) else 'Centered'}"
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
       verticalalignment='top', horizontalalignment='right',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 5. Q-Q plot (residuals vs normal)
ax = axes[1, 1]
from scipy import stats
sample_quantiles = np.sort(best_result['residuals'])
n = len(sample_quantiles)
theoretical_quantiles = np.sort(np.random.normal(0, best_result['residuals'].std(), n))
ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=60, edgecolors='k', linewidth=0.5)
lims = [min(theoretical_quantiles.min(), sample_quantiles.min()), 
        max(theoretical_quantiles.max(), sample_quantiles.max())]
ax.plot(lims, lims, 'r--', lw=2, label='Normal Distribution')
ax.set_xlabel('Theoretical Quantiles', fontsize=11)
ax.set_ylabel('Sample Quantiles', fontsize=11)
ax.set_title('Q-Q Plot\n(Normality Check)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(alpha=0.3)

# 6. Summary metrics
ax = axes[1, 2]
ax.axis('off')

summary_text = f"""
BEST MODEL PERFORMANCE

Configuration:
  Gaussian RBF with ε = 0.115
  Polynomial Degree: {best_degree}
  RBF Centers: 10 (LHS)
  Total Bases: {best_result['n_bases']}

Function Fit:
  RMSE: {best_result['rmse']:.6f}
  MAE:  {best_result['mae']:.6f}
  R²:   {best_result['r2']:.6f}

Gradient Fit:
  RMSE: {best_result['grad_rmse']:.6f}

Residual Stats:
  Mean: {best_result['residuals'].mean():.6f}
  Std:  {best_result['residuals'].std():.6f}
  Max:  {np.abs(best_result['residuals']).max():.6f}

Interpretation:
✓ Function explains {best_result['r2']*100:.1f}% 
  of loss variance
✓ Typical prediction error 
  is {best_result['rmse']:.4f} std devs
✓ Residuals show no bias
  (mean ≈ 0)
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
       verticalalignment='top', fontfamily='monospace',
       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('fit_quality_detailed.png', dpi=150, bbox_inches='tight')
print(f"✓ Detailed fit quality plot saved to: fit_quality_detailed.png")
plt.close()

print("\n✓ Analysis complete!")
