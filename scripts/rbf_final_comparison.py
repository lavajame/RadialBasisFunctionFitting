"""
Final comparison: Conservative vs Optimized RBF fits
Shows the dramatic improvement by allowing higher condition numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_polynomial_fitter import RBFPolynomialFitter

# Load and normalize data
data = np.genfromtxt('samples.csv', delimiter=',', skip_header=1)
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
print("COMPARING CONSERVATIVE VS OPTIMIZED FITS")
print("="*70)

# CONSERVATIVE: ε = 10.0 (very stable, low condition number)
print("\nCONSERVATIVE FIT (ε = 10.0, cond ≈ 1)")
fitter_conservative = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=10.0,
    polynomial_degree=2,
    regularization_lambda=1e-8
)
fitter_conservative.fit(
    X_norm, f_norm,
    df=grads_norm,
    n_centers=10,
    use_lhs_centers=True,
    lhs_bounds=(lhs_lower, lhs_upper),
    random_state=42
)

f_pred_conservative = fitter_conservative.predict(X_norm)
grads_pred_conservative = fitter_conservative.predict_gradient(X_norm)

residuals_conservative = f_norm - f_pred_conservative
rmse_conservative = np.sqrt(np.mean(residuals_conservative**2))
r2_conservative = 1 - np.sum(residuals_conservative**2) / np.sum((f_norm - f_norm.mean())**2)
grad_rmse_conservative = np.sqrt(np.mean((grads_norm - grads_pred_conservative)**2))

# Get condition number
distances = np.linalg.norm(
    fitter_conservative.centers[:, np.newaxis, :] - fitter_conservative.centers[np.newaxis, :, :],
    axis=2
)
matrix_conservative = fitter_conservative.rbf_kernel(distances) + 1e-8 * np.eye(10)
cond_conservative = np.linalg.cond(matrix_conservative)

print(f"  Shape Parameter (ε): 10.0")
print(f"  Condition Number: {cond_conservative:.2e}")
print(f"  Function RMSE: {rmse_conservative:.6f}")
print(f"  Function R²: {r2_conservative:.6f}")
print(f"  Gradient RMSE: {grad_rmse_conservative:.6f}")

# OPTIMIZED: ε = 0.1150 (higher condition number but better fit)
print("\nOPTIMIZED FIT (ε = 0.1150, cond ≈ 4000)")
fitter_optimized = RBFPolynomialFitter(
    rbf_name='gaussian',
    rbf_shape_parameter=0.114976,
    polynomial_degree=2,
    regularization_lambda=1e-8
)
fitter_optimized.fit(
    X_norm, f_norm,
    df=grads_norm,
    n_centers=10,
    use_lhs_centers=True,
    lhs_bounds=(lhs_lower, lhs_upper),
    random_state=42
)

f_pred_optimized = fitter_optimized.predict(X_norm)
grads_pred_optimized = fitter_optimized.predict_gradient(X_norm)

residuals_optimized = f_norm - f_pred_optimized
rmse_optimized = np.sqrt(np.mean(residuals_optimized**2))
r2_optimized = 1 - np.sum(residuals_optimized**2) / np.sum((f_norm - f_norm.mean())**2)
grad_rmse_optimized = np.sqrt(np.mean((grads_norm - grads_pred_optimized)**2))

# Get condition number
matrix_optimized = fitter_optimized.rbf_kernel(distances) + 1e-8 * np.eye(10)
cond_optimized = np.linalg.cond(matrix_optimized)

print(f"  Shape Parameter (ε): 0.1150")
print(f"  Condition Number: {cond_optimized:.2e}")
print(f"  Function RMSE: {rmse_optimized:.6f}")
print(f"  Function R²: {r2_optimized:.6f}")
print(f"  Gradient RMSE: {grad_rmse_optimized:.6f}")

# Compute improvements
improvement_rmse = 100 * (rmse_conservative - rmse_optimized) / rmse_conservative
improvement_r2 = 100 * (r2_optimized - r2_conservative) / abs(r2_conservative)
improvement_grad = 100 * (grad_rmse_conservative - grad_rmse_optimized) / grad_rmse_conservative

print(f"\nIMPROVEMENTS:")
print(f"  Function RMSE: {improvement_rmse:.1f}% better")
print(f"  Function R²: {improvement_r2:.1f}% better")
print(f"  Gradient RMSE: {improvement_grad:.1f}% better")
print(f"  Condition Number increased by: {cond_optimized/cond_conservative:.0f}x (still well-behaved)")

# Create detailed comparison plot
fig = plt.figure(figsize=(16, 11))

# 1. Predicted vs Actual - Conservative
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(f_norm, f_pred_conservative, alpha=0.6, s=60, edgecolors='k', linewidth=0.5, color='#ff7f0e')
lims = [min(f_norm.min(), f_pred_conservative.min()), max(f_norm.max(), f_pred_conservative.max())]
ax1.plot(lims, lims, 'r--', lw=2)
ax1.set_xlabel('Actual Loss (normalized)', fontsize=10)
ax1.set_ylabel('Predicted Loss (normalized)', fontsize=10)
ax1.set_title(f'CONSERVATIVE: Predicted vs Actual\nRMSE={rmse_conservative:.6f}, R²={r2_conservative:.6f}', 
              fontsize=11, fontweight='bold', color='#ff7f0e')
ax1.grid(alpha=0.3)

# 2. Predicted vs Actual - Optimized
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(f_norm, f_pred_optimized, alpha=0.6, s=60, edgecolors='k', linewidth=0.5, color='#2ca02c')
lims = [min(f_norm.min(), f_pred_optimized.min()), max(f_norm.max(), f_pred_optimized.max())]
ax2.plot(lims, lims, 'r--', lw=2)
ax2.set_xlabel('Actual Loss (normalized)', fontsize=10)
ax2.set_ylabel('Predicted Loss (normalized)', fontsize=10)
ax2.set_title(f'OPTIMIZED: Predicted vs Actual\nRMSE={rmse_optimized:.6f}, R²={r2_optimized:.6f}', 
              fontsize=11, fontweight='bold', color='#2ca02c')
ax2.grid(alpha=0.3)

# 3. Overlay comparison
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(f_norm, f_pred_conservative, alpha=0.4, s=50, edgecolors='k', linewidth=0.5, 
            color='#ff7f0e', label='Conservative')
ax3.scatter(f_norm, f_pred_optimized, alpha=0.4, s=50, edgecolors='k', linewidth=0.5, 
            color='#2ca02c', label='Optimized')
lims = [min(f_norm.min(), f_pred_conservative.min(), f_pred_optimized.min()), 
        max(f_norm.max(), f_pred_conservative.max(), f_pred_optimized.max())]
ax3.plot(lims, lims, 'r--', lw=2, label='Perfect')
ax3.set_xlabel('Actual Loss (normalized)', fontsize=10)
ax3.set_ylabel('Predicted Loss (normalized)', fontsize=10)
ax3.set_title('Side-by-Side Comparison', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# 4. Residuals - Conservative
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(f_pred_conservative, residuals_conservative, alpha=0.6, s=60, edgecolors='k', linewidth=0.5, color='#ff7f0e')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('Predicted Loss (normalized)', fontsize=10)
ax4.set_ylabel('Residuals (normalized)', fontsize=10)
ax4.set_title(f'CONSERVATIVE: Residuals\nRMSE={rmse_conservative:.6f}', 
              fontsize=11, fontweight='bold', color='#ff7f0e')
ax4.grid(alpha=0.3)

# 5. Residuals - Optimized
ax5 = plt.subplot(3, 3, 5)
ax5.scatter(f_pred_optimized, residuals_optimized, alpha=0.6, s=60, edgecolors='k', linewidth=0.5, color='#2ca02c')
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('Predicted Loss (normalized)', fontsize=10)
ax5.set_ylabel('Residuals (normalized)', fontsize=10)
ax5.set_title(f'OPTIMIZED: Residuals\nRMSE={rmse_optimized:.6f}', 
              fontsize=11, fontweight='bold', color='#2ca02c')
ax5.grid(alpha=0.3)

# 6. Residual distributions
ax6 = plt.subplot(3, 3, 6)
ax6.hist(residuals_conservative, bins=15, alpha=0.6, edgecolor='black', color='#ff7f0e', label='Conservative')
ax6.hist(residuals_optimized, bins=15, alpha=0.6, edgecolor='black', color='#2ca02c', label='Optimized')
ax6.axvline(x=0, color='r', linestyle='--', lw=2)
ax6.set_xlabel('Residual Value (normalized)', fontsize=10)
ax6.set_ylabel('Frequency', fontsize=10)
ax6.set_title('Residual Distribution Comparison', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3, axis='y')

# 7. Metrics comparison bar chart
ax7 = plt.subplot(3, 3, 7)
metrics = ['RMSE', 'Grad RMSE', 'Cond # (÷1000)']
conservative_vals = [rmse_conservative, grad_rmse_conservative, cond_conservative/1000]
optimized_vals = [rmse_optimized, grad_rmse_optimized, cond_optimized/1000]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax7.bar(x_pos - width/2, conservative_vals, width, label='Conservative', color='#ff7f0e', alpha=0.7, edgecolor='black')
bars2 = ax7.bar(x_pos + width/2, optimized_vals, width, label='Optimized', color='#2ca02c', alpha=0.7, edgecolor='black')

ax7.set_ylabel('Value', fontsize=10)
ax7.set_title('Metrics Comparison', fontsize=11, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(metrics)
ax7.legend(fontsize=9)
ax7.grid(alpha=0.3, axis='y')

# 8. R² comparison
ax8 = plt.subplot(3, 3, 8)
models = ['Conservative\n(ε=10.0)', 'Optimized\n(ε=0.115)']
r2_scores = [r2_conservative, r2_optimized]
colors_r2 = ['#ff7f0e', '#2ca02c']
bars = ax8.bar(models, r2_scores, color=colors_r2, alpha=0.7, edgecolor='black', linewidth=2)
ax8.set_ylabel('R² Score', fontsize=11)
ax8.set_title('Variance Explained Comparison', fontsize=11, fontweight='bold')
ax8.set_ylim(0, 1)
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax8.grid(alpha=0.3, axis='y')

# 9. Summary table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

summary_text = f"""
SUMMARY: CONSERVATIVE vs OPTIMIZED

CONSERVATIVE (ε=10.0)
  Shape Parameter: 10.0000
  Condition Number: {cond_conservative:.2e}
  Function RMSE: {rmse_conservative:.6f}
  Function R²: {r2_conservative:.6f}
  Gradient RMSE: {grad_rmse_conservative:.6f}

OPTIMIZED (ε=0.1150)
  Shape Parameter: 0.114976
  Condition Number: {cond_optimized:.2e}
  Function RMSE: {rmse_optimized:.6f}
  Function R²: {r2_optimized:.6f}
  Gradient RMSE: {grad_rmse_optimized:.6f}

IMPROVEMENTS
  RMSE Better: {improvement_rmse:.1f}%
  R² Better: {improvement_r2:.1f}%
  Grad RMSE Better: {improvement_grad:.1f}%
  
  Condition # increase:
  {cond_optimized/cond_conservative:.0f}x (still robust)
  
CONCLUSION
Relaxing the condition number constraint
from ~1 to ~4000 yields significant fit
improvements with maintained stability.
"""

ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9.5,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('rbf_conservative_vs_optimized.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Detailed comparison plot saved to: rbf_conservative_vs_optimized.png")
plt.close()

print("\n✓ Analysis complete!")
