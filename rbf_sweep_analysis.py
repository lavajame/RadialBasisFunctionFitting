"""
RBF Kernel Sweep Analysis with Automatic Shape Parameter Tuning

Sweeps through different RBF kernels, automatically selects shape parameters
based on maximum condition number threshold, and compares fit quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_polynomial_fitter import RBFPolynomialFitter
import warnings
warnings.filterwarnings('ignore')

# Load and normalize data
print("="*70)
print("LOADING AND NORMALIZING DATA")
print("="*70)

data = np.genfromtxt('samples.csv', delimiter=',', skip_header=1)
X = data[:, :5]
f = data[:, 5]
grads = data[:, 6:11]

grad_dim_names = ['q', 'Merton.sigma', 'Merton.lam', 'Merton.muJ', 'Merton.sigmaJ']

# Normalize each variable independently
X_mean = X.mean(axis=0)
X_std = X.std(axis=0) + 1e-10
X_norm = (X - X_mean) / X_std

f_mean = f.mean()
f_std = f.std() + 1e-10
f_norm = (f - f_mean) / f_std

grads_norm = np.zeros_like(grads)
grad_stats = []
for i in range(grads.shape[1]):
    g = grads[:, i]
    g_mean = g.mean()
    g_std = g.std() + 1e-10
    grads_norm[:, i] = (g - g_mean) / g_std
    grad_stats.append({'mean': g_mean, 'std': g_std})

print(f"✓ Data loaded: {X_norm.shape[0]} samples, {X_norm.shape[1]} features")

# LHS bounds for fitting
lhs_lower = X_norm.min(axis=0)
lhs_upper = X_norm.max(axis=0)

# RBF kernels to test
KERNELS_WITH_SHAPE = {
    'gaussian': (0.01, 10.0, 50),
    'multiquadric': (0.01, 10.0, 50),
    'inverse_multiquadric': (0.01, 10.0, 50),
}

KERNELS_NO_SHAPE = [
    'thin_plate_spline',
    'cubic',
    'linear',
]

MAX_CONDITION_NUMBER = 1e12  # Maximum acceptable condition number

def estimate_condition_number(X_centers, kernel_func, regularization=1e-8):
    """Estimate condition number of the system matrix."""
    n_centers = X_centers.shape[0]
    distances = np.linalg.norm(
        X_centers[:, np.newaxis, :] - X_centers[np.newaxis, :, :],
        axis=2
    )
    matrix = kernel_func(distances)
    
    # Add regularization to diagonal
    matrix = matrix + regularization * np.eye(matrix.shape[0])
    
    # Estimate condition number
    try:
        cond = np.linalg.cond(matrix)
        return cond
    except:
        return np.inf

def tune_shape_parameter(rbf_name, X_centers, lower=0.01, upper=10.0, n_steps=50):
    """Find optimal shape parameter based on max condition number."""
    shape_params = np.logspace(np.log10(lower), np.log10(upper), n_steps)
    best_param = None
    best_cond = np.inf
    
    for eps in shape_params:
        # Create fitter temporarily to get kernel function
        fitter_temp = RBFPolynomialFitter(
            rbf_name=rbf_name,
            rbf_shape_parameter=eps,
            polynomial_degree=2,
            regularization_lambda=1e-8
        )
        
        cond = estimate_condition_number(X_centers, fitter_temp.rbf_kernel)
        
        # Keep track of best parameter that doesn't exceed max condition number
        if cond <= MAX_CONDITION_NUMBER:
            best_param = eps
            best_cond = cond
    
    if best_param is None:
        # If nothing passes, use the one with best (lowest) condition number
        best_param = shape_params[0]
        fitter_temp = RBFPolynomialFitter(
            rbf_name=rbf_name,
            rbf_shape_parameter=best_param,
            polynomial_degree=2,
            regularization_lambda=1e-8
        )
        best_cond = estimate_condition_number(X_centers, fitter_temp.rbf_kernel)
    
    return best_param, best_cond

# Perform sweep
print("\n" + "="*70)
print("SWEEPING RBF KERNELS WITH AUTOMATIC SHAPE PARAMETER TUNING")
print("="*70)

results = {}

# Generate LHS centers
fitter_dummy = RBFPolynomialFitter()
X_centers = fitter_dummy._generate_lhs_centers(10, X_norm.shape[1], lhs_lower, lhs_upper, random_state=42)

# Test kernels with shape parameters
for rbf_name in KERNELS_WITH_SHAPE:
    print(f"\n{rbf_name.upper()}:")
    print("  Tuning shape parameter...", end=" ", flush=True)
    
    eps_opt, cond_opt = tune_shape_parameter(rbf_name, X_centers)
    print(f"ε = {eps_opt:.6f}, condition number = {cond_opt:.2e}")
    
    # Fit model
    fitter = RBFPolynomialFitter(
        rbf_name=rbf_name,
        rbf_shape_parameter=eps_opt,
        polynomial_degree=2,
        regularization_lambda=1e-8
    )
    
    fitter.fit(
        X_norm, f_norm,
        df=grads_norm,
        n_centers=10,
        use_lhs_centers=True,
        lhs_bounds=(lhs_lower, lhs_upper),
        random_state=42
    )
    
    # Predict and compute metrics
    f_pred_norm = fitter.predict(X_norm)
    grads_pred_norm = fitter.predict_gradient(X_norm)
    
    residuals_norm = f_norm - f_pred_norm
    rmse_norm = np.sqrt(np.mean(residuals_norm**2))
    r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)
    
    grad_residuals_norm = grads_norm - grads_pred_norm
    grad_rmse_norm = np.sqrt(np.mean(grad_residuals_norm**2))
    
    results[rbf_name] = {
        'eps': eps_opt,
        'cond': cond_opt,
        'rmse': rmse_norm,
        'r2': r2_norm,
        'grad_rmse': grad_rmse_norm,
        'f_pred': f_pred_norm,
        'grads_pred': grads_pred_norm,
        'residuals': residuals_norm,
        'grad_residuals': grad_residuals_norm,
    }
    
    print(f"  ✓ Fitted: RMSE = {rmse_norm:.6f}, R² = {r2_norm:.6f}, Grad RMSE = {grad_rmse_norm:.6f}")

# Test kernels without shape parameters
for rbf_name in KERNELS_NO_SHAPE:
    print(f"\n{rbf_name.upper()}:")
    
    fitter = RBFPolynomialFitter(
        rbf_name=rbf_name,
        polynomial_degree=2,
        regularization_lambda=1e-8
    )
    
    fitter.fit(
        X_norm, f_norm,
        df=grads_norm,
        n_centers=10,
        use_lhs_centers=True,
        lhs_bounds=(lhs_lower, lhs_upper),
        random_state=42
    )
    
    # Estimate condition number
    cond = estimate_condition_number(X_centers, fitter.rbf_kernel)
    
    # Predict and compute metrics
    f_pred_norm = fitter.predict(X_norm)
    grads_pred_norm = fitter.predict_gradient(X_norm)
    
    residuals_norm = f_norm - f_pred_norm
    rmse_norm = np.sqrt(np.mean(residuals_norm**2))
    r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)
    
    grad_residuals_norm = grads_norm - grads_pred_norm
    grad_rmse_norm = np.sqrt(np.mean(grad_residuals_norm**2))
    
    results[rbf_name] = {
        'eps': None,
        'cond': cond,
        'rmse': rmse_norm,
        'r2': r2_norm,
        'grad_rmse': grad_rmse_norm,
        'f_pred': f_pred_norm,
        'grads_pred': grads_pred_norm,
        'residuals': residuals_norm,
        'grad_residuals': grad_residuals_norm,
    }
    
    print(f"  Fitted: RMSE = {rmse_norm:.6f}, R² = {r2_norm:.6f}, Grad RMSE = {grad_rmse_norm:.6f}")
    print(f"  Condition number = {cond:.2e}")

# Print summary
print("\n" + "="*70)
print("SUMMARY: RBF KERNEL COMPARISON")
print("="*70)

kernel_names = list(results.keys())
rmse_values = [results[k]['rmse'] for k in kernel_names]
r2_values = [results[k]['r2'] for k in kernel_names]
grad_rmse_values = [results[k]['grad_rmse'] for k in kernel_names]
cond_values = [results[k]['cond'] for k in kernel_names]

# Print detailed table
print(f"\n{'Kernel':<20} {'Shape Param':<15} {'Cond Number':<15} {'RMSE':<12} {'R²':<12} {'Grad RMSE':<12}")
print("-" * 90)
for kernel in kernel_names:
    eps_str = f"{results[kernel]['eps']:.6f}" if results[kernel]['eps'] is not None else "N/A"
    cond_str = f"{results[kernel]['cond']:.2e}"
    rmse_str = f"{results[kernel]['rmse']:.6f}"
    r2_str = f"{results[kernel]['r2']:.6f}"
    grad_rmse_str = f"{results[kernel]['grad_rmse']:.6f}"
    print(f"{kernel:<20} {eps_str:<15} {cond_str:<15} {rmse_str:<12} {r2_str:<12} {grad_rmse_str:<12}")

# Find best kernels
best_rmse_kernel = min(kernel_names, key=lambda k: results[k]['rmse'])
best_r2_kernel = max(kernel_names, key=lambda k: results[k]['r2'])
best_grad_rmse_kernel = min(kernel_names, key=lambda k: results[k]['grad_rmse'])

print(f"\n{'Best RMSE':<30}: {best_rmse_kernel} ({results[best_rmse_kernel]['rmse']:.6f})")
print(f"{'Best R²':<30}: {best_r2_kernel} ({results[best_r2_kernel]['r2']:.6f})")
print(f"{'Best Gradient RMSE':<30}: {best_grad_rmse_kernel} ({results[best_grad_rmse_kernel]['grad_rmse']:.6f})")

# Create comparison plots
fig = plt.figure(figsize=(16, 12))

# 1. RMSE Comparison
ax1 = plt.subplot(3, 3, 1)
colors = ['#1f77b4' if k in KERNELS_WITH_SHAPE else '#ff7f0e' for k in kernel_names]
bars = ax1.bar(range(len(kernel_names)), rmse_values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(kernel_names)))
ax1.set_xticklabels(kernel_names, rotation=45, ha='right')
ax1.set_ylabel('RMSE (normalized)', fontsize=11)
ax1.set_title('Function Fit RMSE Comparison', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3, axis='y')
# Highlight best
best_idx = np.argmin(rmse_values)
bars[best_idx].set_color('#2ca02c')
bars[best_idx].set_edgecolor('darkgreen')
bars[best_idx].set_linewidth(2)

# 2. R² Comparison
ax2 = plt.subplot(3, 3, 2)
bars = ax2.bar(range(len(kernel_names)), r2_values, color=colors, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(kernel_names)))
ax2.set_xticklabels(kernel_names, rotation=45, ha='right')
ax2.set_ylabel('R² Score', fontsize=11)
ax2.set_title('Function Fit R² Comparison', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
# Highlight best
best_idx = np.argmax(r2_values)
bars[best_idx].set_color('#2ca02c')
bars[best_idx].set_edgecolor('darkgreen')
bars[best_idx].set_linewidth(2)

# 3. Gradient RMSE Comparison
ax3 = plt.subplot(3, 3, 3)
bars = ax3.bar(range(len(kernel_names)), grad_rmse_values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(kernel_names)))
ax3.set_xticklabels(kernel_names, rotation=45, ha='right')
ax3.set_ylabel('Gradient RMSE (normalized)', fontsize=11)
ax3.set_title('Gradient Fit RMSE Comparison', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')
# Highlight best
best_idx = np.argmin(grad_rmse_values)
bars[best_idx].set_color('#2ca02c')
bars[best_idx].set_edgecolor('darkgreen')
bars[best_idx].set_linewidth(2)

# 4. Condition Number Comparison (log scale)
ax4 = plt.subplot(3, 3, 4)
bars = ax4.bar(range(len(kernel_names)), cond_values, color=colors, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(kernel_names)))
ax4.set_xticklabels(kernel_names, rotation=45, ha='right')
ax4.set_ylabel('Condition Number', fontsize=11)
ax4.set_yscale('log')
ax4.set_title('Condition Number Comparison (log scale)', fontsize=12, fontweight='bold')
ax4.axhline(y=MAX_CONDITION_NUMBER, color='r', linestyle='--', linewidth=2, label=f'Max Threshold ({MAX_CONDITION_NUMBER:.0e})')
ax4.grid(alpha=0.3, which='both')
ax4.legend()

# 5-7. Predicted vs Actual for best three kernels
best_three = sorted(kernel_names, key=lambda k: results[k]['rmse'])[:3]
for idx, kernel in enumerate(best_three):
    ax = plt.subplot(3, 3, 5 + idx)
    f_pred = results[kernel]['f_pred']
    ax.scatter(f_norm, f_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    lims = [min(f_norm.min(), f_pred.min()), max(f_norm.max(), f_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2)
    ax.set_xlabel('Actual Loss (normalized)', fontsize=9)
    ax.set_ylabel('Predicted Loss (normalized)', fontsize=9)
    title = f"{kernel}: Predicted vs Actual\nRMSE={results[kernel]['rmse']:.4f}, R²={results[kernel]['r2']:.4f}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

# 8. Residuals for best kernel
ax8 = plt.subplot(3, 3, 8)
best_kernel = best_rmse_kernel
residuals = results[best_kernel]['residuals']
ax8.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax8.axvline(x=0, color='r', linestyle='--', lw=2)
ax8.set_xlabel('Residual Value (normalized)', fontsize=10)
ax8.set_ylabel('Frequency', fontsize=10)
ax8.set_title(f'Best Kernel ({best_kernel})\nResidual Distribution', fontsize=11, fontweight='bold')
ax8.grid(alpha=0.3, axis='y')

# 9. Summary table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = f"""
SWEEP RESULTS SUMMARY

Best Function Fit:
  Kernel: {best_rmse_kernel}
  RMSE: {results[best_rmse_kernel]['rmse']:.6f}
  R²: {results[best_rmse_kernel]['r2']:.6f}
  
Best Variance Explained:
  Kernel: {best_r2_kernel}
  R²: {results[best_r2_kernel]['r2']:.6f}
  
Best Gradients:
  Kernel: {best_grad_rmse_kernel}
  RMSE: {results[best_grad_rmse_kernel]['grad_rmse']:.6f}

Tuning Settings:
  Max Condition Number: {MAX_CONDITION_NUMBER:.0e}
  RBF Centers: 10 (LHS)
  Polynomial Degree: 2
  Regularization: 1e-8

Training Data:
  Samples: 50
  Features: 5
  
Legend:
  Blue: Kernels with shape parameter
  Orange: Fixed kernels (no shape param)
  Green: Best in category
"""
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('rbf_sweep_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Comparison plot saved to: rbf_sweep_comparison.png")
plt.close()

# Create detailed per-kernel residual analysis
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Residual Analysis by RBF Kernel', fontsize=14, fontweight='bold')

for idx, kernel in enumerate(kernel_names):
    ax = axes[idx // 3, idx % 3]
    residuals = results[kernel]['residuals']
    ax.scatter(results[kernel]['f_pred'], residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Loss (normalized)', fontsize=9)
    ax.set_ylabel('Residuals (normalized)', fontsize=9)
    title = f"{kernel}\nRMSE={results[kernel]['rmse']:.4f}"
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(alpha=0.3)

# Remove empty subplot
axes[1, 2].axis('off')
summary_text = f"""
KERNEL RANKINGS

By RMSE (Lower is Better):
"""
sorted_by_rmse = sorted(kernel_names, key=lambda k: results[k]['rmse'])
for i, k in enumerate(sorted_by_rmse, 1):
    summary_text += f"\n{i}. {k}: {results[k]['rmse']:.6f}"

summary_text += f"\n\nBy R² (Higher is Better):"
sorted_by_r2 = sorted(kernel_names, key=lambda k: results[k]['r2'], reverse=True)
for i, k in enumerate(sorted_by_r2, 1):
    summary_text += f"\n{i}. {k}: {results[k]['r2']:.6f}"

axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('rbf_sweep_residuals.png', dpi=150, bbox_inches='tight')
print(f"✓ Residual analysis plot saved to: rbf_sweep_residuals.png")
plt.close()

print("\n✓ RBF sweep analysis complete!")
