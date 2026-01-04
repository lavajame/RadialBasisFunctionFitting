"""
RBF Kernel Sweep with Variable Condition Number Thresholds

Explores how fit quality improves as we increase the maximum allowed
condition number threshold. Tests multiple thresholds to find the sweet spot
between numerical stability and model accuracy.
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

# Normalize
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

print(f"✓ Data loaded: {X_norm.shape[0]} samples, {X_norm.shape[1]} features")

lhs_lower = X_norm.min(axis=0)
lhs_upper = X_norm.max(axis=0)

# Generate LHS centers
fitter_dummy = RBFPolynomialFitter()
X_centers = fitter_dummy._generate_lhs_centers(10, X_norm.shape[1], lhs_lower, lhs_upper, random_state=42)

def estimate_condition_number(X_centers, kernel_func, regularization=1e-8):
    """Estimate condition number of the system matrix."""
    n_centers = X_centers.shape[0]
    distances = np.linalg.norm(
        X_centers[:, np.newaxis, :] - X_centers[np.newaxis, :, :],
        axis=2
    )
    matrix = kernel_func(distances)
    matrix = matrix + regularization * np.eye(matrix.shape[0])
    try:
        cond = np.linalg.cond(matrix)
        return cond
    except:
        return np.inf

def tune_shape_parameter(rbf_name, X_centers, max_cond, lower=0.01, upper=10.0, n_steps=50):
    """Find optimal shape parameter for given max condition number."""
    shape_params = np.logspace(np.log10(lower), np.log10(upper), n_steps)
    best_param = None
    best_cond = np.inf
    
    for eps in shape_params:
        fitter_temp = RBFPolynomialFitter(
            rbf_name=rbf_name,
            rbf_shape_parameter=eps,
            polynomial_degree=2,
            regularization_lambda=1e-8
        )
        
        cond = estimate_condition_number(X_centers, fitter_temp.rbf_kernel)
        
        if cond <= max_cond:
            best_param = eps
            best_cond = cond
    
    if best_param is None:
        best_param = shape_params[0]
        fitter_temp = RBFPolynomialFitter(
            rbf_name=rbf_name,
            rbf_shape_parameter=best_param,
            polynomial_degree=2,
            regularization_lambda=1e-8
        )
        best_cond = estimate_condition_number(X_centers, fitter_temp.rbf_kernel)
    
    return best_param, best_cond

# Test different condition number thresholds
print("\n" + "="*70)
print("TESTING MULTIPLE CONDITION NUMBER THRESHOLDS")
print("="*70)

condition_thresholds = [1e6, 1e8, 1e10, 1e12, 1e14, 1e16, np.inf]
threshold_names = ['1e6', '1e8', '1e10', '1e12', '1e14', '1e16', 'Unlimited']

kernels_with_shape = ['gaussian', 'multiquadric', 'inverse_multiquadric']
kernels_no_shape = ['thin_plate_spline', 'cubic', 'linear']

# Store results: threshold -> kernel -> metrics
threshold_results = {}

for threshold_idx, max_cond in enumerate(condition_thresholds):
    print(f"\n{'='*70}")
    print(f"THRESHOLD: {threshold_names[threshold_idx]} (max cond = {max_cond:.0e})" if max_cond != np.inf else f"THRESHOLD: {threshold_names[threshold_idx]} (unlimited)")
    print(f"{'='*70}")
    
    threshold_results[threshold_names[threshold_idx]] = {}
    
    # Test kernels with shape parameters
    for rbf_name in kernels_with_shape:
        eps_opt, cond_opt = tune_shape_parameter(rbf_name, X_centers, max_cond)
        
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
        
        f_pred_norm = fitter.predict(X_norm)
        grads_pred_norm = fitter.predict_gradient(X_norm)
        
        residuals_norm = f_norm - f_pred_norm
        rmse_norm = np.sqrt(np.mean(residuals_norm**2))
        r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)
        
        grad_residuals_norm = grads_norm - grads_pred_norm
        grad_rmse_norm = np.sqrt(np.mean(grad_residuals_norm**2))
        
        threshold_results[threshold_names[threshold_idx]][rbf_name] = {
            'eps': eps_opt,
            'cond': cond_opt,
            'rmse': rmse_norm,
            'r2': r2_norm,
            'grad_rmse': grad_rmse_norm,
        }
        
        print(f"{rbf_name:<25} ε={eps_opt:8.6f}  cond={cond_opt:10.2e}  RMSE={rmse_norm:.6f}  R²={r2_norm:.6f}")
    
    # Test fixed kernels
    for rbf_name in kernels_no_shape:
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
        
        cond = estimate_condition_number(X_centers, fitter.rbf_kernel)
        
        f_pred_norm = fitter.predict(X_norm)
        grads_pred_norm = fitter.predict_gradient(X_norm)
        
        residuals_norm = f_norm - f_pred_norm
        rmse_norm = np.sqrt(np.mean(residuals_norm**2))
        r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)
        
        grad_residuals_norm = grads_norm - grads_pred_norm
        grad_rmse_norm = np.sqrt(np.mean(grad_residuals_norm**2))
        
        threshold_results[threshold_names[threshold_idx]][rbf_name] = {
            'eps': None,
            'cond': cond,
            'rmse': rmse_norm,
            'r2': r2_norm,
            'grad_rmse': grad_rmse_norm,
        }
        
        print(f"{rbf_name:<25} (fixed)         cond={cond:10.2e}  RMSE={rmse_norm:.6f}  R²={r2_norm:.6f}")

# Print summary table
print("\n" + "="*70)
print("SUMMARY: FIT QUALITY vs CONDITION NUMBER THRESHOLD")
print("="*70)

all_kernels = kernels_with_shape + kernels_no_shape

for kernel in all_kernels:
    print(f"\n{kernel.upper()}:")
    print(f"{'Threshold':<12} {'Shape Param':<15} {'Cond Number':<15} {'RMSE':<12} {'R²':<12}")
    print("-" * 65)
    for threshold_name in threshold_names:
        result = threshold_results[threshold_name][kernel]
        eps_str = f"{result['eps']:.6f}" if result['eps'] is not None else "N/A"
        cond_str = f"{result['cond']:.2e}"
        rmse_str = f"{result['rmse']:.6f}"
        r2_str = f"{result['r2']:.6f}"
        print(f"{threshold_name:<12} {eps_str:<15} {cond_str:<15} {rmse_str:<12} {r2_str:<12}")

# Extract best results at each threshold
print("\n" + "="*70)
print("BEST KERNEL AT EACH THRESHOLD")
print("="*70)

best_at_threshold = {}
for threshold_name in threshold_names:
    best_kernel = min(all_kernels, key=lambda k: threshold_results[threshold_name][k]['rmse'])
    best_rmse = threshold_results[threshold_name][best_kernel]['rmse']
    best_r2 = threshold_results[threshold_name][best_kernel]['r2']
    best_cond = threshold_results[threshold_name][best_kernel]['cond']
    best_at_threshold[threshold_name] = {
        'kernel': best_kernel,
        'rmse': best_rmse,
        'r2': best_r2,
        'cond': best_cond,
    }
    
    print(f"{threshold_name:<12}: {best_kernel:<25} RMSE={best_rmse:.6f}  R²={best_r2:.6f}  Actual cond={best_cond:.2e}")

# Create comparison plots
fig = plt.figure(figsize=(16, 10))

# Extract data for plotting
threshold_names_plot = threshold_names
x_pos = np.arange(len(threshold_names_plot))

# 1. RMSE progression for each kernel
ax1 = plt.subplot(2, 3, 1)
for kernel in all_kernels:
    rmse_vals = [threshold_results[th][kernel]['rmse'] for th in threshold_names_plot]
    ax1.plot(x_pos, rmse_vals, marker='o', linewidth=2, markersize=8, label=kernel)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(threshold_names_plot, rotation=45, ha='right')
ax1.set_ylabel('RMSE (normalized)', fontsize=11)
ax1.set_xlabel('Max Condition Number Threshold', fontsize=11)
ax1.set_title('Function RMSE vs Condition Number Threshold', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(alpha=0.3)

# 2. R² progression for each kernel
ax2 = plt.subplot(2, 3, 2)
for kernel in all_kernels:
    r2_vals = [threshold_results[th][kernel]['r2'] for th in threshold_names_plot]
    ax2.plot(x_pos, r2_vals, marker='s', linewidth=2, markersize=8, label=kernel)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(threshold_names_plot, rotation=45, ha='right')
ax2.set_ylabel('R² Score', fontsize=11)
ax2.set_xlabel('Max Condition Number Threshold', fontsize=11)
ax2.set_title('Variance Explained vs Condition Number Threshold', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9, loc='best')
ax2.grid(alpha=0.3)

# 3. Improvement from conservative to aggressive
ax3 = plt.subplot(2, 3, 3)
conservative_rmse = threshold_results['1e6']['cubic']['rmse']  # Conservative baseline
improvement_data = {}
for kernel in all_kernels:
    rmse_conservative = threshold_results['1e6'][kernel]['rmse']
    rmse_unlimited = threshold_results['Unlimited'][kernel]['rmse']
    improvement_pct = 100 * (rmse_conservative - rmse_unlimited) / rmse_conservative
    improvement_data[kernel] = improvement_pct

kernels_sorted = sorted(improvement_data.items(), key=lambda x: x[1], reverse=True)
kernel_names = [k[0] for k in kernels_sorted]
improvements = [k[1] for k in kernels_sorted]

colors = ['#1f77b4' if k in kernels_with_shape else '#ff7f0e' for k in kernel_names]
bars = ax3.barh(kernel_names, improvements, color=colors, alpha=0.7, edgecolor='black')
ax3.set_xlabel('RMSE Improvement (%)', fontsize=11)
ax3.set_title('Improvement from Conservative (1e6) to Unlimited', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='x')

# 4. Actual condition numbers achieved
ax4 = plt.subplot(2, 3, 4)
for kernel in all_kernels:
    cond_vals = [threshold_results[th][kernel]['cond'] for th in threshold_names_plot]
    ax4.semilogy(x_pos, cond_vals, marker='^', linewidth=2, markersize=8, label=kernel)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(threshold_names_plot, rotation=45, ha='right')
ax4.set_ylabel('Actual Condition Number (log scale)', fontsize=11)
ax4.set_xlabel('Max Condition Number Threshold', fontsize=11)
ax4.set_title('Actual Condition Numbers Achieved', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='best')
ax4.grid(alpha=0.3, which='both')

# 5. Best kernel at each threshold
ax5 = plt.subplot(2, 3, 5)
best_kernels = [best_at_threshold[th]['kernel'] for th in threshold_names_plot]
best_rmses = [best_at_threshold[th]['rmse'] for th in threshold_names_plot]
best_r2s = [best_at_threshold[th]['r2'] for th in threshold_names_plot]

ax5_twin = ax5.twinx()
line1 = ax5.plot(x_pos, best_rmses, marker='o', linewidth=3, markersize=10, color='#d62728', label='Best RMSE')
line2 = ax5_twin.plot(x_pos, best_r2s, marker='s', linewidth=3, markersize=10, color='#2ca02c', label='Best R²')

ax5.set_xticks(x_pos)
ax5.set_xticklabels(threshold_names_plot, rotation=45, ha='right')
ax5.set_ylabel('Best RMSE (normalized)', fontsize=11, color='#d62728')
ax5_twin.set_ylabel('Best R² Score', fontsize=11, color='#2ca02c')
ax5.set_xlabel('Max Condition Number Threshold', fontsize=11)
ax5.set_title('Best Performance at Each Threshold', fontsize=12, fontweight='bold')
ax5.tick_params(axis='y', labelcolor='#d62728')
ax5_twin.tick_params(axis='y', labelcolor='#2ca02c')
ax5.grid(alpha=0.3)

# Add legend
lines1, labels1 = ax5.get_legend_handles_labels()
lines2, labels2 = ax5_twin.get_legend_handles_labels()
ax5.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)

# 6. Summary table
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

summary_text = "RECOMMENDATIONS\n\n"
summary_text += f"Conservative (1e6):\n"
summary_text += f"  Best: {best_at_threshold['1e6']['kernel']}\n"
summary_text += f"  RMSE: {best_at_threshold['1e6']['rmse']:.6f}\n"
summary_text += f"  R²: {best_at_threshold['1e6']['r2']:.6f}\n\n"

summary_text += f"Recommended (1e14):\n"
summary_text += f"  Best: {best_at_threshold['1e14']['kernel']}\n"
summary_text += f"  RMSE: {best_at_threshold['1e14']['rmse']:.6f}\n"
summary_text += f"  R²: {best_at_threshold['1e14']['r2']:.6f}\n\n"

summary_text += f"Aggressive (Unlimited):\n"
summary_text += f"  Best: {best_at_threshold['Unlimited']['kernel']}\n"
summary_text += f"  RMSE: {best_at_threshold['Unlimited']['rmse']:.6f}\n"
summary_text += f"  R²: {best_at_threshold['Unlimited']['r2']:.6f}\n\n"

best_unlimited = best_at_threshold['Unlimited']
best_conservative = best_at_threshold['1e6']
improvement = 100 * (best_conservative['rmse'] - best_unlimited['rmse']) / best_conservative['rmse']
summary_text += f"Potential Improvement:\n"
summary_text += f"  +{improvement:.1f}% better RMSE\n"
summary_text += f"  by relaxing condition#\n"

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('rbf_sweep_condition_number_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Condition number analysis plot saved to: rbf_sweep_condition_number_analysis.png")
plt.close()

print("\n✓ Analysis complete!")
