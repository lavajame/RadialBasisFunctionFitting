"""
RBF Kernel Sweep with Extended Shape Parameter Range

Tests shape parameters over a much wider range to find the optimal
epsilon values that can achieve better fits at higher condition numbers.
"""

import numpy as np
import matplotlib.pyplot as plt
from rbf_polynomial_fitter import RBFPolynomialFitter
import warnings
warnings.filterwarnings('ignore')

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

print("="*70)
print("EXTENDED SHAPE PARAMETER SWEEP")
print("="*70)

# Test extended range of shape parameters
condition_thresholds = [1e6, 1e8, 1e10, 1e12, 1e14, 1e16]
threshold_names = ['1e6', '1e8', '1e10', '1e12', '1e14', '1e16']

kernels_with_shape = ['gaussian', 'multiquadric', 'inverse_multiquadric']

results_extended = {}

for rbf_name in kernels_with_shape:
    print(f"\n{rbf_name.upper()}")
    print("-" * 70)
    results_extended[rbf_name] = {}
    
    # Test many shape parameter values
    # Extended range from 0.001 to 1000
    eps_values = np.logspace(-3, 3, 100)  # 0.001 to 1000
    
    best_fits = {th: {'eps': None, 'cond': np.inf, 'rmse': np.inf, 'r2': -np.inf} for th in threshold_names}
    
    for eps in eps_values:
        fitter = RBFPolynomialFitter(
            rbf_name=rbf_name,
            rbf_shape_parameter=eps,
            polynomial_degree=2,
            regularization_lambda=1e-8
        )
        
        # Get condition number
        cond = estimate_condition_number(X_centers, fitter.rbf_kernel)
        
        # Fit model
        try:
            fitter.fit(
                X_norm, f_norm,
                df=grads_norm,
                n_centers=10,
                use_lhs_centers=True,
                lhs_bounds=(lhs_lower, lhs_upper),
                random_state=42
            )
            
            f_pred_norm = fitter.predict(X_norm)
            residuals_norm = f_norm - f_pred_norm
            rmse_norm = np.sqrt(np.mean(residuals_norm**2))
            r2_norm = 1 - np.sum(residuals_norm**2) / np.sum((f_norm - f_norm.mean())**2)
        except:
            continue
        
        # Update best fits for each threshold
        for threshold, threshold_name in zip(condition_thresholds, threshold_names):
            if cond <= threshold and rmse_norm < best_fits[threshold_name]['rmse']:
                best_fits[threshold_name] = {
                    'eps': eps,
                    'cond': cond,
                    'rmse': rmse_norm,
                    'r2': r2_norm,
                }
    
    # Print results for this kernel
    print(f"{'Threshold':<12} {'Eps':<12} {'Cond Number':<15} {'RMSE':<12} {'R²':<12}")
    print("-" * 65)
    for threshold_name in threshold_names:
        result = best_fits[threshold_name]
        eps_str = f"{result['eps']:.6f}" if result['eps'] is not None else "N/A"
        cond_str = f"{result['cond']:.2e}"
        rmse_str = f"{result['rmse']:.6f}"
        r2_str = f"{result['r2']:.6f}"
        print(f"{threshold_name:<12} {eps_str:<12} {cond_str:<15} {rmse_str:<12} {r2_str:<12}")
    
    results_extended[rbf_name] = best_fits

# Summary
print("\n" + "="*70)
print("SUMMARY: BEST RMSE AT EACH CONDITION NUMBER THRESHOLD")
print("="*70)

for threshold_name in threshold_names:
    print(f"\n{threshold_name}:")
    best_kernel = min(kernels_with_shape, 
                     key=lambda k: results_extended[k][threshold_name]['rmse'])
    best_result = results_extended[best_kernel][threshold_name]
    
    print(f"  Best Kernel: {best_kernel}")
    print(f"  Shape Parameter (ε): {best_result['eps']:.6f}")
    print(f"  Condition Number: {best_result['cond']:.2e}")
    print(f"  RMSE: {best_result['rmse']:.6f}")
    print(f"  R²: {best_result['r2']:.6f}")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for idx, rbf_name in enumerate(kernels_with_shape):
    ax = axes[idx]
    
    rmse_vals = [results_extended[rbf_name][th]['rmse'] for th in threshold_names]
    eps_vals = [results_extended[rbf_name][th]['eps'] for th in threshold_names]
    
    ax2 = ax.twinx()
    
    line1 = ax.plot(threshold_names, rmse_vals, marker='o', linewidth=3, markersize=10, 
                    color='#d62728', label='RMSE')
    line2 = ax2.semilogy(threshold_names, eps_vals, marker='s', linewidth=3, markersize=10, 
                         color='#2ca02c', label='Optimal ε')
    
    ax.set_ylabel('RMSE (normalized)', fontsize=12, color='#d62728')
    ax2.set_ylabel('Optimal Shape Parameter ε (log scale)', fontsize=12, color='#2ca02c')
    ax.set_xlabel('Max Condition Number Threshold', fontsize=12)
    ax.set_title(f'{rbf_name.upper()}\nExtended Shape Parameter Range', fontsize=13, fontweight='bold')
    
    ax.tick_params(axis='y', labelcolor='#d62728')
    ax2.tick_params(axis='y', labelcolor='#2ca02c')
    
    ax.grid(alpha=0.3)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='best', fontsize=10)

plt.tight_layout()
plt.savefig('rbf_sweep_extended_range.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Extended range analysis plot saved to: rbf_sweep_extended_range.png")
plt.close()

print("\n✓ Extended sweep analysis complete!")
