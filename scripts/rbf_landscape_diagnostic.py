"""
Diagnostic: Explore the full RMSE vs Epsilon landscape
Shows how fit quality and condition number vary across the complete range.
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
print("FULL LANDSCAPE EXPLORATION")
print("="*70)

# Test a very wide range of epsilon values
eps_values = np.logspace(-4, 4, 200)  # 0.0001 to 10000

results = {
    'gaussian': {'eps': [], 'cond': [], 'rmse': [], 'r2': [], 'fit_failed': []},
    'multiquadric': {'eps': [], 'cond': [], 'rmse': [], 'r2': [], 'fit_failed': []},
    'inverse_multiquadric': {'eps': [], 'cond': [], 'rmse': [], 'r2': [], 'fit_failed': []},
}

for rbf_name in results.keys():
    print(f"\nTesting {rbf_name}...")
    
    for i, eps in enumerate(eps_values):
        if i % 20 == 0:
            print(f"  {i}/{len(eps_values)}", end='\r', flush=True)
        
        fitter = RBFPolynomialFitter(
            rbf_name=rbf_name,
            rbf_shape_parameter=eps,
            polynomial_degree=2,
            regularization_lambda=1e-8
        )
        
        # Estimate condition number
        try:
            cond = estimate_condition_number(X_centers, fitter.rbf_kernel)
            results[rbf_name]['cond'].append(cond)
        except:
            results[rbf_name]['cond'].append(np.nan)
            results[rbf_name]['rmse'].append(np.nan)
            results[rbf_name]['r2'].append(np.nan)
            results[rbf_name]['eps'].append(eps)
            results[rbf_name]['fit_failed'].append(True)
            continue
        
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
            
            results[rbf_name]['rmse'].append(rmse_norm)
            results[rbf_name]['r2'].append(r2_norm)
            results[rbf_name]['fit_failed'].append(False)
        except Exception as e:
            results[rbf_name]['rmse'].append(np.nan)
            results[rbf_name]['r2'].append(np.nan)
            results[rbf_name]['fit_failed'].append(True)
        
        results[rbf_name]['eps'].append(eps)
    
    print(f"  {len(eps_values)}/{len(eps_values)} complete")

# Find optima
print("\n" + "="*70)
print("OPTIMA FOUND")
print("="*70)

for rbf_name in results.keys():
    valid_rmse = np.array(results[rbf_name]['rmse'])
    valid_rmse = valid_rmse[~np.isnan(valid_rmse)]
    
    if len(valid_rmse) > 0:
        best_idx = np.nanargmin(results[rbf_name]['rmse'])
        best_eps = results[rbf_name]['eps'][best_idx]
        best_rmse = results[rbf_name]['rmse'][best_idx]
        best_r2 = results[rbf_name]['r2'][best_idx]
        best_cond = results[rbf_name]['cond'][best_idx]
        
        print(f"\n{rbf_name.upper()}:")
        print(f"  Best ε: {best_eps:.6f}")
        print(f"  Best RMSE: {best_rmse:.6f}")
        print(f"  Best R²: {best_r2:.6f}")
        print(f"  Condition Number at Best: {best_cond:.2e}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

rbf_names_list = ['gaussian', 'multiquadric', 'inverse_multiquadric']

for idx, rbf_name in enumerate(rbf_names_list):
    # Filter out failed fits
    eps_arr = np.array(results[rbf_name]['eps'])
    rmse_arr = np.array(results[rbf_name]['rmse'])
    cond_arr = np.array(results[rbf_name]['cond'])
    
    valid_idx = ~np.isnan(rmse_arr)
    eps_valid = eps_arr[valid_idx]
    rmse_valid = rmse_arr[valid_idx]
    cond_valid = cond_arr[valid_idx]
    
    # RMSE vs Epsilon
    ax_rmse = axes[0, idx]
    ax_rmse.semilogx(eps_valid, rmse_valid, 'o-', linewidth=2, markersize=4, color='#1f77b4')
    best_idx = np.nanargmin(rmse_valid)
    ax_rmse.plot(eps_valid[best_idx], rmse_valid[best_idx], 'r*', markersize=20, 
                 label=f'Best: ε={eps_valid[best_idx]:.6f}, RMSE={rmse_valid[best_idx]:.6f}')
    ax_rmse.set_xlabel('Shape Parameter ε (log scale)', fontsize=11)
    ax_rmse.set_ylabel('RMSE (normalized)', fontsize=11)
    ax_rmse.set_title(f'{rbf_name.upper()}\nRMSE vs Shape Parameter', fontsize=12, fontweight='bold')
    ax_rmse.grid(alpha=0.3, which='both')
    ax_rmse.legend(fontsize=9)
    
    # Condition Number vs Epsilon
    ax_cond = axes[1, idx]
    ax_cond.loglog(eps_valid, cond_valid, 'o-', linewidth=2, markersize=4, color='#d62728')
    ax_cond.axhline(y=1e6, color='g', linestyle='--', linewidth=2, label='1e6 threshold')
    ax_cond.axhline(y=1e8, color='orange', linestyle='--', linewidth=2, label='1e8 threshold')
    ax_cond.axhline(y=1e10, color='purple', linestyle='--', linewidth=2, label='1e10 threshold')
    ax_cond.set_xlabel('Shape Parameter ε (log scale)', fontsize=11)
    ax_cond.set_ylabel('Condition Number (log scale)', fontsize=11)
    ax_cond.set_title(f'{rbf_name.upper()}\nCondition Number vs Shape Parameter', fontsize=12, fontweight='bold')
    ax_cond.grid(alpha=0.3, which='both')
    ax_cond.legend(fontsize=8)

plt.tight_layout()
plt.savefig('rbf_landscape_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Landscape analysis plot saved to: rbf_landscape_analysis.png")
plt.close()

# Print detailed summary
print("\n" + "="*70)
print("DETAILED ANALYSIS")
print("="*70)

for rbf_name in rbf_names_list:
    eps_arr = np.array(results[rbf_name]['eps'])
    rmse_arr = np.array(results[rbf_name]['rmse'])
    cond_arr = np.array(results[rbf_name]['cond'])
    
    valid_idx = ~np.isnan(rmse_arr)
    eps_valid = eps_arr[valid_idx]
    rmse_valid = rmse_arr[valid_idx]
    cond_valid = cond_arr[valid_idx]
    
    print(f"\n{rbf_name.upper()}:")
    print(f"  Number of tested epsilon values: {len(eps_valid)}")
    print(f"  Epsilon range: [{eps_valid.min():.6f}, {eps_valid.max():.6f}]")
    print(f"  RMSE range: [{rmse_valid.min():.6f}, {rmse_valid.max():.6f}]")
    print(f"  RMSE spread: {rmse_valid.max() - rmse_valid.min():.6f}")
    print(f"  Condition number range: [{cond_valid.min():.2e}, {cond_valid.max():.2e}]")
    
    best_rmse_idx = np.argmin(rmse_valid)
    print(f"\n  Best fit at ε = {eps_valid[best_rmse_idx]:.6f}:")
    print(f"    RMSE: {rmse_valid[best_rmse_idx]:.6f}")
    print(f"    Condition Number: {cond_valid[best_rmse_idx]:.2e}")
    
    # Check if optimal is at boundary
    if best_rmse_idx == 0:
        print(f"    ⚠️  Optimum at LOWER boundary - may need smaller epsilon")
    elif best_rmse_idx == len(eps_valid) - 1:
        print(f"    ⚠️  Optimum at UPPER boundary - may need larger epsilon")
    else:
        print(f"    ✓ Optimum in interior - good range coverage")

print("\n✓ Analysis complete!")
