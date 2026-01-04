"""
Find minimum model complexity for decent convergence.
Test polynomial degrees 0-5 with varying LHS centers 1-25.
"""

import numpy as np
from sys import path
path.insert(0, '..')
from src.rbf_polynomial_fitter import RBFPolynomialFitter
import matplotlib.pyplot as plt
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

# Grid search
poly_degrees = [1, 2, 3, 4, 5]  # Skip degree 0
n_centers_list = [1, 2, 3, 5, 8, 10, 15, 20]

results = {}

print("="*80)
print("COMPLEXITY SWEEP: POLYNOMIAL DEGREE × LHS CENTERS")
print("="*80)
print(f"\nTesting {len(poly_degrees)} polynomial degrees × {len(n_centers_list)} center counts")
print(f"Training samples: {X_norm.shape[0]}\n")

for poly_degree in poly_degrees:
    for n_centers in n_centers_list:
        key = (poly_degree, n_centers)
        
        try:
            fitter = RBFPolynomialFitter(
                rbf_name='gaussian',
                rbf_shape_parameter=0.115,
                polynomial_degree=poly_degree,
                regularization_lambda=1e-8
            )
            
            fitter.fit(
                X_norm, f_norm,
                df=grads_norm,
                n_centers=n_centers,
                use_lhs_centers=True,
                lhs_bounds=(lhs_lower, lhs_upper),
                random_state=42
            )
            
            # Predictions
            f_pred_norm = fitter.predict(X_norm)
            grads_pred_norm = fitter.predict_gradient(X_norm)
            
            # Metrics
            residuals = f_norm - f_pred_norm
            rmse = np.sqrt(np.mean(residuals**2))
            r2 = 1 - np.sum(residuals**2) / np.sum((f_norm - f_norm.mean())**2)
            grad_residuals = grads_norm - grads_pred_norm
            grad_rmse = np.sqrt(np.mean(grad_residuals**2))
            
            n_bases = fitter.n_rbf_bases + fitter._count_polynomial_bases(X_norm.shape[1])
            
            results[key] = {
                'rmse': rmse,
                'r2': r2,
                'grad_rmse': grad_rmse,
                'n_bases': n_bases,
            }
            
            print(f"  Deg {poly_degree}, Centers {n_centers:2d}: RMSE={rmse:.6f}, R²={r2:.4f}, Bases={n_bases:3d}")
            
        except Exception as e:
            print(f"  Deg {poly_degree}, Centers {n_centers:2d}: FAILED - {str(e)[:60]}")
            results[key] = None

print("\n" + "="*80)
print("PARETO FRONTIER: Best Performance per Basis Count")
print("="*80)

# Find Pareto frontier (best RMSE for each basis count)
rmse_by_bases = {}
for (deg, centers), res in results.items():
    if res is None:
        continue
    n_bases = res['n_bases']
    rmse = res['rmse']
    
    if n_bases not in rmse_by_bases or rmse < rmse_by_bases[n_bases]['rmse']:
        rmse_by_bases[n_bases] = {
            'rmse': rmse,
            'r2': res['r2'],
            'grad_rmse': res['grad_rmse'],
            'config': (deg, centers),
        }

# Sort by bases
sorted_frontier = sorted(rmse_by_bases.items())
print("\nBases  | Config (Deg, Centers) | RMSE      | R²       | Grad RMSE")
print("-------|----------------------|-----------|----------|----------")
for n_bases, data in sorted_frontier:
    deg, centers = data['config']
    rmse = data['rmse']
    r2 = data['r2']
    grad_rmse = data['grad_rmse']
    print(f"{n_bases:5d}  | Deg {deg}, Centers {centers:2d}        | {rmse:.6f} | {r2:.6f} | {grad_rmse:.6f}")

print("\n" + "="*80)
print("CONVERGENCE ANALYSIS")
print("="*80)

# Find minimum complexity for different RMSE targets
rmse_targets = [0.50, 0.55, 0.60, 0.65, 0.70]
print("\nMinimum bases needed for target RMSE:")
print("Target RMSE | Min Bases | Config          | Actual RMSE | R²      ")
print("------------|-----------|-----------------|-------------|--------")

for target in rmse_targets:
    candidates = [(n_bases, data) for n_bases, data in sorted_frontier 
                  if data['rmse'] <= target]
    if candidates:
        n_bases, data = candidates[0]
        deg, centers = data['config']
        print(f"{target:.2f}       | {n_bases:9d} | Deg {deg}, Cent {centers:2d}  | {data['rmse']:.6f}  | {data['r2']:.4f}")
    else:
        print(f"{target:.2f}       | Not achieved")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

best_config = sorted_frontier[0][1]
print(f"\nSimplest model: {best_config['config'][0]} deg, {best_config['config'][1]} centers")
print(f"  Bases: {sorted_frontier[0][0]}")
print(f"  RMSE: {best_config['rmse']:.6f}")
print(f"  R²: {best_config['r2']:.6f}")

# Find elbow point (diminishing returns)
rmses = [data['rmse'] for _, data in sorted_frontier]
improvements = [rmses[i-1] - rmses[i] for i in range(1, len(rmses))]
max_improvement_idx = np.argmax(improvements)
elbow_idx = max_improvement_idx + 1
if elbow_idx < len(sorted_frontier):
    elbow_bases, elbow_data = sorted_frontier[elbow_idx]
    print(f"\nElbow point (diminishing returns): {elbow_bases} bases")
    print(f"  Config: Deg {elbow_data['config'][0]}, {elbow_data['config'][1]} centers")
    print(f"  RMSE: {elbow_data['rmse']:.6f}")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Complexity vs Performance', fontsize=14, fontweight='bold')

# Plot 1: RMSE vs Bases (Pareto frontier)
ax = axes[0, 0]
bases_list = [n for n, _ in sorted_frontier]
rmse_list = [data['rmse'] for _, data in sorted_frontier]
r2_list = [data['r2'] for _, data in sorted_frontier]
configs = [f"D{data['config'][0]}C{data['config'][1]}" for _, data in sorted_frontier]

ax.plot(bases_list, rmse_list, 'o-', linewidth=2, markersize=8, label='RMSE')
for i, (bases, rmse, config) in enumerate(zip(bases_list, rmse_list, configs)):
    if i % 2 == 0:  # Label every other point
        ax.annotate(config, (bases, rmse), textcoords="offset points", 
                   xytext=(0,10), ha='center', fontsize=8)
ax.set_xlabel('Total Basis Functions', fontsize=11)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('Pareto Frontier: RMSE vs Complexity', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 2: R² vs Bases
ax = axes[0, 1]
ax.plot(bases_list, r2_list, 's-', linewidth=2, markersize=8, color='green', label='R²')
ax.set_xlabel('Total Basis Functions', fontsize=11)
ax.set_ylabel('R² Score', fontsize=11)
ax.set_title('Variance Explained vs Complexity', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
ax.set_ylim([0, 1])

# Plot 3: All results heatmap (RMSE)
ax = axes[1, 0]
rmse_matrix = np.full((len(poly_degrees), len(n_centers_list)), np.nan)
for (deg, centers), res in results.items():
    if res is not None:
        i = poly_degrees.index(deg)
        j = n_centers_list.index(centers)
        rmse_matrix[i, j] = res['rmse']

im = ax.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(n_centers_list)))
ax.set_xticklabels(n_centers_list)
ax.set_yticks(range(len(poly_degrees)))
ax.set_yticklabels(poly_degrees)
ax.set_xlabel('LHS Centers', fontsize=11)
ax.set_ylabel('Polynomial Degree', fontsize=11)
ax.set_title('RMSE Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='RMSE')

# Add text annotations
for i in range(len(poly_degrees)):
    for j in range(len(n_centers_list)):
        if not np.isnan(rmse_matrix[i, j]):
            text = ax.text(j, i, f'{rmse_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=7)

# Plot 4: Improvement per additional basis
ax = axes[1, 1]
improvement_per_basis = []
x_vals = []
for i in range(1, len(sorted_frontier)):
    prev_bases, prev_data = sorted_frontier[i-1]
    curr_bases, curr_data = sorted_frontier[i]
    improvement = prev_data['rmse'] - curr_data['rmse']
    bases_added = curr_bases - prev_bases
    improvement_per_basis_unit = improvement / bases_added if bases_added > 0 else 0
    improvement_per_basis.append(improvement_per_basis_unit)
    x_vals.append(curr_bases)

ax.plot(x_vals, improvement_per_basis, 'o-', linewidth=2, markersize=8, color='purple')
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel('Total Basis Functions', fontsize=11)
ax.set_ylabel('RMSE Improvement per Added Basis', fontsize=11)
ax.set_title('Marginal Returns', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('complexity_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n[OK] Complexity analysis plot saved to: complexity_analysis.png")
plt.close()
