#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Robustness Analysis Plots for Best Model
- Residual plots (Q-Q, histogram, scatter)
- Cross-validation stability
- Learning curves
- Prediction intervals
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import learning_curve, cross_val_score
import seaborn as sns

# Set academic plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8

# Load data and results
df = pd.read_excel('DATA_GHAB2.xlsx')
with open('resultados/modelos/all_results.pkl', 'rb') as f:
    results = pickle.load(f)

target = 'G_GPD_PCAP_SLOPE'
res = results[target]
best_name, best_res = res['best']
predictors = res['predictors']

# Prepare data
mask = df[target].notna()
for var in predictors:
    mask &= df[var].notna()

X = df.loc[mask, predictors].values
y = df.loc[mask, target].values

# Get train/test split (same as in analysis)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Get best model
best_model = best_res['model']
y_pred_test = best_res['y_pred']
y_pred_train = best_model.predict(X_train)

residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

print(f"Creating robustness plots for {best_name}...")

# Create figure with 6 subplots (2x3)
fig = plt.figure(figsize=(10, 6.5))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3, left=0.08, right=0.96, top=0.94, bottom=0.08)

# 1. Q-Q Plot
ax1 = fig.add_subplot(gs[0, 0])
stats.probplot(residuals_test, dist="norm", plot=ax1)
ax1.set_title('Q-Q Plot (Test Set)', fontweight='bold')
ax1.set_xlabel('Theoretical Quantiles')
ax1.set_ylabel('Sample Quantiles')
ax1.grid(True, alpha=0.3)

# 2. Residual Histogram
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(residuals_test, bins=10, edgecolor='black', alpha=0.7, color='#4682B4')
ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Zero')
ax2.set_title('Residual Distribution', fontweight='bold')
ax2.set_xlabel('Residuals')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Predicted vs Actual
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_test, y_pred_test, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
ax3.set_title('Predicted vs Actual', fontweight='bold')
ax3.set_xlabel('Actual GDP Growth')
ax3.set_ylabel('Predicted GDP Growth')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Residuals vs Predicted
ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_pred_test, residuals_test, alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
ax4.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax4.set_title('Residuals vs Predicted', fontweight='bold')
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.grid(True, alpha=0.3)

# 5. Cross-Validation Stability
ax5 = fig.add_subplot(gs[1, 1])
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
ax5.bar(range(1, 6), cv_scores, edgecolor='black', alpha=0.7, color='#4682B4')
ax5.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=1.5,
            label=f'Mean: {cv_scores.mean():.3f}')
ax5.set_title('5-Fold CV Stability', fontweight='bold')
ax5.set_xlabel('Fold')
ax5.set_ylabel('R² Score')
ax5.set_ylim([min(cv_scores.min() - 0.1, 0), cv_scores.max() + 0.1])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# 6. Learning Curve
ax6 = fig.add_subplot(gs[1, 2])
train_sizes = np.linspace(0.3, 1.0, 8)
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_model, X_train, y_train, train_sizes=train_sizes, cv=3,
    scoring='r2', n_jobs=-1, random_state=42
)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

ax6.plot(train_sizes_abs, train_mean, 'o-', color='#1f77b4', label='Training', linewidth=2)
ax6.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                 alpha=0.2, color='#1f77b4')
ax6.plot(train_sizes_abs, val_mean, 'o-', color='#ff7f0e', label='Validation', linewidth=2)
ax6.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                 alpha=0.2, color='#ff7f0e')
ax6.set_title('Learning Curve', fontweight='bold')
ax6.set_xlabel('Training Size')
ax6.set_ylabel('R² Score')
ax6.legend(loc='lower right')
ax6.grid(True, alpha=0.3)

# Add main title
fig.suptitle(f'Robustness Analysis: {best_name} (R²={best_res["r2"]:.3f})',
             fontsize=12, fontweight='bold', y=0.98)

# Save
plt.savefig('resultados/graficos_finales/robustness_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: resultados/graficos_finales/robustness_analysis.png")

# Create additional detailed residual analysis
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
fig2.subplots_adjust(hspace=0.3, wspace=0.3)

# Residuals over index
ax1.scatter(range(len(residuals_test)), residuals_test, alpha=0.7, s=40, edgecolors='black', linewidth=0.5)
ax1.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax1.axhline(residuals_test.std(), color='orange', linestyle=':', linewidth=1.5, label='+1 SD')
ax1.axhline(-residuals_test.std(), color='orange', linestyle=':', linewidth=1.5, label='-1 SD')
ax1.set_title('Residual Sequence Plot', fontweight='bold')
ax1.set_xlabel('Observation Index')
ax1.set_ylabel('Residuals')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Scale-Location plot (sqrt of standardized residuals)
standardized_residuals = residuals_test / residuals_test.std()
ax2.scatter(y_pred_test, np.sqrt(np.abs(standardized_residuals)), alpha=0.7, s=40,
           edgecolors='black', linewidth=0.5)
ax2.set_title('Scale-Location Plot', fontweight='bold')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('√|Standardized Residuals|')
ax2.grid(True, alpha=0.3)

# Bootstrap R² Distribution
ax3.hist(best_res.get('bootstrap_scores', [best_res['r2']]), bins=20, edgecolor='black',
        alpha=0.7, color='#4682B4')
ax3.axvline(best_res['r2'], color='red', linestyle='--', linewidth=2, label=f'Test R²: {best_res["r2"]:.3f}')
if 'bootstrap_ci' in res['validation']:
    ci_low, ci_high = res['validation']['bootstrap_ci']
    ax3.axvline(ci_low, color='orange', linestyle=':', linewidth=1.5, label=f'95% CI')
    ax3.axvline(ci_high, color='orange', linestyle=':', linewidth=1.5)
ax3.set_title('Bootstrap R² Distribution', fontweight='bold')
ax3.set_xlabel('R² Score')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Leverage plot (Cook's distance would need statsmodels)
residuals_squared = residuals_test ** 2
ax4.scatter(range(len(residuals_squared)), residuals_squared, alpha=0.7, s=40,
           edgecolors='black', linewidth=0.5)
threshold = 3 * residuals_test.var()
ax4.axhline(threshold, color='red', linestyle='--', linewidth=1.5, label=f'3σ² Threshold')
ax4.set_title('Influential Points (Squared Residuals)', fontweight='bold')
ax4.set_xlabel('Observation Index')
ax4.set_ylabel('Squared Residuals')
ax4.legend()
ax4.grid(True, alpha=0.3)

fig2.suptitle('Detailed Residual Analysis', fontsize=12, fontweight='bold')
plt.savefig('resultados/graficos_finales/residual_details.png', dpi=300, bbox_inches='tight')
print("Saved: resultados/graficos_finales/residual_details.png")

print("\nRobustness Analysis Complete:")
print(f"  - Q-Q plot for normality check")
print(f"  - Residual distribution and patterns")
print(f"  - CV stability across folds")
print(f"  - Learning curve analysis")
print(f"  - Bootstrap distribution")
print(f"  - Influential points detection")
