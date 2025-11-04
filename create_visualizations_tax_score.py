#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create complete visualizations for Tax Score model
SHAP, Robustness, and Residual Analysis
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import cross_val_score, learning_curve, KFold
from sklearn.metrics import r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING VISUALIZATIONS FOR TAX SCORE MODEL")
print("="*80)

# Load results
with open('resultados/modelos/final_dual_improved.pkl', 'rb') as f:
    results = pickle.load(f)

tax_res = results['Tax_Score']
best_name, best_res = tax_res['best']
model = best_res['model']
y_test = tax_res['y_test']
X_test = tax_res['X_test']
y_pred = best_res['y_pred']
predictors = tax_res['predictors']

print(f"\nModel: {best_name}")
print(f"R²: {best_res['r2']:.4f}")
print(f"CV: {best_res['cv']:.4f}")
print(f"Balanced Score: {best_res['balanced_score']:.4f}")

# ========== 1. SHAP ANALYSIS ==========
print("\n1. Creating SHAP analysis...")

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Create SHAP plots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('SHAP Analysis - Tax Score Prediction from Gender Indicators',
             fontsize=14, fontweight='bold')

# Feature importance
ax = axes[0]
shap_importance = np.abs(shap_values).mean(axis=0)
indices = np.argsort(shap_importance)[::-1][:15]

# Truncate names
truncated_names = [predictors[i][:42] + '...' if len(predictors[i]) > 42
                   else predictors[i] for i in indices]

y_pos = np.arange(len(indices))
ax.barh(y_pos, shap_importance[indices], color='steelblue', edgecolor='black', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(truncated_names, fontsize=8)
ax.set_xlabel('Mean |SHAP value|', fontsize=10, fontweight='bold')
ax.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# SHAP beeswarm plot (manual)
ax = axes[1]
for i, idx in enumerate(indices[:10]):
    x_vals = shap_values[:, idx]
    y_vals = np.full_like(x_vals, i) + np.random.normal(0, 0.1, size=len(x_vals))
    colors = X_test[:, idx]
    scatter = ax.scatter(x_vals, y_vals, c=colors, cmap='RdBu_r', s=50, alpha=0.6, edgecolors='k', linewidth=0.5)

truncated_top10 = [predictors[i][:42] + '...' if len(predictors[i]) > 42
                   else predictors[i] for i in indices[:10]]
ax.set_yticks(range(10))
ax.set_yticklabels(truncated_top10, fontsize=8)
ax.set_xlabel('SHAP value (impact on model output)', fontsize=10, fontweight='bold')
ax.set_title('SHAP Value Distribution (Top 10)', fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Feature Value', fontsize=9)

plt.tight_layout()
plt.savefig('resultados/graficos_finales/shap_Tax_Score_new.png', dpi=200, bbox_inches='tight')
plt.close()

print("  SHAP plots saved")

# ========== 2. ROBUSTNESS ANALYSIS ==========
print("\n2. Creating robustness analysis...")

residuals = y_test - y_pred

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f'Robustness Analysis - Tax Score ({best_name})',
             fontsize=14, fontweight='bold')

# Q-Q Plot
ax = axes[0, 0]
stats.probplot(residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot', fontsize=11, fontweight='bold')
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# Residual Distribution
ax = axes[0, 1]
ax.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residual Distribution', fontsize=11, fontweight='bold')
ax.set_xlabel('Residuals', fontsize=9)
ax.set_ylabel('Frequency', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# Predicted vs Actual
ax = axes[0, 2]
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=60, color='steelblue')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect fit')
ax.set_title('Predicted vs Actual', fontsize=11, fontweight='bold')
ax.set_xlabel('Actual Values', fontsize=9)
ax.set_ylabel('Predicted Values', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Residuals vs Predicted
ax = axes[1, 0]
ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=60, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residuals vs Predicted', fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted Values', fontsize=9)
ax.set_ylabel('Residuals', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# Cross-Validation Stability
ax = axes[1, 1]
try:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_test, y_test, cv=kf, scoring='r2')
    ax.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {cv_scores.mean():.3f}')
    ax.set_title('5-Fold Cross-Validation', fontsize=11, fontweight='bold')
    ax.set_xlabel('Fold', fontsize=9)
    ax.set_ylabel('R² Score', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])
except:
    ax.text(0.5, 0.5, 'CV not available', ha='center', va='center', fontsize=9)

# Learning Curve
ax = axes[1, 2]
try:
    train_sizes = np.linspace(0.3, 1.0, 5)
    train_sizes_abs, train_scores, test_scores = learning_curve(
        model, X_test, y_test, train_sizes=train_sizes,
        cv=3, scoring='r2', random_state=42, n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='blue')
    ax.plot(train_sizes_abs, test_mean, 'o-', color='red', label='Validation')
    ax.fill_between(train_sizes_abs, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color='red')
    ax.set_title('Learning Curve', fontsize=11, fontweight='bold')
    ax.set_xlabel('Training Examples', fontsize=9)
    ax.set_ylabel('R² Score', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
except:
    ax.text(0.5, 0.5, 'Learning curve not available', ha='center', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('resultados/graficos_finales/robustness_Tax_Score_new.png', dpi=200, bbox_inches='tight')
plt.close()

print("  Robustness plots saved")

# ========== 3. DETAILED RESIDUAL ANALYSIS ==========
print("\n3. Creating detailed residual analysis...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Detailed Residual Analysis - Tax Score',
             fontsize=14, fontweight='bold')

# Residual Sequence
ax = axes[0, 0]
ax.plot(range(len(residuals)), residuals, 'o-', markersize=5, alpha=0.7, color='steelblue')
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.fill_between(range(len(residuals)), -2*residuals.std(), 2*residuals.std(),
                 alpha=0.2, color='gray', label='±2 SD')
ax.set_title('Residual Sequence', fontsize=11, fontweight='bold')
ax.set_xlabel('Observation Index', fontsize=9)
ax.set_ylabel('Residuals', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Scale-Location
ax = axes[0, 1]
sqrt_abs_resid = np.sqrt(np.abs(residuals))
ax.scatter(y_pred, sqrt_abs_resid, alpha=0.6, edgecolors='k', s=60, color='steelblue')
ax.set_title('Scale-Location Plot', fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted Values', fontsize=9)
ax.set_ylabel('√|Residuals|', fontsize=9)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# Bootstrap R²
ax = axes[1, 0]
bootstrap_scores = []
rng = np.random.RandomState(42)
for _ in range(100):
    indices = rng.randint(0, len(y_test), len(y_test))
    r2_boot = r2_score(y_test[indices], y_pred[indices])
    bootstrap_scores.append(r2_boot)
ax.hist(bootstrap_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(np.mean(bootstrap_scores), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(bootstrap_scores):.3f}')
ci_low, ci_high = np.percentile(bootstrap_scores, [2.5, 97.5])
ax.axvline(ci_low, color='orange', linestyle=':', linewidth=2, label='95% CI')
ax.axvline(ci_high, color='orange', linestyle=':', linewidth=2)
ax.set_title('Bootstrap R² Distribution', fontsize=11, fontweight='bold')
ax.set_xlabel('R² Score', fontsize=9)
ax.set_ylabel('Frequency', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Standardized Residuals
ax = axes[1, 1]
std_resid = residuals / residuals.std()
ax.scatter(range(len(std_resid)), std_resid, alpha=0.6, edgecolors='k', s=60, color='steelblue')
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.axhline(2, color='red', linestyle='--', linewidth=2, alpha=0.5, label='±2 SD')
ax.axhline(-2, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax.set_title('Standardized Residuals', fontsize=11, fontweight='bold')
ax.set_xlabel('Observation Index', fontsize=9)
ax.set_ylabel('Standardized Residuals', fontsize=9)
ax.tick_params(labelsize=8)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados/graficos_finales/residual_details_Tax_Score_new.png', dpi=200, bbox_inches='tight')
plt.close()

print("  Residual detail plots saved")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nFiles created:")
print("  - resultados/graficos_finales/shap_Tax_Score_new.png")
print("  - resultados/graficos_finales/robustness_Tax_Score_new.png")
print("  - resultados/graficos_finales/residual_details_Tax_Score_new.png")
