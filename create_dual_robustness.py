#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Robustness Analysis for Both Targets:
- GDP Growth
- Ease of Doing Business
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DUAL TARGET ROBUSTNESS ANALYSIS")
print("=" * 80)

# Load results
with open('resultados/modelos/dual_target_results.pkl', 'rb') as f:
    dual_results = pickle.load(f)

for target_name in ['GDP_Growth', 'Ease_of_Business']:
    print(f"\n{'='*80}")
    print(f"Creating robustness plots for: {target_name}")
    print(f"{'='*80}")

    res = dual_results[target_name]
    best_name, best_res = res['best']
    model = best_res['model']
    y_test = res['y_test']
    X_test = res['X_test']

    y_pred = best_res['y_pred']
    residuals = y_test - y_pred

    print(f"  Model: {best_name}")
    print(f"  R²: {best_res['r2']:.4f}")
    print(f"  RMSE: {best_res['rmse']:.2f}")

    # --- FIGURE 1: 6-Panel Diagnostic Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Robustness Analysis - {target_name} ({best_name})',
                 fontsize=14, fontweight='bold')

    # 1. Q-Q Plot
    ax = axes[0, 0]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=9)
    ax.set_ylabel('Sample Quantiles', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Residual Distribution
    ax = axes[0, 1]
    ax.hist(residuals, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_title('Residual Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Residuals', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Predicted vs Actual
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

    # 4. Residuals vs Predicted
    ax = axes[1, 0]
    ax.scatter(y_pred, residuals, alpha=0.6, edgecolors='k', s=60, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_title('Residuals vs Predicted', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Values', fontsize=9)
    ax.set_ylabel('Residuals', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

    # 5. Cross-Validation Stability
    ax = axes[1, 1]
    try:
        from sklearn.model_selection import KFold
        X_train = res['X_test']  # Use available data
        y_train = res['y_test']

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')

        ax.bar(range(1, 6), cv_scores, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axhline(cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {cv_scores.mean():.3f}')
        ax.set_title('5-Fold Cross-Validation', fontsize=11, fontweight='bold')
        ax.set_xlabel('Fold', fontsize=9)
        ax.set_ylabel('R² Score', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    except Exception as e:
        ax.text(0.5, 0.5, f'CV not available\n{str(e)[:50]}',
                ha='center', va='center', fontsize=9)
        ax.set_title('5-Fold Cross-Validation', fontsize=11, fontweight='bold')

    # 6. Learning Curve
    ax = axes[1, 2]
    try:
        train_sizes = np.linspace(0.3, 1.0, 5)
        train_sizes_abs, train_scores, test_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes,
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
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
    except Exception as e:
        ax.text(0.5, 0.5, f'Learning curve not available\n{str(e)[:50]}',
                ha='center', va='center', fontsize=9)
        ax.set_title('Learning Curve', fontsize=11, fontweight='bold')

    plt.tight_layout()
    filename1 = f'resultados/graficos_finales/robustness_{target_name}.png'
    plt.savefig(filename1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename1}")

    # --- FIGURE 2: 4-Panel Detailed Residual Analysis ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Detailed Residual Analysis - {target_name}',
                 fontsize=14, fontweight='bold')

    # 1. Residual Sequence Plot
    ax = axes[0, 0]
    ax.plot(range(len(residuals)), residuals, 'o-', markersize=5, alpha=0.7, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.fill_between(range(len(residuals)),
                     -2*residuals.std(), 2*residuals.std(),
                     alpha=0.2, color='gray', label='±2 SD')
    ax.set_title('Residual Sequence Plot', fontsize=11, fontweight='bold')
    ax.set_xlabel('Observation Index', fontsize=9)
    ax.set_ylabel('Residuals', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. Scale-Location Plot
    ax = axes[0, 1]
    sqrt_abs_resid = np.sqrt(np.abs(residuals))
    ax.scatter(y_pred, sqrt_abs_resid, alpha=0.6, edgecolors='k', s=60, color='steelblue')
    ax.set_title('Scale-Location Plot', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Values', fontsize=9)
    ax.set_ylabel('√|Residuals|', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Bootstrap R² Distribution
    ax = axes[1, 0]
    bootstrap_scores = []
    rng = np.random.RandomState(42)
    for _ in range(100):
        indices = rng.randint(0, len(y_test), len(y_test))
        from sklearn.metrics import r2_score
        r2_boot = r2_score(y_test[indices], y_pred[indices])
        bootstrap_scores.append(r2_boot)

    ax.hist(bootstrap_scores, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(bootstrap_scores), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(bootstrap_scores):.3f}')
    ci_low, ci_high = np.percentile(bootstrap_scores, [2.5, 97.5])
    ax.axvline(ci_low, color='orange', linestyle=':', linewidth=2, label=f'95% CI')
    ax.axvline(ci_high, color='orange', linestyle=':', linewidth=2)
    ax.set_title('Bootstrap R² Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('R² Score', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Influential Points (Leverage vs Residuals)
    ax = axes[1, 1]
    std_resid = residuals / residuals.std()
    ax.scatter(range(len(std_resid)), std_resid, alpha=0.6, edgecolors='k',
              s=60, color='steelblue')
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
    filename2 = f'resultados/graficos_finales/residual_details_{target_name}.png'
    plt.savefig(filename2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename2}")

print("\n" + "=" * 80)
print("DUAL ROBUSTNESS ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated files:")
print("  - robustness_GDP_Growth.png")
print("  - residual_details_GDP_Growth.png")
print("  - robustness_Ease_of_Business.png")
print("  - residual_details_Ease_of_Business.png")
print("\nDone!")
