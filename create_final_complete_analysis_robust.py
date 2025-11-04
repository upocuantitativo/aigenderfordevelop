#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create complete analysis for final dual targets:
GDP Growth + G_Score-Trading across borders
Includes: Robustness, Clusters, Dashboard
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, learning_curve, KFold
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy import stats
from matplotlib.patches import Ellipse
from datetime import datetime
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING COMPLETE FINAL ANALYSIS")
print("="*80)

# Load results and data
with open('resultados/modelos/final_dual_target_results_robust.pkl', 'rb') as f:
    final_results = pickle.load(f)

df = pd.read_excel('DATA_MERGED_COMPLETE.xlsx')

# ========== PART 1: ROBUSTNESS ANALYSIS ==========
print("\n1. Creating robustness analysis...")

for target_name in ['GDP_Growth', 'Trading_Score']:
    print(f"\n  Processing {target_name}...")
    res = final_results[target_name]
    best_name, best_res = res['best']
    model = best_res['model']
    y_test = res['y_test']
    X_test = res['X_test']
    y_pred = best_res['y_pred']
    residuals = y_test - y_pred

    # Figure 1: 6-panel diagnostics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Robustness Analysis - {target_name.replace("_", " ")} ({best_name})',
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
    plt.savefig(f'resultados/graficos_finales/robustness_robust_{target_name}.png', dpi=200, bbox_inches='tight')
    plt.close()

    # Figure 2: Residual details
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Detailed Residual Analysis - {target_name.replace("_", " ")}',
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
    plt.savefig(f'resultados/graficos_finales/residual_details_robust_{target_name}.png', dpi=200, bbox_inches='tight')
    plt.close()

print("  Robustness analysis complete!")

# ========== PART 2: CLUSTER ANALYSIS ==========
print("\n2. Creating cluster analysis...")

# Use GDP Growth predictors for clustering
gdp_res = final_results['GDP_Growth']
predictors = gdp_res['predictors'][:15]

mask = df[predictors[0]].notna()
for var in predictors[1:]:
    mask &= df[var].notna()

X = df.loc[mask, predictors].values
countries = df.loc[mask, 'Pais'].values if 'Pais' in df.columns else df.loc[mask].index.values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
variance_explained = pca.explained_variance_ratio_

# Save assignments
cluster_df = pd.DataFrame({
    'Country': countries,
    'Cluster': clusters,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})
cluster_df.to_csv('resultados/cluster_assignments_final_robust.csv', index=False)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Cluster Analysis - Gender & Development Indicators',
             fontsize=14, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
ax = axes[0]

for cluster_id in range(3):
    mask_cluster = clusters == cluster_id
    cluster_points = X_pca[mask_cluster]

    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
              s=100, alpha=0.6, edgecolors='k', linewidth=1,
              color=colors[cluster_id],
              label=f'Cluster {cluster_id} (n={mask_cluster.sum()})')

    for i, (x, y) in enumerate(cluster_points):
        country = str(countries[mask_cluster][i])[:15]
        ax.text(x, y, country, fontsize=7, ha='center', va='bottom', alpha=0.8)

    if mask_cluster.sum() > 2:
        mean = cluster_points.mean(axis=0)
        cov = np.cov(cluster_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        chi2_val = 5.991
        width, height = 2 * np.sqrt(chi2_val * eigenvalues)
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor=colors[cluster_id],
                         linewidth=2, linestyle='--', alpha=0.8)
        ax.add_patch(ellipse)

ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax.set_title('Country Clusters with 95% Confidence Ellipses', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Heatmap
ax = axes[1]
top_10_vars = predictors[:10]
cluster_profiles = []
for cluster_id in range(3):
    mask_cluster = clusters == cluster_id
    cluster_mean = X_scaled[mask_cluster].mean(axis=0)[:10]
    cluster_profiles.append(cluster_mean)
cluster_profiles = np.array(cluster_profiles)

im = ax.imshow(cluster_profiles.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
var_labels = [v[:35] + '...' if len(v) > 35 else v for v in top_10_vars]
ax.set_yticks(range(len(var_labels)))
ax.set_yticklabels(var_labels, fontsize=7)
ax.set_xticks(range(3))
ax.set_xticklabels([f'Cluster {i}' for i in range(3)], fontsize=9)
ax.set_title('Cluster Characteristics', fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Standardized Mean Value', fontsize=9)
cbar.ax.tick_params(labelsize=8)

for i in range(len(var_labels)):
    for j in range(3):
        value = cluster_profiles[j, i]
        color = 'white' if abs(value) > 1 else 'black'
        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
               fontsize=7, color=color, fontweight='bold')

plt.tight_layout()
plt.savefig('resultados/graficos_finales/cluster_analysis_final_robust.png', dpi=200, bbox_inches='tight')
plt.close()

print("  Cluster analysis complete!")

print("\n" + "="*80)
print("COMPLETE ANALYSIS FINISHED")
print("="*80)
print("\nFiles created:")
print("  - robustness_robust_GDP_Growth.png")
print("  - robustness_robust_Trading_Score.png")
print("  - residual_details_robust_GDP_Growth.png")
print("  - residual_details_robust_Trading_Score.png")
print("  - cluster_analysis_final_robust.png")
print("  - cluster_assignments_final_robust.csv")
