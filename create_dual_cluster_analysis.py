#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Cluster Analysis for Dual Target Project
Based on GDP Growth predictors (top 15 SHAP variables)
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DUAL TARGET CLUSTER ANALYSIS")
print("=" * 80)

# Load data and results
df = pd.read_excel('DATA_GHAB2.xlsx')

with open('resultados/modelos/dual_target_results.pkl', 'rb') as f:
    dual_results = pickle.load(f)

# Use GDP Growth predictors (the reliable model)
gdp_res = dual_results['GDP_Growth']
predictors = gdp_res['predictors'][:15]  # Top 15

print(f"\nUsing top 15 predictors from GDP Growth model")
print(f"Number of variables: {len(predictors)}")

# Prepare data
mask = df[predictors[0]].notna()
for var in predictors[1:]:
    mask &= df[var].notna()

X = df.loc[mask, predictors].values
countries = df.loc[mask, 'Country'].values if 'Country' in df.columns else df.loc[mask].index.values

print(f"Countries with complete data: {len(countries)}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with k=3
optimal_k = 3
print(f"\nPerforming K-Means clustering with k={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

print(f"Cluster distribution:")
for i in range(optimal_k):
    count = (clusters == i).sum()
    print(f"  Cluster {i}: {count} countries")

# PCA for visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
variance_explained = pca.explained_variance_ratio_

print(f"\nPCA Variance explained:")
print(f"  PC1: {variance_explained[0]*100:.1f}%")
print(f"  PC2: {variance_explained[1]*100:.1f}%")
print(f"  Total: {variance_explained.sum()*100:.1f}%")

# Save cluster assignments
cluster_df = pd.DataFrame({
    'Country': countries,
    'Cluster': clusters,
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1]
})
cluster_df.to_csv('resultados/cluster_assignments.csv', index=False)
print(f"\nCluster assignments saved: resultados/cluster_assignments.csv")

# --- FIGURE: Cluster Analysis ---
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Cluster Analysis - Gender & Development Indicators',
             fontsize=14, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# LEFT: Scatter with confidence ellipses
ax = axes[0]

for cluster_id in range(optimal_k):
    mask_cluster = clusters == cluster_id
    cluster_points = X_pca[mask_cluster]

    # Scatter points
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
              s=100, alpha=0.6, edgecolors='k', linewidth=1,
              color=colors[cluster_id],
              label=f'Cluster {cluster_id} (n={mask_cluster.sum()})')

    # Add country labels
    for i, (x, y) in enumerate(cluster_points):
        country = countries[mask_cluster][i]
        # Truncate country names
        country_short = str(country)[:15]
        ax.text(x, y, country_short, fontsize=7, ha='center', va='bottom', alpha=0.8)

    # Confidence ellipse (95%)
    if mask_cluster.sum() > 2:
        mean = cluster_points.mean(axis=0)
        cov = np.cov(cluster_points.T)

        # Eigenvalues and eigenvectors for ellipse
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

        # 95% confidence (chi-square quantile for 2 DOF)
        chi2_val = 5.991  # 95% for 2 DOF
        width, height = 2 * np.sqrt(chi2_val * eigenvalues)

        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor=colors[cluster_id],
                         linewidth=2, linestyle='--', alpha=0.8)
        ax.add_patch(ellipse)

ax.set_xlabel(f'PC1 ({variance_explained[0]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax.set_ylabel(f'PC2 ({variance_explained[1]*100:.1f}% variance)', fontsize=11, fontweight='bold')
ax.set_title('Country Clusters with 95% Confidence Ellipses', fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=9)

# RIGHT: Heatmap of cluster characteristics
ax = axes[1]

# Calculate mean standardized values for top 10 variables by cluster
top_10_vars = predictors[:10]
cluster_profiles = []

for cluster_id in range(optimal_k):
    mask_cluster = clusters == cluster_id
    cluster_mean = X_scaled[mask_cluster].mean(axis=0)[:10]  # Top 10
    cluster_profiles.append(cluster_mean)

cluster_profiles = np.array(cluster_profiles)

# Heatmap
im = ax.imshow(cluster_profiles.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

# Labels
var_labels = [v[:35] + '...' if len(v) > 35 else v for v in top_10_vars]
ax.set_yticks(range(len(var_labels)))
ax.set_yticklabels(var_labels, fontsize=7)
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f'Cluster {i}' for i in range(optimal_k)], fontsize=9)
ax.set_title('Cluster Characteristics (Top 10 Variables)', fontsize=12, fontweight='bold')

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Standardized Mean Value', fontsize=9)
cbar.ax.tick_params(labelsize=8)

# Add values in cells
for i in range(len(var_labels)):
    for j in range(optimal_k):
        value = cluster_profiles[j, i]
        color = 'white' if abs(value) > 1 else 'black'
        ax.text(j, i, f'{value:.1f}', ha='center', va='center',
               fontsize=7, color=color, fontweight='bold')

plt.tight_layout()
filename = 'resultados/graficos_finales/cluster_analysis.png'
plt.savefig(filename, dpi=200, bbox_inches='tight')
plt.close()

print(f"\nCluster visualization saved: {filename}")

print("\n" + "=" * 80)
print("CLUSTER ANALYSIS COMPLETE")
print("=" * 80)
print("\nFiles generated:")
print("  - cluster_analysis.png")
print("  - cluster_assignments.csv")
print("\nDone!")
