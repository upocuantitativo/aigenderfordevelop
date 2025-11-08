#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Descriptive Analysis with Correlations and Clusters
For the current best model (Tax Score)
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CREATING DESCRIPTIVE ANALYSIS")
print("Correlations and Cluster Analysis")
print("="*80)

# Load data and results
df = pd.read_excel('DATA_MERGED_NEW.xlsx')

with open('resultados/modelos/final_dual_improved.pkl', 'rb') as f:
    results = pickle.load(f)

# Use Tax Score (the successful model)
tax_res = results['Tax_Score']
predictors = tax_res['predictors'][:15]  # Top 15

print(f"\nUsing top 15 predictors from Tax Score model")
print(f"Number of variables: {len(predictors)}")

# ============================================================
# PART 1: CORRELATION MATRIX
# ============================================================
print("\n" + "="*80)
print("PART 1: CORRELATION ANALYSIS")
print("="*80)

# Prepare data for correlation
target_col = tax_res['target_column']
print(f"\nTarget variable: {target_col}")

# Get correlations for top 15 predictors
corr_data = []
for pred in predictors:
    mask = df[pred].notna() & df[target_col].notna()
    if mask.sum() >= 10:
        try:
            r, p = pearsonr(df.loc[mask, pred], df.loc[mask, target_col])
            corr_data.append({
                'Variable': pred,
                'Correlation': r,
                'Abs_Correlation': abs(r),
                'P_value': p,
                'N': mask.sum()
            })
        except:
            pass

corr_df = pd.DataFrame(corr_data).sort_values('Abs_Correlation', ascending=False)

print(f"\nTop 10 Correlations with {target_col[:50]}:")
for i, row in corr_df.head(10).iterrows():
    var_short = row['Variable'][:60] + '...' if len(row['Variable']) > 60 else row['Variable']
    print(f"  {row['Abs_Correlation']:.3f}  {var_short}")

# Save correlation table
corr_df.to_csv('resultados/correlations_tax_score.csv', index=False)
print(f"\nCorrelation table saved: resultados/correlations_tax_score.csv")

# Visualize correlations
fig, ax = plt.subplots(figsize=(10, 8))

top_10_corr = corr_df.head(10)
y_pos = np.arange(len(top_10_corr))
colors = ['#2ca02c' if c > 0 else '#d62728' for c in top_10_corr['Correlation'].values]

var_labels = [v[:45] + '...' if len(v) > 45 else v for v in top_10_corr['Variable'].values]

ax.barh(y_pos, top_10_corr['Correlation'].values, color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(y_pos)
ax.set_yticklabels(var_labels, fontsize=8)
ax.set_xlabel('Pearson Correlation Coefficient', fontsize=10, fontweight='bold')
ax.set_title(f'Top 10 Correlations with {target_col[:50]}', fontsize=12, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('resultados/graficos_finales/correlation_analysis.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"Correlation plot saved: resultados/graficos_finales/correlation_analysis.png")

# ============================================================
# PART 2: CLUSTER ANALYSIS
# ============================================================
print("\n" + "="*80)
print("PART 2: CLUSTER ANALYSIS")
print("="*80)

# Prepare data for clustering
mask = df[predictors[0]].notna()
for var in predictors[1:]:
    mask &= df[var].notna()

X = df.loc[mask, predictors].values
countries = df.loc[mask, 'Pais'].values if 'Pais' in df.columns else df.loc[mask].index.values

print(f"\nCountries with complete data: {len(countries)}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means with k=3
optimal_k = 3
print(f"Performing K-Means clustering with k={optimal_k}...")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_scaled)

print(f"\nCluster distribution:")
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
cluster_df.to_csv('resultados/cluster_assignments_tax.csv', index=False)
print(f"\nCluster assignments saved: resultados/cluster_assignments_tax.csv")

# ============================================================
# PART 3: CLUSTER VISUALIZATION
# ============================================================
print("\n" + "="*80)
print("PART 3: CREATING CLUSTER VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle('Cluster Analysis - Tax Score Predictors (Excluding H_Cause of death)',
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
        country_short = str(country)[:15]
        ax.text(x, y, country_short, fontsize=7, ha='center', va='bottom', alpha=0.8)

    # Confidence ellipse (95%)
    if mask_cluster.sum() > 2:
        mean = cluster_points.mean(axis=0)
        cov = np.cov(cluster_points.T)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

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

top_10_vars = predictors[:10]
cluster_profiles = []

for cluster_id in range(optimal_k):
    mask_cluster = clusters == cluster_id
    cluster_mean = X_scaled[mask_cluster].mean(axis=0)[:10]  # Top 10
    cluster_profiles.append(cluster_mean)

cluster_profiles = np.array(cluster_profiles)

# Heatmap
im = ax.imshow(cluster_profiles.T, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)

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
plt.savefig('resultados/graficos_finales/cluster_analysis_tax.png', dpi=200, bbox_inches='tight')
plt.close()

print(f"Cluster visualization saved: resultados/graficos_finales/cluster_analysis_tax.png")

print("\n" + "="*80)
print("DESCRIPTIVE ANALYSIS COMPLETE")
print("="*80)
print("\nFiles generated:")
print("  - correlations_tax_score.csv")
print("  - correlation_analysis.png")
print("  - cluster_assignments_tax.csv")
print("  - cluster_analysis_tax.png")
print("\nDone!")
