#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge BASE_COMPLETA with DATA_GHAB2 and train models for all G variables
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, shapiro
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MERGE DATABASES AND TRAIN ALL G VARIABLES")
print("="*80)

# Load databases
df_ghab = pd.read_excel('DATA_GHAB2.xlsx')
df_base = pd.read_excel('BASE_COMPLETA.xlsx')

print(f"\nDATA_GHAB2: {df_ghab.shape}")
print(f"BASE_COMPLETA: {df_base.shape}")

# Create country code mapping (ISO3 codes)
# We need to map country names to ISO3 codes
country_to_iso3 = {
    'Afghanistan': 'AFG', 'Bangladesh': 'BGD', 'Benin': 'BEN', 'Burkina Faso': 'BFA',
    'Burundi': 'BDI', 'Cambodia': 'KHM', 'Cameroon': 'CMR', 'Central African Republic': 'CAF',
    'Chad': 'TCD', 'Comoros': 'COM', 'Congo, Dem. Rep.': 'COD', 'Congo, Rep.': 'COG',
    'Djibouti': 'DJI', 'Egypt, Arab Rep.': 'EGY', 'El Salvador': 'SLV', 'Ethiopia': 'ETH',
    'Gambia, The': 'GMB', 'Ghana': 'GHA', 'Guinea': 'GIN', 'Guinea-Bissau': 'GNB',
    'Haiti': 'HTI', 'Honduras': 'HND', 'India': 'IND', 'Kenya': 'KEN',
    'Kyrgyz Republic': 'KGZ', 'Lao PDR': 'LAO', 'Lesotho': 'LSO', 'Liberia': 'LBR',
    'Madagascar': 'MDG', 'Malawi': 'MWI', 'Mali': 'MLI', 'Mauritania': 'MRT',
    'Mozambique': 'MOZ', 'Myanmar': 'MMR', 'Nepal': 'NPL', 'Nicaragua': 'NIC',
    'Niger': 'NER', 'Nigeria': 'NGA', 'Pakistan': 'PAK', 'Papua New Guinea': 'PNG',
    'Rwanda': 'RWA', 'Senegal': 'SEN', 'Sierra Leone': 'SLE', 'Solomon Islands': 'SLB',
    'Somalia': 'SOM', 'South Sudan': 'SSD', 'Sudan': 'SDN', 'Syrian Arab Republic': 'SYR',
    'Tajikistan': 'TJK', 'Tanzania': 'TZA', 'Togo': 'TGO', 'Uganda': 'UGA',
    'Yemen, Rep.': 'YEM', 'Zambia': 'ZMB', 'Zimbabwe': 'ZWE'
}

# Add country code to DATA_GHAB2
df_ghab['Country code'] = df_ghab['Pais'].map(country_to_iso3)

print(f"\nCountries successfully mapped: {df_ghab['Country code'].notna().sum()}")
print(f"Countries without mapping: {df_ghab['Country code'].isna().sum()}")

if df_ghab['Country code'].isna().any():
    print("\nUnmapped countries:")
    print(df_ghab[df_ghab['Country code'].isna()]['Pais'].tolist())

# Merge databases
df_merged = df_ghab.merge(df_base, on='Country code', how='left')

print(f"\nMerged database: {df_merged.shape}")
print(f"Successfully merged countries: {df_merged.notna().any(axis=1).sum()}")

# Identify all G variables (excluding G_GPD_PCAP_SLOPE which is our primary target)
all_g_vars = [c for c in df_merged.columns if c.startswith('G_')]
g_targets = [c for c in all_g_vars if c != 'G_GPD_PCAP_SLOPE']

print(f"\n{'='*80}")
print(f"G variables to test as targets: {len(g_targets)}")
print(f"{'='*80}")

for i, var in enumerate(g_targets, 1):
    print(f"{i:2d}. {var}")

# Get predictors (all non-G variables except targets and identifiers)
exclude_cols = ['Pais', 'Etiqueta_pais', 'Country code', 'Ease of Doing Business',
                'New_Business_Density'] + all_g_vars
potential_predictors = [col for col in df_merged.columns if col not in exclude_cols]

print(f"\nPotential predictors: {len(potential_predictors)}")

# Train models for each G variable
all_g_results = {}

for target_var in g_targets:
    print(f"\n{'='*80}")
    print(f"TRAINING: {target_var}")
    print(f"{'='*80}")

    # Check data availability
    valid_mask = df_merged[target_var].notna()
    n_valid = valid_mask.sum()

    print(f"Valid observations: {n_valid}")

    if n_valid < 20:
        print(f"WARNING: Insufficient data (< 20 observations). Skipping.")
        continue

    # Calculate correlations
    correlations = []
    for var in potential_predictors:
        mask = valid_mask & df_merged[var].notna()
        if mask.sum() >= 10:
            try:
                r, p = pearsonr(df_merged.loc[mask, var], df_merged.loc[mask, target_var])
                correlations.append({
                    'Variable': var,
                    'Correlation': abs(r),
                    'R_value': r,
                    'P_value': p
                })
            except:
                pass

    if len(correlations) < 5:
        print(f"WARNING: Insufficient correlations found. Skipping.")
        continue

    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    print(f"Correlated variables found: {len(corr_df)}")

    # Select top 15 predictors
    top_predictors = corr_df.head(15)['Variable'].tolist()
    print(f"Using top {len(top_predictors)} predictors")

    # Prepare data
    mask = df_merged[target_var].notna()
    for var in top_predictors:
        mask &= df_merged[var].notna()

    if mask.sum() < 15:
        print(f"WARNING: Final dataset too small ({mask.sum()} obs). Skipping.")
        continue

    X = df_merged.loc[mask, top_predictors].values
    y = df_merged.loc[mask, target_var].values

    print(f"Final dataset: {X.shape[0]} observations, {X.shape[1]} features")

    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    except:
        print("WARNING: Cannot split data. Skipping.")
        continue

    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    models = {}

    print("  Training models...")

    # 1. Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                                  min_samples_split=2, min_samples_leaf=1,
                                  random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        cv_rf = cross_val_score(rf, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()

        models['Random Forest'] = {
            'model': rf,
            'y_pred': y_pred_rf,
            'r2': r2_score(y_test, y_pred_rf),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'cv': cv_rf,
            'importance': dict(zip(top_predictors, rf.feature_importances_))
        }
    except Exception as e:
        print(f"    Random Forest failed: {str(e)[:50]}")

    # 2. XGBoost
    try:
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                          random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        cv_xgb = cross_val_score(xgb, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()

        models['XGBoost'] = {
            'model': xgb,
            'y_pred': y_pred_xgb,
            'r2': r2_score(y_test, y_pred_xgb),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'mae': mean_absolute_error(y_test, y_pred_xgb),
            'cv': cv_xgb
        }
    except Exception as e:
        print(f"    XGBoost failed: {str(e)[:50]}")

    # 3. Gradient Boosting
    try:
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                      learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        cv_gb = cross_val_score(gb, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()

        models['Gradient Boosting'] = {
            'model': gb,
            'y_pred': y_pred_gb,
            'r2': r2_score(y_test, y_pred_gb),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
            'mae': mean_absolute_error(y_test, y_pred_gb),
            'cv': cv_gb
        }
    except Exception as e:
        print(f"    Gradient Boosting failed: {str(e)[:50]}")

    # 4. Neural Network
    try:
        nn = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu',
                         max_iter=1000, random_state=42, early_stopping=True)
        nn.fit(X_train, y_train)
        y_pred_nn = nn.predict(X_test)
        cv_nn = cross_val_score(nn, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()

        models['Neural Network'] = {
            'model': nn,
            'y_pred': y_pred_nn,
            'r2': r2_score(y_test, y_pred_nn),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
            'mae': mean_absolute_error(y_test, y_pred_nn),
            'cv': cv_nn
        }
    except Exception as e:
        print(f"    Neural Network failed: {str(e)[:50]}")

    if len(models) == 0:
        print("WARNING: All models failed. Skipping.")
        continue

    # Rank models
    ranked = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)
    best_name, best_res = ranked[0]

    print(f"\n  Best model: {best_name} (R²={best_res['r2']:.4f}, CV={best_res['cv']:.4f})")

    # Validation
    residuals = y_test - best_res['y_pred']

    try:
        shapiro_stat, shapiro_p = shapiro(residuals)
    except:
        shapiro_p = np.nan

    # Bootstrap
    n_bootstrap = 100
    bootstrap_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(y_test), len(y_test))
        r2_boot = r2_score(y_test[indices], best_res['y_pred'][indices])
        bootstrap_scores.append(r2_boot)

    bootstrap_ci = np.percentile(bootstrap_scores, [2.5, 97.5])

    validation = {
        'shapiro_p': shapiro_p,
        'bootstrap_r2_mean': np.mean(bootstrap_scores),
        'bootstrap_ci': bootstrap_ci
    }

    # Store results
    all_g_results[target_var] = {
        'target_column': target_var,
        'predictors': top_predictors,
        'best': (best_name, best_res),
        'top_3': ranked[:min(3, len(ranked))],
        'all_models': models,
        'validation': validation,
        'y_test': y_test,
        'X_test': X_test,
        'n_observations': mask.sum()
    }

# Save all results
with open('resultados/modelos/all_G_variables_results.pkl', 'wb') as f:
    pickle.dump(all_g_results, f)

# Save merged database
df_merged.to_excel('DATA_MERGED_COMPLETE.xlsx', index=False)

print("\n" + "="*80)
print("SUMMARY OF ALL G VARIABLES")
print("="*80)

# Sort by R² score
sorted_results = sorted(all_g_results.items(),
                       key=lambda x: x[1]['best'][1]['r2'],
                       reverse=True)

print(f"\n{'Rank':<5} {'Variable':<50} {'R²':<8} {'CV':<8} {'N':<6} {'Model':<20}")
print("-"*100)

for i, (var, res) in enumerate(sorted_results, 1):
    best_name, best_res = res['best']
    var_short = var[:47] + '...' if len(var) > 47 else var
    print(f"{i:<5} {var_short:<50} {best_res['r2']:>6.3f}  {best_res['cv']:>6.3f}  {res['n_observations']:<6} {best_name:<20}")

# Identify best G variable
if sorted_results:
    best_g_var, best_g_res = sorted_results[0]
    best_model_name, best_model_res = best_g_res['best']

    print(f"\n{'='*80}")
    print(f"BEST G VARIABLE: {best_g_var}")
    print(f"{'='*80}")
    print(f"  Best Model: {best_model_name}")
    print(f"  R² Score: {best_model_res['r2']:.4f}")
    print(f"  CV R²: {best_model_res['cv']:.4f}")
    print(f"  RMSE: {best_model_res['rmse']:.2f}")
    print(f"  MAE: {best_model_res['mae']:.2f}")
    print(f"  Observations: {best_g_res['n_observations']}")
    print(f"  Bootstrap 95% CI: [{best_g_res['validation']['bootstrap_ci'][0]:.3f}, {best_g_res['validation']['bootstrap_ci'][1]:.3f}]")

print(f"\nResults saved: resultados/modelos/all_G_variables_results.pkl")
print(f"Merged database saved: DATA_MERGED_COMPLETE.xlsx")
