#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Search for Best Non-G Variable from Updated BASE_COMPLETA
Uses Country code2 for merging
Balanced selection: simultaneous R² and CV optimization
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
print("SEARCHING NEW VARIABLES (NON-G) FROM UPDATED BASE_COMPLETA")
print("Merging using Country code2")
print("Balanced selection: simultaneous R² and CV optimization")
print("="*80)

# Load databases
df_ghab = pd.read_excel('DATA_GHAB2.xlsx')
df_base = pd.read_excel('BASE_COMPLETA.xlsx')

print(f"\nDATA_GHAB2: {df_ghab.shape}")
print(f"BASE_COMPLETA (updated): {df_base.shape}")

# Check if Country code2 exists in BASE_COMPLETA
if 'Country code2' in df_base.columns:
    merge_col = 'Country code2'
    print(f"\nUsing '{merge_col}' for merging")
else:
    print(f"\nWARNING: 'Country code2' not found in BASE_COMPLETA")
    print(f"Available columns: {df_base.columns.tolist()[:10]}...")
    # Try alternative column names
    potential_cols = [c for c in df_base.columns if 'country' in c.lower() or 'code' in c.lower()]
    if potential_cols:
        merge_col = potential_cols[0]
        print(f"Using alternative: '{merge_col}'")
    else:
        raise ValueError("No suitable merge column found")

# Check if Country code2 exists in DATA_GHAB2, if not create mapping
if 'Country code2' not in df_ghab.columns:
    # Check if we have Country code
    if 'Country code' in df_ghab.columns:
        df_ghab['Country code2'] = df_ghab['Country code']
        print(f"\nCopied 'Country code' to 'Country code2' in DATA_GHAB2")
    else:
        # Create from Pais mapping
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
        df_ghab['Country code2'] = df_ghab['Pais'].map(country_to_iso3)
        print(f"\nCreated 'Country code2' from Pais mapping")

# Merge
df_merged = df_ghab.merge(df_base, left_on='Country code2', right_on=merge_col, how='left')
print(f"\nMerged database: {df_merged.shape}")
print(f"Countries with merged data: {df_merged[merge_col].notna().sum()}")

# Identify NEW variables (those that DON'T start with G_)
# Exclude G variables, GDP slope, and metadata columns
all_g_vars = [c for c in df_merged.columns if c.startswith('G_')]
exclude_cols = (['Pais', 'Etiqueta_pais', 'Country code', 'Country code2',
                 'Ease of Doing Business', 'New_Business_Density', 'Country'] +
                all_g_vars)

# Get all columns from BASE_COMPLETA that were merged (excluding merge column)
base_cols = [c for c in df_base.columns if c != merge_col]

# NEW variables are those from BASE_COMPLETA that don't start with G_
new_target_candidates = [c for c in base_cols
                         if c not in exclude_cols
                         and not c.startswith('G_')
                         and c in df_merged.columns]

print(f"\n{'='*80}")
print(f"NEW VARIABLES TO TEST (non-G from BASE_COMPLETA): {len(new_target_candidates)}")
print(f"{'='*80}")

if len(new_target_candidates) > 0:
    print(f"\nFirst 10 new variables:")
    for i, var in enumerate(new_target_candidates[:10], 1):
        print(f"  {i}. {var}")
else:
    print("\nWARNING: No new non-G variables found!")
    print(f"Total columns in merged: {len(df_merged.columns)}")
    print(f"Columns from BASE_COMPLETA: {len(base_cols)}")

# Get predictors (gender/development indicators from DATA_GHAB2)
potential_predictors = [col for col in df_merged.columns if col not in exclude_cols]
# Remove new target candidates from predictors
potential_predictors = [col for col in potential_predictors if col not in new_target_candidates]

print(f"\nPotential predictors (from DATA_GHAB2): {len(potential_predictors)}")

# Balanced scoring function
def get_balanced_score(r2, cv):
    """
    Balanced score considering BOTH R² and CV simultaneously
    - Requires BOTH R² > 0.3 AND CV > 0 for full credit
    - Penalizes if either is too low
    - Final score = sqrt(R²) × sqrt(CV) if both positive and R²>0.3
    """
    if r2 < 0 or cv < 0:
        return 0.0  # Reject negative values

    if r2 < 0.3:
        return r2 * 0.2  # Heavy penalty for very low R²

    # Geometric mean gives balanced weight to both metrics
    return np.sqrt(max(0, r2)) * np.sqrt(cv)

# Train models for each new variable
all_results = {}

for target_var in new_target_candidates:
    print(f"\n{'='*60}")
    print(f"Testing: {target_var[:50]}")

    valid_mask = df_merged[target_var].notna()
    n_valid = valid_mask.sum()

    if n_valid < 20:
        print(f"  Skipped: insufficient data ({n_valid})")
        continue

    # Correlations
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
        print(f"  Skipped: insufficient correlations")
        continue

    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    top_predictors = corr_df.head(15)['Variable'].tolist()

    # Prepare data
    mask = df_merged[target_var].notna()
    for var in top_predictors:
        mask &= df_merged[var].notna()

    if mask.sum() < 15:
        print(f"  Skipped: final dataset too small")
        continue

    X = df_merged.loc[mask, top_predictors].values
    y = df_merged.loc[mask, target_var].values

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    except:
        print(f"  Skipped: cannot split")
        continue

    # Train all 4 models
    models = {}

    # Random Forest
    try:
        rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                                  min_samples_split=2, min_samples_leaf=1,
                                  random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        cv_rf = cross_val_score(rf, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()
        r2_rf = r2_score(y_test, y_pred_rf)

        models['Random Forest'] = {
            'model': rf,
            'y_pred': y_pred_rf,
            'r2': r2_rf,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'cv': cv_rf,
            'balanced_score': get_balanced_score(r2_rf, cv_rf),
            'importance': dict(zip(top_predictors, rf.feature_importances_))
        }
    except:
        pass

    # XGBoost
    try:
        xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                          random_state=42, n_jobs=-1)
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        cv_xgb = cross_val_score(xgb, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()
        r2_xgb = r2_score(y_test, y_pred_xgb)

        models['XGBoost'] = {
            'model': xgb,
            'y_pred': y_pred_xgb,
            'r2': r2_xgb,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'mae': mean_absolute_error(y_test, y_pred_xgb),
            'cv': cv_xgb,
            'balanced_score': get_balanced_score(r2_xgb, cv_xgb)
        }
    except:
        pass

    # Gradient Boosting
    try:
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                      learning_rate=0.1, random_state=42)
        gb.fit(X_train, y_train)
        y_pred_gb = gb.predict(X_test)
        cv_gb = cross_val_score(gb, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()
        r2_gb = r2_score(y_test, y_pred_gb)

        models['Gradient Boosting'] = {
            'model': gb,
            'y_pred': y_pred_gb,
            'r2': r2_gb,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
            'mae': mean_absolute_error(y_test, y_pred_gb),
            'cv': cv_gb,
            'balanced_score': get_balanced_score(r2_gb, cv_gb)
        }
    except:
        pass

    # Neural Network
    try:
        nn = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu',
                         max_iter=1000, random_state=42, early_stopping=True)
        nn.fit(X_train, y_train)
        y_pred_nn = nn.predict(X_test)
        cv_nn = cross_val_score(nn, X_train, y_train, cv=min(5, len(X_train)), scoring='r2').mean()
        r2_nn = r2_score(y_test, y_pred_nn)

        models['Neural Network'] = {
            'model': nn,
            'y_pred': y_pred_nn,
            'r2': r2_nn,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
            'mae': mean_absolute_error(y_test, y_pred_nn),
            'cv': cv_nn,
            'balanced_score': get_balanced_score(r2_nn, cv_nn)
        }
    except:
        pass

    if len(models) == 0:
        print(f"  Skipped: all models failed")
        continue

    # Select best by BALANCED SCORE
    ranked = sorted(models.items(),
                   key=lambda x: x[1]['balanced_score'],
                   reverse=True)
    best_name, best_res = ranked[0]

    print(f"  Best: {best_name}")
    print(f"    R²={best_res['r2']:.3f}, CV={best_res['cv']:.3f}, Balanced={best_res['balanced_score']:.3f}")

    # Validation
    residuals = y_test - best_res['y_pred']
    try:
        shapiro_stat, shapiro_p = shapiro(residuals)
    except:
        shapiro_p = np.nan

    # Bootstrap
    bootstrap_scores = []
    rng = np.random.RandomState(42)
    for _ in range(100):
        indices = rng.randint(0, len(y_test), len(y_test))
        r2_boot = r2_score(y_test[indices], best_res['y_pred'][indices])
        bootstrap_scores.append(r2_boot)

    bootstrap_ci = np.percentile(bootstrap_scores, [2.5, 97.5])

    # Store
    all_results[target_var] = {
        'target_column': target_var,
        'predictors': top_predictors,
        'best': (best_name, best_res),
        'top_3': ranked[:min(3, len(ranked))],
        'all_models': models,
        'validation': {
            'shapiro_p': shapiro_p,
            'bootstrap_r2_mean': np.mean(bootstrap_scores),
            'bootstrap_ci': bootstrap_ci
        },
        'y_test': y_test,
        'X_test': X_test,
        'n_observations': mask.sum(),
        'balanced_score': best_res['balanced_score']
    }

# Save
with open('resultados/modelos/new_variables_balanced.pkl', 'wb') as f:
    pickle.dump(all_results, f)

# Save merged data
df_merged.to_excel('DATA_MERGED_NEW.xlsx', index=False)

print("\n" + "="*80)
print("RANKING BY BALANCED SCORE (sqrt(R²) × sqrt(CV))")
print("="*80)

# Sort by balanced score
sorted_results = sorted(all_results.items(),
                       key=lambda x: x[1]['balanced_score'],
                       reverse=True)

print(f"\n{'Rank':<5} {'Variable':<45} {'R²':<7} {'CV':<7} {'Bal':<7} {'N':<5} {'Model':<15}")
print("-"*95)

for i, (var, res) in enumerate(sorted_results, 1):
    best_name, best_res = res['best']
    var_short = var[:42] + '...' if len(var) > 42 else var
    print(f"{i:<5} {var_short:<45} {best_res['r2']:>5.3f}  {best_res['cv']:>5.3f}  {res['balanced_score']:>5.3f}  {res['n_observations']:<5} {best_name:<15}")

# Best variable
if sorted_results:
    best_var, best_var_res = sorted_results[0]
    best_model_name, best_model_res = best_var_res['best']

    print(f"\n{'='*80}")
    print(f"BEST NEW VARIABLE: {best_var}")
    print(f"{'='*80}")
    print(f"  Model: {best_model_name}")
    print(f"  R²: {best_model_res['r2']:.4f}")
    print(f"  CV: {best_model_res['cv']:.4f}")
    print(f"  Balanced Score: {best_var_res['balanced_score']:.4f}")
    print(f"  RMSE: {best_model_res['rmse']:.2f}")
    print(f"  N: {best_var_res['n_observations']}")
    print(f"  Bootstrap CI: [{best_var_res['validation']['bootstrap_ci'][0]:.3f}, {best_var_res['validation']['bootstrap_ci'][1]:.3f}]")

    if best_model_res['r2'] > 0.3 and best_model_res['cv'] > 0:
        print(f"\n  Status: GOOD - Both R² and CV are positive and adequate")
    else:
        print(f"\n  Status: Consider next best option")
else:
    print("\nNo suitable variables found with balanced score > 0")

print(f"\nResults saved: resultados/modelos/new_variables_balanced.pkl")
print(f"Merged data saved: DATA_MERGED_NEW.xlsx")
print("\nNote: Balanced score = sqrt(R²) × sqrt(CV) when R²>0.3 and CV>0")
print("This ensures BOTH metrics are good simultaneously")
