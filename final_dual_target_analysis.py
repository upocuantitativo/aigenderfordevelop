#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Dual Target Analysis: GDP Growth + Best G Variable
G_GPD_PCAP_SLOPE + G_Score-Trading across borders (DB16-20 methodology)
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
print("FINAL DUAL TARGET ANALYSIS")
print("GDP Growth + G_Score-Trading across borders")
print("="*80)

# Load merged data
df = pd.read_excel('DATA_MERGED_COMPLETE.xlsx')
print(f"\nMerged data loaded: {df.shape[0]} countries, {df.shape[1]} variables")

# Define both targets
targets = {
    'GDP_Growth': 'G_GPD_PCAP_SLOPE',
    'Trading_Score': 'G_Score-Trading across borders (DB16-20 methodology)'
}

all_results = {}

for target_name, target_col in targets.items():
    print(f"\n{'='*80}")
    print(f"ANALYZING: {target_name} ({target_col})")
    print("="*80)

    # Check data availability
    if target_col not in df.columns:
        print(f"ERROR: Column '{target_col}' not found!")
        continue

    valid_mask = df[target_col].notna()
    print(f"Valid observations: {valid_mask.sum()}")

    if valid_mask.sum() < 20:
        print(f"WARNING: Insufficient data for {target_name}")
        continue

    # Get predictors (exclude all G variables, targets, and identifiers)
    g_vars = [c for c in df.columns if c.startswith('G_')]
    exclude_cols = ['Pais', 'Etiqueta_pais', 'Country code', 'Ease of Doing Business',
                    'New_Business_Density', 'Country'] + g_vars
    potential_predictors = [col for col in df.columns if col not in exclude_cols]

    # Calculate correlations
    correlations = []
    for var in potential_predictors:
        mask = valid_mask & df[var].notna()
        if mask.sum() >= 10:
            try:
                r, p = pearsonr(df.loc[mask, var], df.loc[mask, target_col])
                correlations.append({
                    'Variable': var,
                    'Correlation': abs(r),
                    'R_value': r,
                    'P_value': p
                })
            except:
                pass

    corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
    print(f"\nCorrelated variables found: {len(corr_df)}")

    # Select top 15 predictors
    top_predictors = corr_df.head(15)['Variable'].tolist()
    print(f"Using top 15 predictors")

    # Prepare data
    mask = df[target_col].notna()
    for var in top_predictors:
        mask &= df[var].notna()

    X = df.loc[mask, top_predictors].values
    y = df.loc[mask, target_col].values

    print(f"Final dataset: {X.shape[0]} observations, {X.shape[1]} features")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models
    models = {}

    print("\nTraining models...")

    # 1. Random Forest
    print("  1. Random Forest...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20,
                              min_samples_split=2, min_samples_leaf=1,
                              random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2').mean()

    models['Random Forest'] = {
        'model': rf,
        'y_pred': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'cv': cv_rf,
        'importance': dict(zip(top_predictors, rf.feature_importances_))
    }

    # 2. XGBoost
    print("  2. XGBoost...")
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                      random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    cv_xgb = cross_val_score(xgb, X_train, y_train, cv=5, scoring='r2').mean()

    models['XGBoost'] = {
        'model': xgb,
        'y_pred': y_pred_xgb,
        'r2': r2_score(y_test, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'mae': mean_absolute_error(y_test, y_pred_xgb),
        'cv': cv_xgb
    }

    # 3. Gradient Boosting
    print("  3. Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    cv_gb = cross_val_score(gb, X_train, y_train, cv=5, scoring='r2').mean()

    models['Gradient Boosting'] = {
        'model': gb,
        'y_pred': y_pred_gb,
        'r2': r2_score(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'cv': cv_gb
    }

    # 4. Neural Network
    print("  4. Neural Network...")
    nn = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu',
                     max_iter=1000, random_state=42, early_stopping=True)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    cv_nn = cross_val_score(nn, X_train, y_train, cv=5, scoring='r2').mean()

    models['Neural Network'] = {
        'model': nn,
        'y_pred': y_pred_nn,
        'r2': r2_score(y_test, y_pred_nn),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
        'mae': mean_absolute_error(y_test, y_pred_nn),
        'cv': cv_nn
    }

    # Rank models
    ranked = sorted(models.items(), key=lambda x: x[1]['r2'], reverse=True)
    top_3 = ranked[:3]
    best_name, best_res = ranked[0]

    print(f"\nModel Rankings:")
    for i, (name, res) in enumerate(ranked, 1):
        print(f"  {i}. {name:20s} R2={res['r2']:.4f} CV={res['cv']:.4f}")

    # Validation
    print(f"\nValidation analysis...")
    residuals = y_test - best_res['y_pred']
    shapiro_stat, shapiro_p = shapiro(residuals)

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

    print(f"  Shapiro-Wilk p-value: {shapiro_p:.4f}")
    print(f"  Bootstrap R2: {np.mean(bootstrap_scores):.4f} [{bootstrap_ci[0]:.4f}, {bootstrap_ci[1]:.4f}]")

    # Store results
    all_results[target_name] = {
        'target_column': target_col,
        'predictors': top_predictors,
        'best': (best_name, best_res),
        'top_3': top_3,
        'all_models': models,
        'validation': validation,
        'y_test': y_test,
        'X_test': X_test
    }

# Save results
with open('resultados/modelos/final_dual_target_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

for target_name, res in all_results.items():
    best_name, best_res = res['best']
    print(f"\n{target_name}:")
    print(f"  Target: {res['target_column']}")
    print(f"  Best Model: {best_name}")
    print(f"  R2 Test: {best_res['r2']:.4f}")
    print(f"  CV: {best_res['cv']:.4f}")
    print(f"  RMSE: {best_res['rmse']:.2f}")

print(f"\nResults saved: resultados/modelos/final_dual_target_results.pkl")
