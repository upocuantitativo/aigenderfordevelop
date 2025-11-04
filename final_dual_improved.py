#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPROVED DUAL TARGET ANALYSIS
Both GDP Growth and Tax Score predicted from SAME gender/development indicators
Using robust hyperparameter optimization and ensemble methods
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, shapiro
import warnings
warnings.filterwarnings('ignore')

def get_balanced_score(r2, cv):
    """Balanced score for simultaneous R² and CV optimization"""
    if r2 < 0 or cv < 0:
        return 0.0
    if r2 < 0.3:
        return r2 * 0.2
    return np.sqrt(max(0, r2)) * np.sqrt(cv)

print("="*80)
print("IMPROVED DUAL TARGET ANALYSIS")
print("GDP Growth + Tax Score from SAME gender predictors")
print("With hyperparameter optimization")
print("="*80)

# Load merged data
df = pd.read_excel('DATA_MERGED_NEW.xlsx')
print(f"\nMerged data loaded: {df.shape[0]} countries, {df.shape[1]} variables")

# Define both targets
target_gdp = 'G_GPD_PCAP_SLOPE'
target_tax = 'Score-Total tax and contribution rate (% of profit)'

# Check both targets exist and have overlap
valid_gdp = df[target_gdp].notna()
valid_tax = df[target_tax].notna()
valid_both = valid_gdp & valid_tax

print(f"\nGDP Growth valid: {valid_gdp.sum()}")
print(f"Tax Score valid: {valid_tax.sum()}")
print(f"Both valid (overlap): {valid_both.sum()}")

# Get gender/development predictors ONLY (original DATA_GHAB2 columns)
g_vars = [c for c in df.columns if c.startswith('G_')]
exclude_cols = (['Pais', 'Etiqueta_pais', 'Country code', 'Country code2',
                 'Ease of Doing Business', 'New_Business_Density', 'Country', 'Economy'] +
                g_vars)

# Get only original gender/development indicators (not Doing Business variables)
potential_predictors = [col for col in df.columns
                       if col not in exclude_cols
                       and not col.startswith('Score-')
                       and not col.startswith('Rank-')
                       and 'DB' not in col
                       and 'index' not in col.lower()
                       and 'methodology' not in col.lower()
                       and 'Time (days)' not in col
                       and 'Time (years)' not in col
                       and 'Time (hours' not in col
                       and 'Cost (' not in col
                       and 'Procedures' not in col]

print(f"\nGender/development predictors available: {len(potential_predictors)}")

all_results = {}

for target_name, target_col in [('GDP_Growth', target_gdp), ('Tax_Score', target_tax)]:
    print(f"\n{'='*80}")
    print(f"ANALYZING: {target_name}")
    print(f"Variable: {target_col}")
    print("="*80)

    valid_mask = df[target_col].notna()
    print(f"Valid observations: {valid_mask.sum()}")

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

    print("\nTop 5 predictors:")
    for i, var in enumerate(top_predictors[:5], 1):
        corr_val = corr_df[corr_df['Variable'] == var]['Correlation'].values[0]
        print(f"  {i}. {var[:60]} (r={corr_val:.3f})")

    # Prepare data
    mask = df[target_col].notna()
    for var in top_predictors:
        mask &= df[var].notna()

    X = df.loc[mask, top_predictors].values
    y = df.loc[mask, target_col].values

    print(f"\nFinal dataset: {X.shape[0]} observations, {X.shape[1]} features")

    if X.shape[0] < 20:
        print("WARNING: Insufficient data, skipping...")
        continue

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Train models with optimized hyperparameters
    models = {}

    print("\nTraining models with optimization...")

    # 1. Random Forest with GridSearch
    print("  1. Random Forest (optimizing)...")
    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_params, cv=min(3, len(X_train)//2),
                          scoring='r2', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train, y_train)
    rf_best = rf_grid.best_estimator_

    y_pred_rf = rf_best.predict(X_test)
    cv_rf = cross_val_score(rf_best, X_train, y_train,
                           cv=min(5, len(X_train)//2), scoring='r2').mean()

    models['Random Forest'] = {
        'model': rf_best,
        'y_pred': y_pred_rf,
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'cv': cv_rf,
        'balanced_score': get_balanced_score(r2_score(y_test, y_pred_rf), cv_rf),
        'importance': dict(zip(top_predictors, rf_best.feature_importances_)),
        'best_params': rf_grid.best_params_
    }
    print(f"     Best params: {rf_grid.best_params_}")
    print(f"     R²={r2_score(y_test, y_pred_rf):.3f}, CV={cv_rf:.3f}")

    # 2. XGBoost with GridSearch
    print("  2. XGBoost (optimizing)...")
    xgb_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    xgb_base = XGBRegressor(random_state=42, n_jobs=-1)
    xgb_grid = GridSearchCV(xgb_base, xgb_params, cv=min(3, len(X_train)//2),
                           scoring='r2', n_jobs=-1, verbose=0)
    xgb_grid.fit(X_train, y_train)
    xgb_best = xgb_grid.best_estimator_

    y_pred_xgb = xgb_best.predict(X_test)
    cv_xgb = cross_val_score(xgb_best, X_train, y_train,
                            cv=min(5, len(X_train)//2), scoring='r2').mean()

    models['XGBoost'] = {
        'model': xgb_best,
        'y_pred': y_pred_xgb,
        'r2': r2_score(y_test, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
        'mae': mean_absolute_error(y_test, y_pred_xgb),
        'cv': cv_xgb,
        'balanced_score': get_balanced_score(r2_score(y_test, y_pred_xgb), cv_xgb),
        'best_params': xgb_grid.best_params_
    }
    print(f"     Best params: {xgb_grid.best_params_}")
    print(f"     R²={r2_score(y_test, y_pred_xgb):.3f}, CV={cv_xgb:.3f}")

    # 3. Gradient Boosting
    print("  3. Gradient Boosting...")
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                  learning_rate=0.1, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    cv_gb = cross_val_score(gb, X_train, y_train,
                           cv=min(5, len(X_train)//2), scoring='r2').mean()

    models['Gradient Boosting'] = {
        'model': gb,
        'y_pred': y_pred_gb,
        'r2': r2_score(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'cv': cv_gb,
        'balanced_score': get_balanced_score(r2_score(y_test, y_pred_gb), cv_gb)
    }
    print(f"     R²={r2_score(y_test, y_pred_gb):.3f}, CV={cv_gb:.3f}")

    # 4. Neural Network
    print("  4. Neural Network...")
    nn = MLPRegressor(hidden_layer_sizes=(100, 50, 25), activation='relu',
                     max_iter=2000, random_state=42, early_stopping=True)
    nn.fit(X_train, y_train)
    y_pred_nn = nn.predict(X_test)
    cv_nn = cross_val_score(nn, X_train, y_train,
                           cv=min(5, len(X_train)//2), scoring='r2').mean()

    models['Neural Network'] = {
        'model': nn,
        'y_pred': y_pred_nn,
        'r2': r2_score(y_test, y_pred_nn),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_nn)),
        'mae': mean_absolute_error(y_test, y_pred_nn),
        'cv': cv_nn,
        'balanced_score': get_balanced_score(r2_score(y_test, y_pred_nn), cv_nn)
    }
    print(f"     R²={r2_score(y_test, y_pred_nn):.3f}, CV={cv_nn:.3f}")

    # Rank by BALANCED SCORE
    ranked = sorted(models.items(), key=lambda x: x[1]['balanced_score'], reverse=True)
    top_3 = ranked[:3]
    best_name, best_res = ranked[0]

    print(f"\nModel Rankings (by balanced score):")
    for i, (name, res) in enumerate(ranked, 1):
        cv_status = "POSITIVE" if res['cv'] > 0 else "negative"
        print(f"  {i}. {name:20s} R2={res['r2']:.4f} CV={res['cv']:.4f} ({cv_status}) Balanced={res['balanced_score']:.4f}")

    # Validation
    print(f"\nValidation analysis for {best_name}...")
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
with open('resultados/modelos/final_dual_improved.pkl', 'wb') as f:
    pickle.dump(all_results, f)

print("\n" + "="*80)
print("SUMMARY (IMPROVED DUAL ANALYSIS)")
print("="*80)

for target_name, res in all_results.items():
    best_name, best_res = res['best']
    cv_status = "POSITIVE (good!)" if best_res['cv'] > 0 else "negative (caution)"

    print(f"\n{target_name}:")
    print(f"  Target: {res['target_column'][:50]}...")
    print(f"  Best Model: {best_name}")
    print(f"  R2 Test: {best_res['r2']:.4f}")
    print(f"  CV: {best_res['cv']:.4f} - {cv_status}")
    print(f"  Balanced Score: {best_res['balanced_score']:.4f}")
    print(f"  RMSE: {best_res['rmse']:.2f}")
    print(f"  Bootstrap CI: [{res['validation']['bootstrap_ci'][0]:.3f}, {res['validation']['bootstrap_ci'][1]:.3f}]")

print(f"\nResults saved: resultados/modelos/final_dual_improved.pkl")
print("\nNote: Both targets predicted from SAME gender/development indicators")
print("Hyperparameter optimization applied to Random Forest and XGBoost")
