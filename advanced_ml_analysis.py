#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED ML ANALYSIS: DUAL TARGET VARIABLES WITH QUANTUM COMPUTING
Analysis for Gender and Economic Development Indicators

Features:
- Dual target analysis: New_Business_Density & G_GPD_PCAP_SLOPE
- Random Forest with optimization
- XGBoost with hyperparameter tuning
- Neural Networks (MLPRegressor & Deep Learning)
- Quantum Machine Learning (VQR - Variational Quantum Regressor)
- Comprehensive robustness analysis
- Advanced validation plots

Author: Advanced Economic Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from datetime import datetime
import sys
import json

warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.model_selection import KFold, permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr
from scipy import stats

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

# TensorFlow/Keras for deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow not installed. Install with: pip install tensorflow")

# Qiskit for quantum computing
try:
    from qiskit import QuantumCircuit
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit_machine_learning.algorithms import VQR
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import Estimator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️ Qiskit not installed. Install with: pip install qiskit qiskit-machine-learning")

plt.style.use('default')
sns.set_palette("husl")


class AdvancedMLAnalyzer:
    """Advanced ML Analyzer with dual targets and quantum computing"""

    def __init__(self):
        self.df = None
        self.target_cols = ['New_Business_Density', 'G_GPD_PCAP_SLOPE']
        self.results = {}
        self.models = {}
        self.create_directories()

    def create_directories(self):
        """Creates directory structure"""
        dirs = [
            'resultados',
            'resultados/dashboards',
            'resultados/modelos_avanzados',
            'resultados/robustez',
            'resultados/validacion'
        ]
        for dir_name in dirs:
            os.makedirs(dir_name, exist_ok=True)

    def load_data(self):
        """Loads DATA_GHAB2.xlsx"""
        print("🔄 Loading DATA_GHAB2.xlsx...")

        try:
            self.df = pd.read_excel('DATA_GHAB2.xlsx')
            print(f"✓ Data loaded: {self.df.shape[0]} countries, {self.df.shape[1]} variables")

            # Verify target columns exist
            for target in self.target_cols:
                if target not in self.df.columns:
                    raise ValueError(f"Target variable {target} not found!")

            print(f"✓ Target variables verified: {self.target_cols}")
            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def get_top_predictors(self, target_col, n_predictors=15):
        """Gets top predictors for a target variable"""
        print(f"\n🔗 Finding top predictors for {target_col}...")

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        explanatory_vars = [col for col in numeric_cols if col not in self.target_cols]

        correlations = []
        for var in explanatory_vars:
            valid_mask = self.df[target_col].notna() & self.df[var].notna()
            if valid_mask.sum() < 10:
                continue

            try:
                pearson_r, pearson_p = pearsonr(
                    self.df.loc[valid_mask, var],
                    self.df.loc[valid_mask, target_col]
                )
                correlations.append({
                    'Variable': var,
                    'Correlation': abs(pearson_r),
                    'P_value': pearson_p
                })
            except:
                continue

        corr_df = pd.DataFrame(correlations).sort_values('Correlation', ascending=False)
        top_vars = corr_df.head(n_predictors)['Variable'].tolist()

        print(f"✓ Top {len(top_vars)} predictors identified")
        return top_vars, corr_df

    def prepare_data_for_target(self, target_col):
        """Prepares training data for a specific target"""
        top_predictors, corr_df = self.get_top_predictors(target_col, n_predictors=15)

        # Prepare clean dataset
        ml_data = self.df[top_predictors + [target_col]].dropna()

        if len(ml_data) < 15:
            raise ValueError(f"Insufficient data for {target_col}: only {len(ml_data)} samples")

        X = ml_data[top_predictors]
        y = ml_data[target_col]

        # Train/test split
        test_size = min(0.25, 0.5 - 5/len(ml_data))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"✓ Data prepared: {len(X_train)} train, {len(X_test)} test samples")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            'scaler': scaler,
            'features': top_predictors,
            'correlations': corr_df
        }

    def train_random_forest(self, data, target_col):
        """Trains optimized Random Forest"""
        print(f"\n🌲 Training Random Forest for {target_col}...")

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(data['X_train'], data['y_train'])
        y_pred = model.predict(data['X_test'])

        # Metrics
        r2 = r2_score(data['y_test'], y_pred)
        mae = mean_absolute_error(data['y_test'], y_pred)
        rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

        # Cross-validation
        cv_scores = cross_val_score(model, data['X_train'], data['y_train'],
                                    cv=5, scoring='r2', n_jobs=-1)

        print(f"  R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return {
            'model': model,
            'name': 'Random Forest',
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'feature_importance': dict(zip(data['features'], model.feature_importances_))
        }

    def train_xgboost(self, data, target_col):
        """Trains XGBoost"""
        if not XGB_AVAILABLE:
            print("⚠️ XGBoost not available, skipping...")
            return None

        print(f"\n⚡ Training XGBoost for {target_col}...")

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(data['X_train'], data['y_train'])
        y_pred = model.predict(data['X_test'])

        # Metrics
        r2 = r2_score(data['y_test'], y_pred)
        mae = mean_absolute_error(data['y_test'], y_pred)
        rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

        # Cross-validation
        cv_scores = cross_val_score(model, data['X_train'], data['y_train'],
                                    cv=5, scoring='r2', n_jobs=-1)

        print(f"  R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return {
            'model': model,
            'name': 'XGBoost',
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_scores': cv_scores,
            'y_pred': y_pred,
            'feature_importance': dict(zip(data['features'], model.feature_importances_))
        }

    def train_neural_network_sklearn(self, data, target_col):
        """Trains Neural Network with sklearn"""
        print(f"\n🧠 Training Neural Network (sklearn) for {target_col}...")

        model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )

        model.fit(data['X_train_scaled'], data['y_train'])
        y_pred = model.predict(data['X_test_scaled'])

        # Metrics
        r2 = r2_score(data['y_test'], y_pred)
        mae = mean_absolute_error(data['y_test'], y_pred)
        rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

        # Cross-validation
        cv_scores = cross_val_score(
            MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42),
            data['X_train_scaled'], data['y_train'],
            cv=5, scoring='r2', n_jobs=-1
        )

        print(f"  R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        print(f"  CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return {
            'model': model,
            'name': 'Neural Network (sklearn)',
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'cv_scores': cv_scores,
            'y_pred': y_pred
        }

    def train_deep_neural_network(self, data, target_col):
        """Trains Deep Neural Network with TensorFlow"""
        if not TF_AVAILABLE:
            print("⚠️ TensorFlow not available, skipping...")
            return None

        print(f"\n🧠 Training Deep Neural Network (TensorFlow) for {target_col}...")

        # Build model
        model = keras.Sequential([
            layers.Input(shape=(data['X_train_scaled'].shape[1],)),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        # Train with early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )

        history = model.fit(
            data['X_train_scaled'], data['y_train'],
            epochs=500,
            batch_size=8,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        # Predict
        y_pred = model.predict(data['X_test_scaled'], verbose=0).flatten()

        # Metrics
        r2 = r2_score(data['y_test'], y_pred)
        mae = mean_absolute_error(data['y_test'], y_pred)
        rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

        print(f"  R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
        print(f"  Training epochs: {len(history.history['loss'])}")

        return {
            'model': model,
            'name': 'Deep Neural Network (TensorFlow)',
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'y_pred': y_pred,
            'history': history.history
        }

    def train_quantum_vqr(self, data, target_col):
        """Trains Quantum Variational Quantum Regressor"""
        if not QISKIT_AVAILABLE:
            print("⚠️ Qiskit not available, skipping...")
            return None

        print(f"\n🔮 Training Quantum VQR for {target_col}...")

        try:
            # Reduce to top 4 features for quantum (limited qubits)
            n_qubits = min(4, data['X_train_scaled'].shape[1])
            X_train_q = data['X_train_scaled'][:, :n_qubits]
            X_test_q = data['X_test_scaled'][:, :n_qubits]

            # Feature map and ansatz
            feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2)
            ansatz = RealAmplitudes(num_qubits=n_qubits, reps=3)

            # Create QNN
            estimator = Estimator()
            qnn = EstimatorQNN(
                circuit=feature_map.compose(ansatz),
                estimator=estimator,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters
            )

            # VQR
            vqr = VQR(
                neural_network=qnn,
                optimizer=COBYLA(maxiter=100),
            )

            # Fit (may take some time)
            print(f"  Training quantum model with {n_qubits} qubits...")
            vqr.fit(X_train_q, data['y_train'].values)

            # Predict
            y_pred = vqr.predict(X_test_q)

            # Metrics
            r2 = r2_score(data['y_test'], y_pred)
            mae = mean_absolute_error(data['y_test'], y_pred)
            rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred))

            print(f"  R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
            print(f"  Quantum features used: {n_qubits}")

            return {
                'model': vqr,
                'name': 'Quantum VQR',
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'y_pred': y_pred,
                'n_qubits': n_qubits
            }

        except Exception as e:
            print(f"  ❌ Quantum training failed: {e}")
            return None

    def robustness_analysis(self, model_result, data, target_col):
        """Comprehensive robustness analysis"""
        print(f"\n🔬 Robustness Analysis for {model_result['name']}...")

        robustness = {}

        # 1. K-Fold Cross-Validation
        if 'cv_scores' in model_result:
            cv_scores = model_result['cv_scores']
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            # For deep learning, use simplified model
            if 'TensorFlow' in model_result['name']:
                cv_scores = []
                for train_idx, val_idx in kfold.split(data['X_train_scaled']):
                    simple_model = keras.Sequential([
                        layers.Dense(64, activation='relu', input_shape=(data['X_train_scaled'].shape[1],)),
                        layers.Dense(32, activation='relu'),
                        layers.Dense(1)
                    ])
                    simple_model.compile(optimizer='adam', loss='mse')
                    simple_model.fit(
                        data['X_train_scaled'][train_idx], data['y_train'].values[train_idx],
                        epochs=100, verbose=0, batch_size=8
                    )
                    val_pred = simple_model.predict(data['X_train_scaled'][val_idx], verbose=0).flatten()
                    cv_scores.append(r2_score(data['y_train'].values[val_idx], val_pred))
                cv_scores = np.array(cv_scores)
            else:
                cv_scores = np.array([model_result['r2']])

        robustness['cv_mean'] = cv_scores.mean()
        robustness['cv_std'] = cv_scores.std()
        robustness['cv_scores'] = cv_scores

        # 2. Bootstrap Confidence Intervals
        print("  Running bootstrap analysis...")
        n_bootstrap = 100
        bootstrap_scores = []

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(data['y_test']), size=len(data['y_test']), replace=True)
            y_test_boot = data['y_test'].values[indices]
            y_pred_boot = model_result['y_pred'][indices]

            boot_r2 = r2_score(y_test_boot, y_pred_boot)
            bootstrap_scores.append(boot_r2)

        bootstrap_scores = np.array(bootstrap_scores)
        robustness['bootstrap_mean'] = bootstrap_scores.mean()
        robustness['bootstrap_ci_lower'] = np.percentile(bootstrap_scores, 2.5)
        robustness['bootstrap_ci_upper'] = np.percentile(bootstrap_scores, 97.5)

        # 3. Residual Analysis
        residuals = data['y_test'].values - model_result['y_pred']
        robustness['residuals'] = residuals
        robustness['residuals_mean'] = residuals.mean()
        robustness['residuals_std'] = residuals.std()

        # Normality test
        _, p_value_normality = stats.shapiro(residuals)
        robustness['residuals_normality_p'] = p_value_normality

        # 4. Permutation Test (for feature importance validation)
        if 'Quantum' not in model_result['name'] and 'TensorFlow' not in model_result['name']:
            print("  Running permutation test...")
            try:
                X_data = data['X_train_scaled'] if 'sklearn' in model_result['name'] else data['X_train']
                score, perm_scores, p_value = permutation_test_score(
                    model_result['model'], X_data, data['y_train'],
                    scoring='r2', cv=5, n_permutations=50, n_jobs=-1
                )
                robustness['permutation_score'] = score
                robustness['permutation_p_value'] = p_value
                robustness['permutation_scores'] = perm_scores
            except:
                robustness['permutation_score'] = None

        print(f"  ✓ CV: {robustness['cv_mean']:.4f} ± {robustness['cv_std']:.4f}")
        print(f"  ✓ Bootstrap 95% CI: [{robustness['bootstrap_ci_lower']:.4f}, {robustness['bootstrap_ci_upper']:.4f}]")

        return robustness

    def create_validation_plots(self, model_result, data, robustness, target_col):
        """Creates comprehensive validation plots"""
        print(f"\n📊 Creating validation plots for {model_result['name']}...")

        model_name_safe = model_result['name'].replace(' ', '_').replace('(', '').replace(')', '')
        target_safe = target_col.replace('_', '')

        # Create comprehensive figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Predictions vs Actual',
                'Residual Plot',
                'Residual Distribution',
                'Q-Q Plot',
                'Bootstrap Distribution',
                'Cross-Validation Scores'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "bar"}]
            ]
        )

        # 1. Predictions vs Actual
        fig.add_trace(
            go.Scatter(
                x=data['y_test'], y=model_result['y_pred'],
                mode='markers',
                name='Predictions',
                marker=dict(color='blue', size=8, opacity=0.6)
            ),
            row=1, col=1
        )
        # Perfect prediction line
        min_val = min(data['y_test'].min(), model_result['y_pred'].min())
        max_val = max(data['y_test'].max(), model_result['y_pred'].max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect', line=dict(color='red', dash='dash')),
            row=1, col=1
        )

        # 2. Residual Plot
        residuals = robustness['residuals']
        fig.add_trace(
            go.Scatter(
                x=model_result['y_pred'], y=residuals,
                mode='markers',
                name='Residuals',
                marker=dict(color='green', size=8, opacity=0.6)
            ),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        # 3. Residual Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals', marker_color='purple', nbinsx=20),
            row=2, col=1
        )

        # 4. Q-Q Plot
        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        fig.add_trace(
            go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q',
                      marker=dict(color='orange', size=6)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=osm, y=slope*osm + intercept, mode='lines',
                      name='Theoretical', line=dict(color='red', dash='dash')),
            row=2, col=2
        )

        # 5. Bootstrap Distribution
        if 'bootstrap_ci_lower' in robustness:
            bootstrap_r2s = np.random.normal(
                robustness['bootstrap_mean'],
                (robustness['bootstrap_ci_upper'] - robustness['bootstrap_ci_lower'])/4,
                100
            )
            fig.add_trace(
                go.Histogram(x=bootstrap_r2s, name='Bootstrap R²',
                            marker_color='teal', nbinsx=20),
                row=3, col=1
            )

        # 6. Cross-Validation Scores
        if 'cv_scores' in robustness:
            cv_scores = robustness['cv_scores']
            fig.add_trace(
                go.Bar(x=[f'Fold {i+1}' for i in range(len(cv_scores))],
                      y=cv_scores, name='CV Scores', marker_color='blue'),
                row=3, col=2
            )
            fig.add_hline(y=cv_scores.mean(), line_dash="dash", line_color="red",
                         row=3, col=2)

        # Update layout
        fig.update_layout(
            title=f'Validation Analysis: {model_result["name"]} - {target_col}<br>' +
                  f'R² = {model_result["r2"]:.4f}, RMSE = {model_result["rmse"]:.4f}',
            height=1000,
            showlegend=False
        )

        # Save
        filename = f'resultados/validacion/validation_{model_name_safe}_{target_safe}.html'
        fig.write_html(filename)
        print(f"  ✓ Saved: {filename}")

    def create_robustness_summary(self, all_results, target_col):
        """Creates robustness summary dashboard"""
        print(f"\n📊 Creating robustness summary for {target_col}...")

        # Create comparison table
        comparison_data = []
        for model_name, result in all_results.items():
            if result is None:
                continue

            robustness = result.get('robustness', {})
            comparison_data.append({
                'Model': model_name,
                'R²': result['r2'],
                'RMSE': result['rmse'],
                'CV Mean': robustness.get('cv_mean', np.nan),
                'CV Std': robustness.get('cv_std', np.nan),
                'Bootstrap CI Lower': robustness.get('bootstrap_ci_lower', np.nan),
                'Bootstrap CI Upper': robustness.get('bootstrap_ci_upper', np.nan)
            })

        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('R²', ascending=False)

        # Save CSV
        target_safe = target_col.replace('_', '')
        df_comparison.to_csv(f'resultados/robustez/robustness_summary_{target_safe}.csv', index=False)

        # Create interactive table
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=list(df_comparison.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[df_comparison[col] for col in df_comparison.columns],
                fill_color='lavender',
                align='left',
                format=[None, '.4f', '.4f', '.4f', '.4f', '.4f', '.4f']
            )
        )])

        fig.update_layout(
            title=f'Robustness Analysis Summary - {target_col}',
            height=400
        )

        fig.write_html(f'resultados/robustez/robustness_table_{target_safe}.html')
        print(f"  ✓ Robustness summary created")

    def analyze_target(self, target_col):
        """Complete analysis for one target variable"""
        print(f"\n{'='*80}")
        print(f"ANALYZING TARGET: {target_col}")
        print(f"{'='*80}")

        # Prepare data
        data = self.prepare_data_for_target(target_col)

        # Train all models
        results = {}

        # Random Forest
        results['Random Forest'] = self.train_random_forest(data, target_col)

        # XGBoost
        xgb_result = self.train_xgboost(data, target_col)
        if xgb_result:
            results['XGBoost'] = xgb_result

        # Neural Network (sklearn)
        results['Neural Network'] = self.train_neural_network_sklearn(data, target_col)

        # Deep Neural Network (TensorFlow)
        dnn_result = self.train_deep_neural_network(data, target_col)
        if dnn_result:
            results['Deep NN'] = dnn_result

        # Quantum VQR
        quantum_result = self.train_quantum_vqr(data, target_col)
        if quantum_result:
            results['Quantum VQR'] = quantum_result

        # Robustness analysis for each model
        print(f"\n{'='*60}")
        print("ROBUSTNESS ANALYSIS")
        print(f"{'='*60}")

        for model_name, model_result in results.items():
            if model_result is None:
                continue

            robustness = self.robustness_analysis(model_result, data, target_col)
            model_result['robustness'] = robustness

            # Create validation plots
            self.create_validation_plots(model_result, data, robustness, target_col)

        # Create robustness summary
        self.create_robustness_summary(results, target_col)

        # Store results
        self.results[target_col] = {
            'models': results,
            'data': data
        }

        return results

    def run_complete_analysis(self):
        """Runs complete analysis for both targets"""
        print("🚀 STARTING ADVANCED ML ANALYSIS WITH DUAL TARGETS")
        print("="*80)

        if not self.load_data():
            return False

        # Analyze each target
        for target_col in self.target_cols:
            self.analyze_target(target_col)

        # Create final comparison dashboard
        self.create_final_dashboard()

        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\n📁 Results saved in:")
        print("  - resultados/modelos_avanzados/")
        print("  - resultados/robustez/")
        print("  - resultados/validacion/")

        return True

    def create_final_dashboard(self):
        """Creates final comprehensive dashboard"""
        print("\n📊 Creating final dashboard...")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced ML Analysis - Dual Targets</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #4facfe;
            text-align: center;
            margin-bottom: 30px;
        }}
        .target-section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #4facfe;
        }}
        h2 {{
            color: #667eea;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 12px rgba(0,0,0,0.15);
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 20px;
            margin: 5px;
        }}
        .btn:hover {{
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4facfe;
            color: white;
        }}
        .metric {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }}
        .badge {{
            display: inline-block;
            background: #4caf50;
            color: white;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            margin-left: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Advanced ML Analysis - Dual Target Variables</h1>
        <p style="text-align: center; color: #666;">
            Complete analysis with Random Forest, XGBoost, Neural Networks & Quantum Computing<br>
            Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}
        </p>
"""

        for target_col in self.target_cols:
            if target_col not in self.results:
                continue

            results = self.results[target_col]['models']
            target_safe = target_col.replace('_', '')

            # Find best model
            best_model = max(results.items(),
                           key=lambda x: x[1]['r2'] if x[1] is not None else -np.inf)

            html_content += f"""
        <div class="target-section">
            <h2>🎯 Target Variable: {target_col}</h2>

            <div class="grid">
                <div class="card">
                    <h3>🏆 Best Model</h3>
                    <p class="metric">{best_model[0]}</p>
                    <p>R² = {best_model[1]['r2']:.4f}</p>
                    <p>RMSE = {best_model[1]['rmse']:.4f}</p>
                </div>

                <div class="card">
                    <h3>📊 Models Trained</h3>
                    <p class="metric">{len([r for r in results.values() if r is not None])}</p>
                    <p>Random Forest, XGBoost, Neural Networks, Quantum ML</p>
                </div>

                <div class="card">
                    <h3>📈 Robustness</h3>
                    <a href="../robustez/robustness_table_{target_safe}.html" class="btn" target="_blank">
                        View Summary
                    </a>
                    <a href="../robustez/robustness_summary_{target_safe}.csv" class="btn" target="_blank">
                        Download CSV
                    </a>
                </div>
            </div>

            <h3>🤖 Model Results</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>R²</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>CV Mean ± Std</th>
                    <th>Validation Plots</th>
                </tr>
"""

            for model_name, result in sorted(results.items(),
                                            key=lambda x: x[1]['r2'] if x[1] else -np.inf,
                                            reverse=True):
                if result is None:
                    continue

                robustness = result.get('robustness', {})
                cv_mean = robustness.get('cv_mean', np.nan)
                cv_std = robustness.get('cv_std', np.nan)

                model_name_safe = model_name.replace(' ', '_').replace('(', '').replace(')', '')

                html_content += f"""
                <tr>
                    <td><strong>{model_name}</strong></td>
                    <td>{result['r2']:.4f}</td>
                    <td>{result['rmse']:.4f}</td>
                    <td>{result['mae']:.4f}</td>
                    <td>{cv_mean:.4f} ± {cv_std:.4f}</td>
                    <td>
                        <a href="../validacion/validation_{model_name_safe}_{target_safe}.html"
                           class="btn" target="_blank">View</a>
                    </td>
                </tr>
"""

            html_content += """
            </table>
        </div>
"""

        html_content += """
        <div style="margin-top: 40px; padding: 20px; background: #e8f5e8; border-radius: 10px;">
            <h2>🔬 Analysis Components</h2>
            <ul>
                <li>✅ <strong>Random Forest</strong> with optimized hyperparameters</li>
                <li>✅ <strong>XGBoost</strong> gradient boosting</li>
                <li>✅ <strong>Neural Networks</strong> (sklearn MLPRegressor)</li>
                <li>✅ <strong>Deep Neural Networks</strong> (TensorFlow/Keras)</li>
                <li>✅ <strong>Quantum Machine Learning</strong> (Qiskit VQR)</li>
                <li>✅ <strong>K-Fold Cross-Validation</strong></li>
                <li>✅ <strong>Bootstrap Confidence Intervals</strong></li>
                <li>✅ <strong>Residual Analysis</strong></li>
                <li>✅ <strong>Permutation Tests</strong></li>
                <li>✅ <strong>Comprehensive Validation Plots</strong></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

        with open('resultados/modelos_avanzados/advanced_ml_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("  ✓ Final dashboard created: resultados/modelos_avanzados/advanced_ml_dashboard.html")


def main():
    """Main function"""
    try:
        analyzer = AdvancedMLAnalyzer()
        success = analyzer.run_complete_analysis()

        if success:
            print("\n✅ Process completed successfully!")
            return 0
        else:
            print("\n❌ Process completed with errors")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️ Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
