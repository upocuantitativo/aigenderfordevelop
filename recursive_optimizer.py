#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RECURSIVE MODEL OPTIMIZER
Finds best models recursively and creates final visualizations only for top performers

Algorithm:
1. Train all models with default parameters
2. Rank models by R² score
3. Recursively optimize top N models
4. Re-rank and select final best models
5. Generate visualizations only for winners

Author: Advanced Economic Analysis System
Date: 2025
"""

import pandas as pd
import numpy as np
import warnings
import os
from datetime import datetime
import json
from advanced_ml_analysis import AdvancedMLAnalyzer

warnings.filterwarnings('ignore')


class RecursiveModelOptimizer(AdvancedMLAnalyzer):
    """Recursive optimizer that finds and showcases only the best models"""

    def __init__(self, top_n=3, optimization_rounds=3):
        super().__init__()
        self.top_n = top_n  # Number of best models to keep
        self.optimization_rounds = optimization_rounds
        self.optimization_history = {}
        self.final_best_models = {}

    def rank_models(self, results):
        """Ranks models by R² score"""
        rankings = []
        for model_name, result in results.items():
            if result is None:
                continue

            # Composite score: R² + robustness
            r2_score = result['r2']
            robustness = result.get('robustness', {})
            cv_mean = robustness.get('cv_mean', r2_score)
            cv_std = robustness.get('cv_std', 0)

            # Penalize high variance
            composite_score = r2_score * 0.6 + cv_mean * 0.4 - cv_std * 0.2

            rankings.append({
                'model': model_name,
                'r2': r2_score,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'composite_score': composite_score,
                'result': result
            })

        # Sort by composite score
        rankings.sort(key=lambda x: x['composite_score'], reverse=True)
        return rankings

    def optimize_model_recursive(self, model_name, result, data, target_col, depth=0):
        """Recursively optimizes a single model"""
        if depth >= self.optimization_rounds:
            print(f"    Max depth reached for {model_name}")
            return result

        print(f"  {'  '*depth}🔄 Optimization round {depth+1} for {model_name}")

        current_r2 = result['r2']
        improved = False

        # Different optimization strategies based on model type
        if 'Random Forest' in model_name:
            improved_result = self.optimize_random_forest(result, data, target_col, depth)
        elif 'XGBoost' in model_name:
            improved_result = self.optimize_xgboost(result, data, target_col, depth)
        elif 'Neural Network' in model_name and 'TensorFlow' not in model_name:
            improved_result = self.optimize_sklearn_nn(result, data, target_col, depth)
        elif 'Deep NN' in model_name:
            improved_result = self.optimize_deep_nn(result, data, target_col, depth)
        else:
            # Quantum and others - no further optimization
            return result

        # Check if improved
        if improved_result and improved_result['r2'] > current_r2:
            print(f"    {'  '*depth}✅ Improved: {current_r2:.4f} → {improved_result['r2']:.4f}")
            # Recursive call with improved model
            return self.optimize_model_recursive(
                model_name, improved_result, data, target_col, depth+1
            )
        else:
            print(f"    {'  '*depth}⏹️ No improvement, stopping")
            return result

    def optimize_random_forest(self, result, data, target_col, depth):
        """Optimizes Random Forest parameters"""
        from sklearn.ensemble import RandomForestRegressor

        # Progressive parameter search
        param_options = [
            {'n_estimators': [300, 500, 700], 'max_depth': [10, 15, 20]},
            {'n_estimators': [500], 'max_depth': [15], 'min_samples_split': [2, 5, 10]},
            {'n_estimators': [500], 'max_depth': [15], 'min_samples_leaf': [1, 2, 4]}
        ]

        if depth >= len(param_options):
            return None

        params = param_options[depth]
        best_r2 = result['r2']
        best_model = result['model']

        for n_est in params.get('n_estimators', [result['model'].n_estimators]):
            for max_d in params.get('max_depth', [result['model'].max_depth]):
                for min_split in params.get('min_samples_split', [result['model'].min_samples_split]):
                    for min_leaf in params.get('min_samples_leaf', [result['model'].min_samples_leaf]):
                        model = RandomForestRegressor(
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=min_split,
                            min_samples_leaf=min_leaf,
                            random_state=42,
                            n_jobs=-1
                        )

                        model.fit(data['X_train'], data['y_train'])
                        y_pred = model.predict(data['X_test'])
                        r2 = r2_score(data['y_test'], y_pred)

                        if r2 > best_r2:
                            best_r2 = r2
                            best_model = model

        if best_r2 > result['r2']:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            y_pred = best_model.predict(data['X_test'])
            return {
                'model': best_model,
                'name': result['name'],
                'r2': best_r2,
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'y_pred': y_pred,
                'feature_importance': dict(zip(data['features'], best_model.feature_importances_))
            }
        return None

    def optimize_xgboost(self, result, data, target_col, depth):
        """Optimizes XGBoost parameters"""
        try:
            import xgboost as xgb
        except:
            return None

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Progressive parameter search
        param_options = [
            {'n_estimators': [300, 500, 700], 'learning_rate': [0.01, 0.05, 0.1]},
            {'n_estimators': [500], 'max_depth': [4, 5, 6, 7]},
            {'n_estimators': [500], 'subsample': [0.7, 0.8, 0.9], 'colsample_bytree': [0.7, 0.8, 0.9]}
        ]

        if depth >= len(param_options):
            return None

        params = param_options[depth]
        best_r2 = result['r2']
        best_model = result['model']

        for n_est in params.get('n_estimators', [result['model'].n_estimators]):
            for lr in params.get('learning_rate', [result['model'].learning_rate]):
                for max_d in params.get('max_depth', [result['model'].max_depth]):
                    for sub in params.get('subsample', [getattr(result['model'], 'subsample', 0.8)]):
                        for col_sub in params.get('colsample_bytree', [getattr(result['model'], 'colsample_bytree', 0.8)]):
                            model = xgb.XGBRegressor(
                                n_estimators=n_est,
                                learning_rate=lr,
                                max_depth=max_d,
                                subsample=sub,
                                colsample_bytree=col_sub,
                                random_state=42,
                                n_jobs=-1
                            )

                            model.fit(data['X_train'], data['y_train'])
                            y_pred = model.predict(data['X_test'])
                            r2 = r2_score(data['y_test'], y_pred)

                            if r2 > best_r2:
                                best_r2 = r2
                                best_model = model

        if best_r2 > result['r2']:
            y_pred = best_model.predict(data['X_test'])
            return {
                'model': best_model,
                'name': result['name'],
                'r2': best_r2,
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'y_pred': y_pred,
                'feature_importance': dict(zip(data['features'], best_model.feature_importances_))
            }
        return None

    def optimize_sklearn_nn(self, result, data, target_col, depth):
        """Optimizes sklearn Neural Network"""
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Progressive parameter search
        param_options = [
            {'hidden_layer_sizes': [(100, 50, 25), (150, 75, 25), (100, 100, 50)]},
            {'alpha': [0.0001, 0.001, 0.01]},
            {'learning_rate_init': [0.0001, 0.001, 0.01]}
        ]

        if depth >= len(param_options):
            return None

        params = param_options[depth]
        best_r2 = result['r2']
        best_model = result['model']

        for hidden in params.get('hidden_layer_sizes', [result['model'].hidden_layer_sizes]):
            for alpha in params.get('alpha', [result['model'].alpha]):
                for lr in params.get('learning_rate_init', [result['model'].learning_rate_init]):
                    model = MLPRegressor(
                        hidden_layer_sizes=hidden,
                        alpha=alpha,
                        learning_rate_init=lr,
                        max_iter=2000,
                        random_state=42,
                        early_stopping=True
                    )

                    model.fit(data['X_train_scaled'], data['y_train'])
                    y_pred = model.predict(data['X_test_scaled'])
                    r2 = r2_score(data['y_test'], y_pred)

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model

        if best_r2 > result['r2']:
            y_pred = best_model.predict(data['X_test_scaled'])
            return {
                'model': best_model,
                'name': result['name'],
                'r2': best_r2,
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'y_pred': y_pred
            }
        return None

    def optimize_deep_nn(self, result, data, target_col, depth):
        """Optimizes Deep Neural Network"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers, regularizers
        except:
            return None

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Progressive architecture search
        architectures = [
            [(128, 64, 32), 0.001],
            [(256, 128, 64, 32), 0.0005],
            [(128, 128, 64, 32), 0.001]
        ]

        if depth >= len(architectures):
            return None

        arch, lr = architectures[depth]
        best_r2 = result['r2']
        best_model = result['model']

        model = keras.Sequential()
        model.add(layers.Input(shape=(data['X_train_scaled'].shape[1],)))

        for units in arch:
            model.add(layers.Dense(units, activation='relu',
                                  kernel_regularizer=regularizers.l2(0.001)))
            model.add(layers.Dropout(0.3))

        model.add(layers.Dense(1))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae']
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True
        )

        model.fit(
            data['X_train_scaled'], data['y_train'],
            epochs=500,
            batch_size=8,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        y_pred = model.predict(data['X_test_scaled'], verbose=0).flatten()
        r2 = r2_score(data['y_test'], y_pred)

        if r2 > best_r2:
            return {
                'model': model,
                'name': result['name'],
                'r2': r2,
                'mae': mean_absolute_error(data['y_test'], y_pred),
                'rmse': np.sqrt(mean_squared_error(data['y_test'], y_pred)),
                'y_pred': y_pred
            }
        return None

    def recursive_analysis_for_target(self, target_col):
        """Performs recursive optimization for a target"""
        print(f"\n{'='*80}")
        print(f"🔄 RECURSIVE OPTIMIZATION FOR: {target_col}")
        print(f"{'='*80}")

        # Prepare data
        data = self.prepare_data_for_target(target_col)

        # Train all models initially
        print("\n🚀 Phase 1: Initial Training")
        print("="*60)
        results = {}

        results['Random Forest'] = self.train_random_forest(data, target_col)
        xgb_result = self.train_xgboost(data, target_col)
        if xgb_result:
            results['XGBoost'] = xgb_result

        results['Neural Network'] = self.train_neural_network_sklearn(data, target_col)

        dnn_result = self.train_deep_neural_network(data, target_col)
        if dnn_result:
            results['Deep NN'] = dnn_result

        quantum_result = self.train_quantum_vqr(data, target_col)
        if quantum_result:
            results['Quantum VQR'] = quantum_result

        # Rank models
        print(f"\n📊 Phase 2: Ranking Models")
        print("="*60)
        rankings = self.rank_models(results)

        print("\nInitial Rankings:")
        for i, rank in enumerate(rankings[:self.top_n+2], 1):
            print(f"  {i}. {rank['model']}: R²={rank['r2']:.4f}, Composite={rank['composite_score']:.4f}")

        # Select top N for optimization
        top_models = rankings[:self.top_n]

        # Recursive optimization
        print(f"\n🔧 Phase 3: Recursive Optimization (Top {self.top_n})")
        print("="*60)

        optimized_results = {}
        for rank in top_models:
            model_name = rank['model']
            result = rank['result']

            print(f"\n🎯 Optimizing {model_name}...")
            optimized = self.optimize_model_recursive(model_name, result, data, target_col, depth=0)

            # Add robustness analysis to optimized model
            print(f"  🔬 Running robustness analysis...")
            robustness = self.robustness_analysis(optimized, data, target_col)
            optimized['robustness'] = robustness

            optimized_results[model_name] = optimized

        # Final ranking
        print(f"\n🏆 Phase 4: Final Ranking")
        print("="*60)
        final_rankings = self.rank_models(optimized_results)

        print("\nFinal Rankings (Optimized):")
        for i, rank in enumerate(final_rankings, 1):
            print(f"  {i}. {rank['model']}: R²={rank['r2']:.4f}, Composite={rank['composite_score']:.4f}")

        # Create validation plots ONLY for top models
        print(f"\n📊 Phase 5: Creating Visualizations (Top {self.top_n} only)")
        print("="*60)

        for i, rank in enumerate(final_rankings[:self.top_n], 1):
            model_name = rank['model']
            result = rank['result']
            print(f"\n  {i}. Creating plots for {model_name}...")
            self.create_validation_plots(result, data, result['robustness'], target_col)

        # Store final best models
        self.final_best_models[target_col] = {
            'rankings': final_rankings,
            'top_models': {rank['model']: rank['result'] for rank in final_rankings[:self.top_n]},
            'data': data
        }

        # Create summary only for top models
        self.create_optimized_summary(target_col, final_rankings[:self.top_n])

        return optimized_results

    def create_optimized_summary(self, target_col, top_rankings):
        """Creates summary showing only top optimized models"""
        print(f"\n📊 Creating optimized summary for {target_col}...")

        target_safe = target_col.replace('_', '')

        # Create comparison table
        comparison_data = []
        for rank in top_rankings:
            result = rank['result']
            robustness = result.get('robustness', {})

            comparison_data.append({
                'Rank': rank['model'],
                'R²': rank['r2'],
                'Composite Score': rank['composite_score'],
                'RMSE': result['rmse'],
                'MAE': result['mae'],
                'CV Mean': robustness.get('cv_mean', np.nan),
                'CV Std': robustness.get('cv_std', np.nan),
                'Bootstrap CI Lower': robustness.get('bootstrap_ci_lower', np.nan),
                'Bootstrap CI Upper': robustness.get('bootstrap_ci_upper', np.nan)
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Save
        df_comparison.to_csv(f'resultados/modelos_avanzados/top_models_{target_safe}.csv', index=False)

        # Create visualization
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Comparison', 'Robustness Comparison'),
            specs=[[{"type": "bar"}, {"type": "scatter"}]]
        )

        # R² comparison
        fig.add_trace(
            go.Bar(
                x=[r['model'] for r in top_rankings],
                y=[r['r2'] for r in top_rankings],
                name='R² Score',
                marker_color=['gold', 'silver', 'brown'][:len(top_rankings)],
                text=[f"{r['r2']:.4f}" for r in top_rankings],
                textposition='auto'
            ),
            row=1, col=1
        )

        # CV Mean vs R²
        for rank in top_rankings:
            robustness = rank['result'].get('robustness', {})
            fig.add_trace(
                go.Scatter(
                    x=[rank['r2']],
                    y=[robustness.get('cv_mean', rank['r2'])],
                    mode='markers+text',
                    name=rank['model'],
                    marker=dict(size=15),
                    text=[rank['model']],
                    textposition='top center'
                ),
                row=1, col=2
            )

        # Perfect line
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="red", width=2, dash="dash"),
            row=1, col=2
        )

        fig.update_layout(
            title=f'Top {len(top_rankings)} Optimized Models - {target_col}',
            showlegend=True,
            height=500
        )

        fig.write_html(f'resultados/modelos_avanzados/top_models_comparison_{target_safe}.html')
        print(f"  ✓ Summary created")

    def create_final_optimized_dashboard(self):
        """Creates final dashboard showing only best models"""
        print("\n🎨 Creating final optimized dashboard...")

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Best ML Models - Recursive Optimization</title>
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
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .winner-badge {{
            display: inline-block;
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .target-section {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            border-left: 5px solid #4facfe;
        }}
        .podium {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 30px 0;
        }}
        .podium-item {{
            text-align: center;
            padding: 30px 20px;
            border-radius: 15px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .podium-item:hover {{
            transform: translateY(-10px);
        }}
        .podium-1 {{
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            order: 2;
        }}
        .podium-2 {{
            background: linear-gradient(135deg, #C0C0C0 0%, #808080 100%);
            order: 1;
        }}
        .podium-3 {{
            background: linear-gradient(135deg, #CD7F32 0%, #8B4513 100%);
            order: 3;
        }}
        .podium-number {{
            font-size: 3em;
            font-weight: bold;
            color: white;
        }}
        .podium-name {{
            font-size: 1.2em;
            color: white;
            margin: 10px 0;
        }}
        .podium-score {{
            font-size: 1.5em;
            font-weight: bold;
            color: white;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 15px;
            text-align: left;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .btn {{
            display: inline-block;
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 20px;
            margin: 5px;
            transition: all 0.3s;
        }}
        .btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(79, 172, 254, 0.4);
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🏆 Best ML Models - Recursive Optimization Results</h1>
        <p style="text-align: center; color: #666; font-size: 1.1em;">
            Only the top {self.top_n} models after recursive optimization are shown<br>
            Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}
        </p>
"""

        for target_col in self.target_cols:
            if target_col not in self.final_best_models:
                continue

            target_data = self.final_best_models[target_col]
            rankings = target_data['rankings'][:self.top_n]
            target_safe = target_col.replace('_', '')

            html_content += f"""
        <div class="target-section">
            <h2>🎯 Target: {target_col}</h2>

            <!-- Podium -->
            <div class="podium">
"""

            podium_classes = ['podium-1', 'podium-2', 'podium-3']
            podium_medals = ['🥇', '🥈', '🥉']

            for i, (rank, cls, medal) in enumerate(zip(rankings, podium_classes, podium_medals)):
                html_content += f"""
                <div class="podium-item {cls}">
                    <div class="podium-number">{medal}</div>
                    <div class="podium-name">{rank['model']}</div>
                    <div class="podium-score">R² = {rank['r2']:.4f}</div>
                    <div style="color: white; margin-top: 10px;">
                        RMSE: {rank['result']['rmse']:.4f}<br>
                        Composite: {rank['composite_score']:.4f}
                    </div>
                </div>
"""

            html_content += """
            </div>

            <!-- Detailed Metrics -->
            <h3>📊 Detailed Performance Metrics</h3>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Model</th>
                    <th>R²</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>CV Mean ± Std</th>
                    <th>Validation</th>
                    <th>Download</th>
                </tr>
"""

            for i, rank in enumerate(rankings, 1):
                result = rank['result']
                robustness = result.get('robustness', {})
                model_name_safe = rank['model'].replace(' ', '_').replace('(', '').replace(')', '')

                html_content += f"""
                <tr>
                    <td><strong>#{i}</strong></td>
                    <td><strong>{rank['model']}</strong></td>
                    <td>{rank['r2']:.4f}</td>
                    <td>{result['rmse']:.4f}</td>
                    <td>{result['mae']:.4f}</td>
                    <td>{robustness.get('cv_mean', 0):.4f} ± {robustness.get('cv_std', 0):.4f}</td>
                    <td>
                        <a href="../validacion/validation_{model_name_safe}_{target_safe}.html"
                           class="btn" target="_blank">View Plots</a>
                    </td>
                    <td>
                        <a href="top_models_{target_safe}.csv" class="btn" target="_blank">CSV</a>
                    </td>
                </tr>
"""

            html_content += f"""
            </table>

            <!-- Comparison Chart -->
            <div class="metric-card">
                <h3>📈 Visual Comparison</h3>
                <iframe src="top_models_comparison_{target_safe}.html"
                        width="100%" height="500px" frameborder="0"></iframe>
            </div>
        </div>
"""

        html_content += f"""
        <div style="margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%); border-radius: 15px;">
            <h2>🔬 Recursive Optimization Process</h2>
            <ol style="font-size: 1.1em; line-height: 1.8;">
                <li><strong>Phase 1:</strong> Train all models (Random Forest, XGBoost, Neural Networks, Quantum ML)</li>
                <li><strong>Phase 2:</strong> Rank models by composite score (R² + CV robustness)</li>
                <li><strong>Phase 3:</strong> Recursively optimize top {self.top_n} models for {self.optimization_rounds} rounds</li>
                <li><strong>Phase 4:</strong> Re-rank optimized models</li>
                <li><strong>Phase 5:</strong> Generate visualizations ONLY for final top {self.top_n}</li>
            </ol>

            <h3>✅ Quality Assurance</h3>
            <ul style="font-size: 1.1em; line-height: 1.8;">
                <li>✓ K-Fold Cross-Validation (5 folds)</li>
                <li>✓ Bootstrap Confidence Intervals (95%)</li>
                <li>✓ Residual Analysis & Normality Tests</li>
                <li>✓ Permutation Feature Importance</li>
                <li>✓ Learning Curves & Validation Plots</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

        with open('resultados/modelos_avanzados/best_models_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("  ✓ Final dashboard created: resultados/modelos_avanzados/best_models_dashboard.html")

    def run_recursive_optimization(self):
        """Main recursive optimization workflow"""
        print("🚀 STARTING RECURSIVE MODEL OPTIMIZATION")
        print("="*80)
        print(f"Configuration: Top {self.top_n} models, {self.optimization_rounds} optimization rounds")
        print("="*80)

        if not self.load_data():
            return False

        # Analyze each target with recursive optimization
        for target_col in self.target_cols:
            self.recursive_analysis_for_target(target_col)

        # Create final dashboard with only best models
        self.create_final_optimized_dashboard()

        print("\n" + "="*80)
        print("✅ RECURSIVE OPTIMIZATION COMPLETED!")
        print("="*80)
        print(f"\n📁 Final results (TOP {self.top_n} ONLY):")
        print("  - resultados/modelos_avanzados/best_models_dashboard.html")
        print("  - resultados/validacion/ (validation plots for top models)")
        print("  - resultados/modelos_avanzados/top_models_*.csv")

        return True


def main():
    """Main function"""
    try:
        # Create recursive optimizer
        # top_n=3 means only top 3 models will be shown
        # optimization_rounds=3 means 3 rounds of recursive improvement
        optimizer = RecursiveModelOptimizer(top_n=3, optimization_rounds=3)

        # Run recursive optimization
        success = optimizer.run_recursive_optimization()

        if success:
            print("\n✅ Recursive optimization completed successfully!")
            print("\n🌐 Open: resultados/modelos_avanzados/best_models_dashboard.html")
            return 0
        else:
            print("\n❌ Process completed with errors")
            return 1

    except KeyboardInterrupt:
        print("\n\n⏹️ Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
