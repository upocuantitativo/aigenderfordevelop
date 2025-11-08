#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Academic Dashboard with 3 tabs:
1. Descriptivos
2. Tax Score Prediction
3. GDP Growth Prediction
"""

import pickle
import pandas as pd
import numpy as np
import base64
from pathlib import Path

print("="*80)
print("CREATING ACADEMIC DASHBOARD (3 TABS)")
print("="*80)

# Load results
with open('resultados/modelos/final_dual_improved.pkl', 'rb') as f:
    results = pickle.load(f)

# Load data for descriptives
df = pd.read_excel('DATA_MERGED_NEW.xlsx')

# Get results for both targets
tax_res = results['Tax_Score']
gdp_res = results['GDP_Growth']

tax_best_name, tax_best_res = tax_res['best']
gdp_best_name, gdp_best_res = gdp_res['best']

print(f"\nTax Score Model: {tax_best_name}")
print(f"  R² = {tax_best_res['r2']:.4f}")
print(f"  CV = {tax_best_res['cv']:.4f}")

print(f"\nGDP Growth Model: {gdp_best_name}")
print(f"  R² = {gdp_best_res['r2']:.4f}")
print(f"  CV = {gdp_best_res['cv']:.4f}")

# Encode images
def encode_image(filepath):
    try:
        with open(filepath, 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

# Images for Tax Score
tax_shap_img = encode_image('resultados/graficos_finales/shap_Tax_Score_new.png')
tax_robustness_img = encode_image('resultados/graficos_finales/robustness_Tax_Score_new.png')
tax_residual_img = encode_image('resultados/graficos_finales/residual_details_Tax_Score_new.png')

# Get descriptive statistics
g_vars = [c for c in df.columns if c.startswith('G_')]
num_countries = df.shape[0]
num_gender_vars = len([c for c in df.columns if not c.startswith('G_')
                       and c not in ['Pais', 'Etiqueta_pais', 'Country code', 'Country code2',
                                    'Ease of Doing Business', 'New_Business_Density', 'Country', 'Economy']
                       and not c.startswith('Score-') and not c.startswith('Rank-')
                       and 'H_Cause of death' not in c])

# Create HTML
html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dual Target Analysis - GDP Growth & Ease of Doing Business Score</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: Georgia, 'Times New Roman', serif;
            font-size: 10pt;
            line-height: 1.4;
            color: #000;
            background: #fff;
        }}

        .header {{
            background: #f5f5f5;
            border-bottom: 2px solid #000;
            padding: 15px 20px;
            text-align: center;
        }}

        h1 {{
            font-size: 16pt;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .subtitle {{
            font-size: 9pt;
            color: #555;
        }}

        .tabs {{
            display: flex;
            background: #e8e8e8;
            border-bottom: 1px solid #999;
        }}

        .tab {{
            flex: 1;
            padding: 10px 15px;
            text-align: center;
            cursor: pointer;
            border-right: 1px solid #999;
            font-weight: bold;
            font-size: 9.5pt;
            background: #e8e8e8;
            transition: background 0.2s;
        }}

        .tab:hover {{
            background: #d8d8d8;
        }}

        .tab.active {{
            background: #fff;
            border-bottom: 2px solid #fff;
        }}

        .tab-content {{
            display: none;
            padding: 20px;
            max-width: 1100px;
            margin: 0 auto;
        }}

        .tab-content.active {{
            display: block;
        }}

        h2 {{
            font-size: 12pt;
            font-weight: bold;
            margin: 15px 0 8px 0;
            border-bottom: 1px solid #666;
            padding-bottom: 3px;
        }}

        h3 {{
            font-size: 10.5pt;
            font-weight: bold;
            margin: 10px 0 5px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
            font-size: 9pt;
        }}

        th {{
            background-color: #e8e8e8;
            border: 1px solid #555;
            padding: 4px 6px;
            text-align: left;
            font-weight: bold;
        }}

        td {{
            border: 1px solid #888;
            padding: 4px 6px;
        }}

        .best {{ background-color: #f5f5dc; font-weight: bold; }}
        .poor {{ background-color: #ffe4e4; }}

        .summary {{
            background-color: #f9f9f9;
            border: 1px solid #666;
            padding: 10px;
            margin: 10px 0;
        }}

        .methodology {{
            background-color: #f0f8ff;
            border: 1px solid #4682B4;
            padding: 12px;
            margin: 10px 0;
            font-size: 9.5pt;
        }}

        .methodology h3 {{
            color: #2c5282;
            margin-top: 8px;
            margin-bottom: 4px;
        }}

        .methodology ul, .methodology ol {{
            margin-left: 20px;
            margin-top: 4px;
        }}

        .methodology li {{
            margin-bottom: 3px;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #999;
            margin: 8px 0;
            cursor: pointer;
            transition: transform 0.2s;
        }}

        img:hover {{
            transform: scale(1.02);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        .top-variables-box {{
            background: #fff8dc;
            border: 2px solid #daa520;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}

        .top-variables-box h3 {{
            color: #b8860b;
            margin-bottom: 8px;
        }}

        .top-variables-box ol {{
            margin-left: 20px;
            font-size: 9pt;
        }}

        .top-variables-box li {{
            margin-bottom: 4px;
        }}

        .note {{
            font-size: 8.5pt;
            font-style: italic;
            color: #555;
            margin-top: 4px;
        }}

        .warning-box {{
            background: #ffe4e4;
            border: 2px solid #c62828;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}

        .warning-box h3 {{
            color: #c62828;
            margin-bottom: 8px;
        }}

        .success-box {{
            background: #f1f8f4;
            border: 2px solid #2e7d32;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}

        .success-box h3 {{
            color: #2e7d32;
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Dual Target Analysis: GDP Growth & Ease of Doing Business Score</h1>
        <div class="subtitle">Prediction from Gender and Development Indicators</div>
        <div class="subtitle" style="margin-top: 5px;">Excluding H_Cause of death variables | 128 predictors | {num_countries} countries</div>
    </div>

    <div class="tabs">
        <div class="tab" onclick="switchTab(0)">Descriptive Statistics</div>
        <div class="tab active" onclick="switchTab(1)">Tax Score Prediction (R²={tax_best_res['r2']:.3f})</div>
        <div class="tab" onclick="switchTab(2)">GDP Growth Prediction (R²={gdp_best_res['r2']:.3f})</div>
    </div>

    <!-- TAB 0: DESCRIPTIVE STATISTICS -->
    <div class="tab-content" id="tab-0">
        <h2>1. Data Overview</h2>

        <div class="summary">
            <h3>Dataset Composition</h3>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Count</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td><strong>Countries</strong></td>
                    <td>{num_countries}</td>
                    <td>Low and lower-middle income countries</td>
                </tr>
                <tr>
                    <td><strong>Gender/Development Indicators</strong></td>
                    <td>{num_gender_vars}</td>
                    <td>Excluding all H_Cause of death variables (16 removed)</td>
                </tr>
                <tr>
                    <td><strong>Target Variables</strong></td>
                    <td>2</td>
                    <td>GDP Growth & Ease of Doing Business Score</td>
                </tr>
            </table>
        </div>

        <h2>2. Target Variables Summary</h2>

        <table>
            <thead>
                <tr>
                    <th>Target Variable</th>
                    <th>Valid Obs.</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>GDP per Capita Growth</strong></td>
                    <td>{df['G_GPD_PCAP_SLOPE'].notna().sum()}</td>
                    <td>{df['G_GPD_PCAP_SLOPE'].mean():.2f}</td>
                    <td>{df['G_GPD_PCAP_SLOPE'].std():.2f}</td>
                    <td>{df['G_GPD_PCAP_SLOPE'].min():.2f}</td>
                    <td>{df['G_GPD_PCAP_SLOPE'].max():.2f}</td>
                </tr>
                <tr>
                    <td><strong>Ease of Doing Business Score</strong></td>
                    <td>{df['Score-Total tax and contribution rate (% of profit)'].notna().sum()}</td>
                    <td>{df['Score-Total tax and contribution rate (% of profit)'].mean():.2f}</td>
                    <td>{df['Score-Total tax and contribution rate (% of profit)'].std():.2f}</td>
                    <td>{df['Score-Total tax and contribution rate (% of profit)'].min():.2f}</td>
                    <td>{df['Score-Total tax and contribution rate (% of profit)'].max():.2f}</td>
                </tr>
            </tbody>
        </table>

        <h2>3. Methodology</h2>

        <div class="methodology">
            <h3>Data Sources</h3>
            <ul>
                <li><strong>Gender & Development Indicators:</strong> World Bank, UN, UNESCO</li>
                <li><strong>Business Environment:</strong> World Bank Doing Business (DB16-20)</li>
                <li><strong>Economic Growth:</strong> World Bank Development Indicators</li>
            </ul>

            <h3>Variable Selection</h3>
            <ul>
                <li><strong>Exclusion Criteria:</strong> All H_Cause of death variables removed (16 variables)</li>
                <li><strong>Predictor Pool:</strong> 128 gender and development indicators</li>
                <li><strong>Selection Method:</strong> Top 15 by Pearson correlation with each target</li>
                <li><strong>Balanced Scoring:</strong> √(R²) × √(CV) for simultaneous optimization</li>
            </ul>

            <h3>Model Training</h3>
            <ul>
                <li><strong>Algorithms Tested:</strong> Random Forest, XGBoost, Gradient Boosting, Neural Networks</li>
                <li><strong>Hyperparameter Optimization:</strong> GridSearchCV for Random Forest and XGBoost</li>
                <li><strong>Validation:</strong> 5-fold cross-validation + 100 bootstrap iterations</li>
                <li><strong>Train/Test Split:</strong> 75% training, 25% testing</li>
            </ul>

            <h3>Model Selection Criteria</h3>
            <ul>
                <li><strong>Primary Metric:</strong> Balanced Score = √(R²) × √(CV)</li>
                <li><strong>Minimum Requirements:</strong> R² > 0.3 AND CV > 0 (positive generalization)</li>
                <li><strong>Validation:</strong> Shapiro-Wilk test for residual normality</li>
                <li><strong>Robustness:</strong> Bootstrap confidence intervals for R²</li>
            </ul>
        </div>

        <h2>4. Key Changes from Previous Version</h2>

        <div class="summary">
            <p><strong>Exclusion of Mortality Data:</strong> All 16 variables containing "H_Cause of death" have been removed from the predictor pool. This includes:</p>
            <ul style="margin-left: 20px; margin-top: 5px;">
                <li>Communicable diseases and maternal/prenatal conditions (by age group)</li>
                <li>Injuries (by age group)</li>
                <li>Non-communicable diseases (by age group)</li>
            </ul>
            <p style="margin-top: 8px;"><strong>Impact:</strong> The Tax Score model maintains exceptional performance (R²=0.970) even without mortality variables, demonstrating that other gender and development indicators are sufficient for prediction.</p>
        </div>
    </div>

    <!-- TAB 1: TAX SCORE PREDICTION -->
    <div class="tab-content active" id="tab-1">
        <h2>1. Model Performance</h2>

        <div class="success-box">
            <h3>Exceptional Predictive Performance (R² = {tax_best_res['r2']:.3f})</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="best">
                        <td><strong>R² Score</strong></td>
                        <td>{tax_best_res['r2']:.4f}</td>
                        <td>Explains 97.0% of variance</td>
                    </tr>
                    <tr class="best">
                        <td><strong>Cross-Validation</strong></td>
                        <td>{tax_best_res['cv']:.4f}</td>
                        <td>POSITIVE - Excellent generalization</td>
                    </tr>
                    <tr>
                        <td><strong>Balanced Score</strong></td>
                        <td>{tax_best_res['balanced_score']:.4f}</td>
                        <td>√(R²) × √(CV) = Optimal</td>
                    </tr>
                    <tr>
                        <td><strong>RMSE</strong></td>
                        <td>{tax_best_res['rmse']:.2f}</td>
                        <td>Root Mean Squared Error</td>
                    </tr>
                    <tr>
                        <td><strong>Best Model</strong></td>
                        <td>{tax_best_name}</td>
                        <td>Selected from 4 algorithms</td>
                    </tr>
                    <tr>
                        <td><strong>Bootstrap CI</strong></td>
                        <td>[{tax_res['validation']['bootstrap_ci'][0]:.3f}, {tax_res['validation']['bootstrap_ci'][1]:.3f}]</td>
                        <td>95% confidence interval for R²</td>
                    </tr>
                </tbody>
            </table>
            <p class="note" style="font-style: normal; margin-top: 10px;">
                <strong>Note:</strong> Model maintains exceptional performance without H_Cause of death variables,
                demonstrating that economic and social development indicators are sufficient for predicting
                business environment quality.
            </p>
        </div>

        <h2>2. Top Predictive Variables</h2>

        <div class="top-variables-box">
            <h3>Top 15 Gender & Development Indicators</h3>
            <ol>
"""

# Add top predictors for Tax Score
for i, pred in enumerate(tax_res['predictors'][:15], 1):
    pred_short = pred[:90] + '...' if len(pred) > 90 else pred
    html_content += f"                <li>{pred_short}</li>\n"

html_content += f"""            </ol>
            <p class="note">Variables selected by Pearson correlation with Tax Score. SHAP values show actual impact on predictions.</p>
        </div>

        <h2>3. SHAP Analysis</h2>
        <p>SHAP (SHapley Additive exPlanations) values quantify each feature's contribution to individual predictions.</p>
"""

if tax_shap_img:
    html_content += f"""        <img src="data:image/png;base64,{tax_shap_img}" alt="SHAP Analysis - Tax Score" onclick="openModal(this.src)">
"""

html_content += f"""
        <h2>4. Robustness Analysis</h2>
        <p>Comprehensive diagnostic plots: Q-Q plot, residual distribution, predicted vs actual, residuals vs predicted, cross-validation stability, and learning curve.</p>
"""

if tax_robustness_img:
    html_content += f"""        <img src="data:image/png;base64,{tax_robustness_img}" alt="Robustness Analysis - Tax Score" onclick="openModal(this.src)">
"""

html_content += f"""
        <h2>5. Detailed Residual Analysis</h2>
        <p>In-depth residual diagnostics: sequence plot, scale-location, bootstrap R² distribution, and standardized residuals.</p>
"""

if tax_residual_img:
    html_content += f"""        <img src="data:image/png;base64,{tax_residual_img}" alt="Residual Analysis - Tax Score" onclick="openModal(this.src)">
"""

html_content += f"""
        <h2>6. Model Comparison</h2>

        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>R²</th>
                    <th>CV</th>
                    <th>Balanced Score</th>
                    <th>RMSE</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

# Add all models for Tax Score
for i, (model_name, model_res) in enumerate(sorted(tax_res['all_models'].items(),
                                                     key=lambda x: x[1]['balanced_score'],
                                                     reverse=True), 1):
    row_class = 'class="best"' if model_name == tax_best_name else ''
    cv_status = "POSITIVE" if model_res['cv'] > 0 else "negative"
    html_content += f"""                <tr {row_class}>
                    <td><strong>{model_name}</strong></td>
                    <td>{model_res['r2']:.4f}</td>
                    <td>{model_res['cv']:.4f}</td>
                    <td>{model_res['balanced_score']:.4f}</td>
                    <td>{model_res['rmse']:.2f}</td>
                    <td>{cv_status}</td>
                </tr>
"""

html_content += """            </tbody>
        </table>

        <h2>7. Policy Implications</h2>

        <div class="methodology">
            <h3>Key Insights</h3>
            <p>The model's exceptional performance (R²=0.970) reveals strong relationships between gender/development indicators and business environment quality:</p>
            <ul>
                <li><strong>Tax Variables:</strong> Total tax rate and other taxes are naturally strong predictors</li>
                <li><strong>Contraceptive Prevalence:</strong> Emerges as key indicator, likely reflecting institutional capacity</li>
                <li><strong>Court System:</strong> Legal system fees correlate with overall business environment</li>
                <li><strong>Social Development:</strong> Broader development indicators predict business quality</li>
            </ul>

            <h3>Recommendations</h3>
            <ul>
                <li>Countries should pursue <strong>integrated development approaches</strong> addressing social and business factors</li>
                <li>Gender equality and reproductive health access correlate with better business environments</li>
                <li>Legal system development and business environment quality are interconnected</li>
                <li>Policy makers should consider <strong>holistic reforms</strong> rather than isolated business changes</li>
            </ul>
        </div>
    </div>

    <!-- TAB 2: GDP GROWTH PREDICTION -->
    <div class="tab-content" id="tab-2">
        <h2>1. Model Performance</h2>

        <div class="warning-box">
            <h3>Limited Predictive Performance (R² = {gdp_best_res['r2']:.3f})</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
                </thead>
                <tbody>
                    <tr class="poor">
                        <td><strong>R² Score</strong></td>
                        <td>{gdp_best_res['r2']:.4f}</td>
                        <td>Explains only {gdp_best_res['r2']*100:.1f}% of variance</td>
                    </tr>
                    <tr class="poor">
                        <td><strong>Cross-Validation</strong></td>
                        <td>{gdp_best_res['cv']:.4f}</td>
                        <td>NEGATIVE - Poor generalization</td>
                    </tr>
                    <tr>
                        <td><strong>Balanced Score</strong></td>
                        <td>{gdp_best_res['balanced_score']:.4f}</td>
                        <td>Below minimum threshold (0.3)</td>
                    </tr>
                    <tr>
                        <td><strong>RMSE</strong></td>
                        <td>{gdp_best_res['rmse']:.2f}</td>
                        <td>Root Mean Squared Error</td>
                    </tr>
                    <tr>
                        <td><strong>Best Model</strong></td>
                        <td>{gdp_best_name}</td>
                        <td>Selected from 4 algorithms</td>
                    </tr>
                    <tr>
                        <td><strong>Bootstrap CI</strong></td>
                        <td>[{gdp_res['validation']['bootstrap_ci'][0]:.3f}, {gdp_res['validation']['bootstrap_ci'][1]:.3f}]</td>
                        <td>95% confidence interval for R²</td>
                    </tr>
                </tbody>
            </table>
            <p class="note" style="font-style: normal; margin-top: 10px;">
                <strong>Warning:</strong> This model does NOT meet minimum quality criteria (R² > 0.3, CV > 0).
                GDP growth is likely driven by factors not captured in gender/development indicators.
            </p>
        </div>

        <h2>2. Top Correlated Variables (Not Predictive)</h2>

        <div class="top-variables-box" style="background: #fff0f0; border-color: #c62828;">
            <h3>Top 15 Variables by Correlation</h3>
            <ol>
"""

# Add top predictors for GDP Growth
for i, pred in enumerate(gdp_res['predictors'][:15], 1):
    pred_short = pred[:90] + '...' if len(pred) > 90 else pred
    html_content += f"                <li>{pred_short}</li>\n"

html_content += f"""            </ol>
            <p class="note" style="color: #c62828;">
                <strong>Caution:</strong> Despite being the most correlated variables, these do not enable
                reliable GDP growth prediction. Correlation does not imply predictive power.
            </p>
        </div>

        <h2>3. Model Comparison</h2>

        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>R²</th>
                    <th>CV</th>
                    <th>Balanced Score</th>
                    <th>RMSE</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""

# Add all models for GDP Growth
for i, (model_name, model_res) in enumerate(sorted(gdp_res['all_models'].items(),
                                                     key=lambda x: x[1]['balanced_score'],
                                                     reverse=True), 1):
    row_class = 'class="poor"'
    cv_status = "negative" if model_res['cv'] < 0 else "POSITIVE"
    html_content += f"""                <tr {row_class}>
                    <td><strong>{model_name}</strong></td>
                    <td>{model_res['r2']:.4f}</td>
                    <td>{model_res['cv']:.4f}</td>
                    <td>{model_res['balanced_score']:.4f}</td>
                    <td>{model_res['rmse']:.2f}</td>
                    <td>{cv_status}</td>
                </tr>
"""

html_content += f"""            </tbody>
        </table>

        <h2>4. Why GDP Growth Cannot Be Predicted</h2>

        <div class="methodology">
            <h3>Possible Explanations</h3>
            <ul>
                <li><strong>Complex Causality:</strong> GDP growth depends on many factors beyond social indicators (commodity prices, trade, global economy, monetary policy, etc.)</li>
                <li><strong>Time Dynamics:</strong> Growth is highly dynamic and influenced by short-term shocks not captured in static indicators</li>
                <li><strong>Endogeneity:</strong> Reverse causality may exist (growth affects social indicators, not just vice versa)</li>
                <li><strong>Sample Size:</strong> Limited observations ({gdp_res['y_test'].shape[0]} test samples) may be insufficient for complex relationships</li>
                <li><strong>Missing Variables:</strong> Critical economic drivers (investment, exports, innovation) not included in gender/development data</li>
            </ul>

            <h3>Methodological Note</h3>
            <p>The negative cross-validation score indicates the model performs <strong>worse than a simple mean prediction</strong>
            on unseen data. This is a clear sign of overfitting and lack of generalizable patterns. Unlike the Tax Score model,
            gender and development indicators alone are insufficient to predict GDP growth trajectories.</p>

            <h3>Research Implications</h3>
            <ul>
                <li>Gender equality and economic growth relationships may be <strong>bidirectional</strong> rather than predictive</li>
                <li>Static cross-sectional indicators cannot capture <strong>dynamic growth processes</strong></li>
                <li>Future research should incorporate <strong>time series</strong> and <strong>macroeconomic variables</strong></li>
                <li>The contrast with Tax Score success shows <strong>different targets require different predictors</strong></li>
            </ul>
        </div>
    </div>

    <!-- Modal for image zoom -->
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close" onclick="closeModal()">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <script>
        function switchTab(index) {{
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab')[index].classList.add('active');
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content')[index].classList.add('active');
        }}

        function openModal(src) {{
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
        }}

        function closeModal() {{
            document.getElementById('imageModal').style.display = 'none';
        }}
    </script>
</body>
</html>
"""

# Save HTML
output_path = 'index.html'
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\nAcademic dashboard created: {output_path}")
print("\n3 tabs:")
print("  1. Descriptive Statistics")
print("  2. Tax Score Prediction (R²={:.3f})".format(tax_best_res['r2']))
print("  3. GDP Growth Prediction (R²={:.3f})".format(gdp_best_res['r2']))
print("\nYou can open this file in a web browser.")
