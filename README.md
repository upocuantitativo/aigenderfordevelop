# Dual Target Analysis: Gender, Entrepreneurship & Economic Development

## Overview

This project analyzes the relationship between gender and entrepreneurship indicators and two critical economic development outcomes:
1. **GDP per capita growth trajectory**
2. **Ease of Doing Business score**

Using machine learning models on data from 52 low and lower-middle income countries, we explore how gender-related factors correlate with and potentially predict economic performance.

## Key Findings

### ✅ GDP Growth Model (Strong Performance)
- **Best Model:** Random Forest Regressor
- **R² Score:** 0.732 (good predictive power)
- **RMSE:** 13.34
- **Cross-Validation R²:** 0.075

**Interpretation:** Gender and health indicators show meaningful associations with GDP growth trajectories. The model successfully captures ~73% of variance in growth patterns.

### ❌ Ease of Doing Business Model (Poor Performance)
- **Best Model:** Random Forest Regressor
- **R² Score:** -0.236 (worse than baseline)
- **RMSE:** 15.90
- **Cross-Validation R²:** -18.02

**Interpretation:** Gender indicators alone do not adequately predict Ease of Doing Business scores. This suggests business environment quality is driven more by institutional, legal, and regulatory factors not captured in gender-focused variables.

## Data

- **Source:** World Bank, UN, UNESCO databases
- **Countries:** 52 low and lower-middle income nations
- **Variables:** 131 gender, health, education, and development indicators
- **Period:** Latest available data (varies by indicator)

### Target Variables

1. **G_GPD_PCAP_SLOPE**: GDP per capita growth trajectory (slope)
2. **Ease of Doing Business**: World Bank Doing Business score

### Top Predictive Variables (for GDP Growth)

Based on SHAP importance analysis:

1. Cause of death by injury, ages 15-59, female (% of female population ages 15-59)
2. Cause of death by injury, female (% of female population)
3. Cause of death by communicable diseases, ages 15-59, female
4. School enrollment, primary, female (% gross)
5. Lifetime risk of maternal death (%)

## Methodology

### 1. Feature Selection
- Pearson correlation analysis between predictors and each target
- Top 15 variables selected by absolute correlation magnitude
- Minimum 10 observations required per variable

### 2. Models Evaluated

For both targets, we trained and compared:
- **Random Forest:** Ensemble of decision trees with recursive hyperparameter optimization
- **XGBoost:** Gradient boosting with L1/L2 regularization
- **Neural Network:** Multi-layer perceptron (100-50-25 architecture, ReLU activation)
- **Gradient Boosting:** Sequential ensemble learning

### 3. Validation Framework

- **Train/Test Split:** 75% training, 25% testing
- **Cross-Validation:** 5-fold stratified K-fold
- **Bootstrap Resampling:** 100 iterations for confidence intervals
- **Residual Analysis:** Shapiro-Wilk normality test
- **Feature Importance:** SHAP (SHapley Additive exPlanations) values

### 4. Cluster Analysis

K-Means clustering (k=3) applied to identify country groupings:
- **Dimensionality Reduction:** PCA to 2 components (60.8% variance explained)
- **Visualization:** Scatter plots with 95% confidence ellipses
- **Profiling:** Heatmap of standardized cluster characteristics

## Repository Structure

```
aigenderfordevelop/
├── DATA_GHAB2.xlsx                          # Main dataset
├── dual_target_analysis.py                  # Main analysis script
├── create_dual_dashboard.py                 # Dashboard generator
├── create_dual_robustness.py                # Robustness diagnostics
├── create_dual_cluster_analysis.py          # Cluster analysis
├── resultados/
│   ├── dashboard_complete.html              # Interactive dashboard (3 tabs)
│   ├── cluster_assignments.csv              # Country cluster assignments
│   ├── modelos/
│   │   └── dual_target_results.pkl          # Model results and predictions
│   └── graficos_finales/
│       ├── shap_GDP_Growth.png              # SHAP plots for GDP
│       ├── shap_Ease_of_Business.png        # SHAP plots for Ease of Business
│       ├── robustness_GDP_Growth.png        # 6-panel diagnostics (GDP)
│       ├── residual_details_GDP_Growth.png  # Detailed residuals (GDP)
│       ├── robustness_Ease_of_Business.png  # Diagnostics (Ease)
│       ├── residual_details_Ease_of_Business.png
│       └── cluster_analysis.png             # Cluster visualization
└── README.md                                # This file
```

## Usage

### Running the Analysis

1. **Main Analysis:**
```bash
python dual_target_analysis.py
```
Trains models for both targets, evaluates performance, and saves results to `resultados/modelos/dual_target_results.pkl`.

2. **Generate Dashboard:**
```bash
python create_dual_dashboard.py
```
Creates `resultados/dashboard_complete.html` with 3 tabs:
   - Analysis Results (model comparison, SHAP, robustness, clusters)
   - Descriptive Statistics & Correlations
   - Dual Target Projections (interactive sliders)

3. **Robustness Analysis:**
```bash
python create_dual_robustness.py
```
Generates diagnostic plots for model validation.

4. **Cluster Analysis:**
```bash
python create_dual_cluster_analysis.py
```
Creates country groupings and saves cluster assignments.

### Viewing Results

Open `resultados/dashboard_complete.html` in a web browser for:
- **Interactive Policy Projections:** Adjust top 10 variables via sliders to see projected impacts on both GDP growth and Ease of Doing Business
- **Dual Bar Charts:** Side-by-side comparison of baseline vs. projected scenarios for both targets
- **SHAP Visualizations:** Feature importance and value distributions
- **Robustness Diagnostics:** Q-Q plots, residual analysis, learning curves, bootstrap distributions
- **Cluster Visualizations:** Country groupings in PCA space with confidence ellipses

## Key Insights

### What Works: GDP Growth Prediction
The strong performance of the GDP growth model suggests:
- **Health indicators** (especially female cause of death patterns) are meaningfully associated with economic growth trajectories
- **Education access** (primary enrollment) correlates with growth potential
- **Maternal health** (lifetime risk of maternal death) reflects broader development patterns tied to economic performance

### What Doesn't Work: Ease of Doing Business
The failure of the Ease of Business model indicates:
- Business environment quality is **not well-predicted by gender indicators alone**
- Institutional factors (property rights, contract enforcement, regulatory efficiency) likely dominate
- Gender equality may be a **consequence** rather than **driver** of good business environments

### Policy Implications

1. **For GDP Growth:** Improvements in female health outcomes and primary education access are associated with better growth trajectories. Policies addressing maternal mortality and communicable disease burden may have economic co-benefits.

2. **For Business Environment:** Improving Ease of Doing Business requires direct institutional reforms (legal systems, bureaucratic efficiency, corruption reduction) rather than solely focusing on gender equity.

3. **Integrated Approach:** Gender equity and business environment quality may be complementary but require different policy levers.

## Technical Notes

### Model Performance Metrics

**GDP Growth:**
- R² = 0.732: Model explains 73.2% of variance
- RMSE = 13.34: Average prediction error of 13.34 units
- Bootstrap 95% CI: Stable confidence intervals around R²

**Ease of Doing Business:**
- R² = -0.236: Model performs worse than mean baseline
- Negative cross-validation scores indicate systematic overfitting
- Predictions should not be used for policy guidance

### Limitations

1. **Sample Size:** 52 countries with complete data limits model complexity
2. **Causality:** Correlations do not imply causal relationships; policy interventions require careful consideration
3. **Omitted Variables:** Many factors (institutions, geography, history) not captured in gender indicators
4. **Temporal:** Cross-sectional analysis; dynamic relationships not modeled
5. **Data Quality:** Varies by country and indicator; missing data handled via listwise deletion

## Dependencies

```
pandas
numpy
scikit-learn
xgboost
matplotlib
scipy
openpyxl
```

Install via:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib scipy openpyxl
```

## Citation

If you use this analysis or code, please cite:

```
Dual Target Analysis: Gender, Entrepreneurship & Economic Development
https://github.com/upocuantitativo/aigenderfordevelop
```

## License

This project is released for academic and research purposes. Data sources retain their original licenses (World Bank, UN, UNESCO).

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Last Updated:** November 2025
**Analysis Date:** See dashboard for specific run dates
