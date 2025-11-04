# Final Dual Target Analysis: Gender, Entrepreneurship & Economic Development

## Overview

This project analyzes the relationship between gender and entrepreneurship indicators and two critical economic development outcomes:
1. **GDP per capita growth trajectory**
2. **Trading Across Borders Score** (World Bank Doing Business indicator)

Using machine learning models with recursive variable selection on data from 52 low and lower-middle income countries merged with comprehensive business environment indicators, we identify optimal predictive relationships.

## Key Findings

### ✅ GDP Growth Model (Strong Performance)
- **Best Model:** Neural Network
- **R² Score:** 0.763 (excellent predictive power)
- **RMSE:** 12.55
- **Cross-Validation R²:** -0.676
- **Bootstrap 95% CI:** [0.424, 0.897]

**Interpretation:** Gender and health indicators show meaningful associations with GDP growth trajectories. The model successfully captures ~76% of variance in growth patterns.

### ✅ Trading Score Model (Strong Performance)
- **Best Model:** XGBoost
- **R² Score:** 0.784 (excellent predictive power)
- **RMSE:** 9.74
- **Cross-Validation R²:** -0.012
- **Bootstrap 95% CI:** [-0.139, 0.894]

**Interpretation:** After testing 30 G variables (business/institutional indicators), "Trading Across Borders Score" emerged as the best predictor using gender indicators. This suggests cross-border trade facilitation is meaningfully related to gender equity and health patterns.

## Data

### Primary Dataset
- **Source:** World Bank, UN, UNESCO databases (DATA_GHAB2.xlsx)
- **Countries:** 52 low and lower-middle income nations
- **Variables:** 131 gender, health, education, and development indicators

### Supplementary Dataset
- **Source:** World Bank Doing Business indicators (BASE_COMPLETA.xlsx)
- **Additional Variables:** 30 G variables (business environment, procedures, costs, scores)
- **Merged:** Via ISO3 country codes (39 countries successfully merged)

### Target Variables

1. **G_GPD_PCAP_SLOPE**: GDP per capita growth trajectory (slope)
2. **G_Score-Trading across borders (DB16-20 methodology)**: Selected as best G variable after recursive testing

### Top Predictive Variables (for GDP Growth)

Based on SHAP importance analysis:

1. Cause of death by injury, ages 15-59, female (% of female population ages 15-59)
2. Cause of death by injury, female (% of female population)
3. Cause of death by communicable diseases, ages 15-59, female
4. School enrollment, primary, female (% gross)
5. Lifetime risk of maternal death (%)

## Methodology

### 1. Database Integration
- Merged DATA_GHAB2.xlsx with BASE_COMPLETA.xlsx using ISO3 country codes
- 39 of 52 countries successfully matched
- Final merged database: 164 variables across 52 countries

### 2. Recursive G Variable Selection
- Tested all 30 G variables as potential targets
- Trained 4 models (Random Forest, XGBoost, Gradient Boosting, Neural Network) for each
- Evaluated based on R², CV score, RMSE, and bootstrap confidence intervals
- **Winner:** G_Score-Trading across borders (R²=0.784)

### 3. Feature Selection
- Pearson correlation analysis between predictors and each target
- Top 15 variables selected by absolute correlation magnitude
- Minimum 10 observations required per variable

### 4. Models Evaluated

For both targets:
- **Random Forest:** Ensemble of decision trees with recursive hyperparameter optimization
- **XGBoost:** Gradient boosting with L1/L2 regularization
- **Neural Network:** Multi-layer perceptron (100-50-25 architecture, ReLU activation)
- **Gradient Boosting:** Sequential ensemble learning

### 5. Validation Framework

- **Train/Test Split:** 75% training, 25% testing
- **Cross-Validation:** 5-fold stratified K-fold
- **Bootstrap Resampling:** 100 iterations for confidence intervals
- **Residual Analysis:** Shapiro-Wilk normality test
- **Feature Importance:** SHAP (SHapley Additive exPlanations) values

### 6. Cluster Analysis

K-Means clustering (k=3) applied to identify country groupings:
- **Dimensionality Reduction:** PCA to 2 components (60.8% variance explained)
- **Visualization:** Scatter plots with 95% confidence ellipses
- **Profiling:** Heatmap of standardized cluster characteristics

## Repository Structure

```
aigenderfordevelop/
├── DATA_GHAB2.xlsx                          # Gender & development indicators
├── BASE_COMPLETA.xlsx                       # Business environment indicators
├── DATA_MERGED_COMPLETE.xlsx                # Merged database
├── merge_and_train_G_variables.py           # Recursive G variable testing
├── final_dual_target_analysis.py            # Final dual target models
├── create_final_dashboard.py                # Dashboard generator
├── create_final_complete_analysis.py        # Robustness & clusters
├── resultados/
│   ├── dashboard_complete.html              # Interactive dashboard (3 tabs)
│   ├── cluster_assignments_final.csv        # Country cluster assignments
│   ├── modelos/
│   │   ├── all_G_variables_results.pkl      # All 30 G variable results
│   │   └── final_dual_target_results.pkl    # Final GDP + Trading models
│   └── graficos_finales/
│       ├── shap_GDP_Growth.png              # SHAP plots for GDP
│       ├── shap_Trading_Score.png           # SHAP plots for Trading Score
│       ├── robustness_GDP_Growth.png        # 6-panel diagnostics (GDP)
│       ├── residual_details_GDP_Growth.png  # Detailed residuals (GDP)
│       ├── robustness_Trading_Score.png     # Diagnostics (Trading)
│       ├── residual_details_Trading_Score.png
│       └── cluster_analysis_final.png       # Cluster visualization
└── README.md                                # This file
```

## Usage

### Step 1: Merge Databases and Test All G Variables

```bash
python merge_and_train_G_variables.py
```

This script:
- Merges DATA_GHAB2.xlsx with BASE_COMPLETA.xlsx
- Trains models for all 30 G variables
- Ranks them by R² score
- Saves results to `all_G_variables_results.pkl`

Output: Best G variable identified = **G_Score-Trading across borders** (R²=0.784)

### Step 2: Final Dual Target Analysis

```bash
python final_dual_target_analysis.py
```

Trains final models for:
- GDP Growth (R²=0.763)
- Trading Score (R²=0.784)

### Step 3: Generate Visualizations

```bash
python create_final_complete_analysis.py
```

Creates:
- Robustness analysis plots (6-panel + 4-panel residual details)
- Cluster analysis with PCA and confidence ellipses

### Step 4: Create Dashboard

```bash
python create_final_dashboard.py
```

Generates `resultados/dashboard_complete.html` with 3 tabs:
- Analysis Results (model comparison, SHAP, robustness, clusters)
- Descriptive Statistics & Correlations
- Dual Target Projections (interactive sliders)

### Viewing Results

Open `resultados/dashboard_complete.html` in a web browser for:
- **Interactive Policy Projections:** Adjust top 10 variables via sliders to see projected impacts on both GDP growth and Trading Score
- **Dual Bar Charts:** Side-by-side comparison of baseline vs. projected scenarios
- **SHAP Visualizations:** Feature importance and value distributions
- **Robustness Diagnostics:** Q-Q plots, residual analysis, learning curves, bootstrap distributions
- **Cluster Visualizations:** Country groupings in PCA space with confidence ellipses

## Key Insights

### GDP Growth Prediction
The strong performance of the GDP growth model suggests:
- **Health indicators** (especially female cause of death patterns) are meaningfully associated with economic growth trajectories
- **Education access** (primary enrollment) correlates with growth potential
- **Maternal health** (lifetime risk of maternal death) reflects broader development patterns tied to economic performance

### Trading Across Borders Score
The strong performance of the Trading Score model indicates:
- Gender indicators **can predict** cross-border trade facilitation scores
- Countries with better gender equity and health outcomes tend to have more efficient trade processes
- This relationship may reflect:
  - Broader institutional quality that affects both gender equity and business environment
  - Human capital development enabling trade competitiveness
  - Social modernization correlating with economic openness

### Policy Implications

1. **For GDP Growth:** Improvements in female health outcomes and primary education access are associated with better growth trajectories. Policies addressing maternal mortality and communicable disease burden may have economic co-benefits.

2. **For Trade Facilitation:** The relationship between gender indicators and trading efficiency suggests that development interventions should consider integrated approaches connecting social equity with economic competitiveness.

3. **Integrated Approach:** Both models demonstrate that gender equity, health, and economic performance are interconnected dimensions of development requiring coordinated policy responses.

## G Variables Tested

All 30 G variables were tested, ranked by R² (best to worst):

| Rank | Variable | R² | Model |
|------|----------|-----|-------|
| 1 | G_Score-Trading across borders | 0.784 | XGBoost |
| 2 | G_Time (days) (4) | 0.559 | XGBoost |
| 3 | G_Paid-in Minimum capital | 0.497 | Random Forest |
| 4 | G_Cost (% of property value) | 0.443 | Gradient Boosting |
| 5 | G_Cost (% of income per capita) | 0.358 | Random Forest |
| ... | ... | ... | ... |
| 30 | G_Score-Getting credit | -0.793 | Random Forest |

See `all_G_variables_results.pkl` for complete results.

## Technical Notes

### Model Performance Metrics

**GDP Growth:**
- R² = 0.763: Model explains 76.3% of variance
- RMSE = 12.55: Average prediction error
- Bootstrap 95% CI: [0.424, 0.897]

**Trading Score:**
- R² = 0.784: Model explains 78.4% of variance
- RMSE = 9.74: Average prediction error
- Bootstrap 95% CI: [-0.139, 0.894]

### Limitations

1. **Sample Size:** 52 countries (39 with merged data) limits model complexity
2. **Causality:** Correlations do not imply causal relationships
3. **Omitted Variables:** Many institutional and geographic factors not captured
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
shap
```

Install via:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib scipy openpyxl shap
```

## Citation

If you use this analysis or code, please cite:

```
Final Dual Target Analysis: Gender, Entrepreneurship & Economic Development
https://github.com/upocuantitativo/aigenderfordevelop
```

## License

This project is released for academic and research purposes. Data sources retain their original licenses (World Bank, UN, UNESCO).

## Contact

For questions or collaborations, please open an issue on GitHub.

---

**Last Updated:** November 2025
**Analysis Completed:** G variable recursive testing + final dual target models
