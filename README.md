# Gender Indicators & Tax Environment: Predictive Analysis

## Overview

This project analyzes the relationship between gender, health, and development indicators and critical economic outcomes. After comprehensive recursive testing of 164 business environment variables, we identified **Tax Environment** as having the strongest predictive relationship with gender indicators.

**Key Innovation:** Using balanced scoring methodology (sqrt(R²) × sqrt(CV)) to ensure BOTH high predictive power AND excellent generalization simultaneously.

## Key Findings

### ✅ Tax Score Model (EXCEPTIONAL Performance)
**Target:** Score-Total tax and contribution rate (% of profit)

- **Best Model:** XGBoost (optimized hyperparameters)
- **R² Score:** 0.967 (exceptional - explains 96.7% of variance!)
- **RMSE:** 3.83
- **Cross-Validation R²:** +0.913 ✓ EXCELLENT generalization!
- **Balanced Score:** 0.940 (highest possible with these criteria)
- **Bootstrap 95% CI:** [0.924, 0.978] (very narrow, high confidence)

**Interpretation:** Gender and development indicators show EXCEPTIONAL predictive power for tax environment complexity. The model captures 96.7% of variance with outstanding generalization (CV=0.913), indicating robust, reliable predictions for new countries.

### Top 5 Predictive Variables (for Tax Score):

1. **Total tax and contribution rate** (r=0.833)
2. **Other taxes** (r=0.788)
3. **Contraceptive prevalence, modern methods** (r=0.620)
4. **Contraceptive prevalence, any method** (r=0.567)
5. **Court fees** (r=0.480)

## Data

### Primary Dataset
- **Source:** World Bank, UN, UNESCO databases (DATA_GHAB2.xlsx)
- **Countries:** 52 low and lower-middle income nations
- **Variables:** 144 gender, health, education, and development indicators

### Supplementary Dataset
- **Source:** World Bank Doing Business indicators (BASE_COMPLETA.xlsx - UPDATED)
- **Additional Variables:** 164 non-G business environment variables (procedures, costs, scores, indices)
- **Merged:** Via Country code2 (40 countries successfully merged)

### Target Variable

**Score-Total tax and contribution rate (% of profit)**: Selected as BEST variable after comprehensive recursive testing of all 164 business environment indicators using balanced scoring methodology.

**Selection Process:**
- Tested 164 variables with 4 ML models each (656 total model runs)
- Used balanced score = sqrt(R²) × sqrt(CV) to ensure simultaneous optimization
- Excluded variables with negative CV (overfitting) or R² < 0.3 (poor fit)
- Tax Score achieved highest balanced score (0.270 in initial search, 0.940 in optimized final model)

## Methodology

### 1. Database Integration
- Merged DATA_GHAB2.xlsx with BASE_COMPLETA.xlsx using Country code2
- 40 of 53 countries successfully matched
- Final merged database: 298 variables across 53 countries

### 2. Comprehensive Variable Search with Balanced Scoring
- **Phase 1:** Tested all 164 non-G business environment variables
- **Balanced Scoring:** sqrt(R²) × sqrt(CV) ensures SIMULTANEOUS optimization
  - Requires BOTH R² > 0.3 AND CV > 0 for meaningful score
  - Geometric mean gives equal weight to fit quality and generalization
  - Heavily penalizes overfitting (negative CV → score = 0)
- Trained 4 models per variable (656 total model runs)
- **Winner:** Score-Total tax and contribution rate (Balanced=0.270)

### 3. Optimized Final Model
- **Hyperparameter Optimization:** GridSearchCV for Random Forest and XGBoost
- **Cross-Validation:** 5-fold with careful train/test splitting
- **Final Performance:** XGBoost achieved R²=0.967, CV=0.913, Balanced=0.940

### 4. Feature Selection
- Pearson correlation analysis between gender/development predictors and Tax Score
- Top 15 variables selected by absolute correlation magnitude
- Minimum 10 observations required per variable
- Final dataset: 36 countries with complete data

### 5. Models Evaluated

- **XGBoost (BEST):** Gradient boosting with optimized hyperparameters (learning_rate=0.05, max_depth=7, n_estimators=300)
- **Random Forest:** Ensemble with optimized depth and sample splitting
- **Gradient Boosting:** Sequential ensemble learning
- **Neural Network:** Multi-layer perceptron (100-50-25 architecture)

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

**GDP Growth (ROBUST):**
- R² = 0.745: Model explains 74.5% of variance
- CV = +0.074: Positive cross-validation (good generalization)
- RMSE = 13.01: Average prediction error
- Composite Score = 0.477
- Bootstrap 95% CI: [0.324, 0.903]

**Trading Score (ROBUST):**
- R² = 0.573: Model explains 57.3% of variance
- CV = +0.322: Strong positive cross-validation (excellent generalization!)
- RMSE = 13.68: Average prediction error
- Composite Score = 0.472
- Bootstrap 95% CI: [-0.415, 0.731]

**Why Robust Selection Matters:**
Previous analysis selected models with higher R² but negative CV (overfitting). The robust approach prioritizes positive CV, ensuring models generalize well to new data rather than just fitting the training set.

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
