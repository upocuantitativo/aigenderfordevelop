# AI for Development - Advanced ML Analysis with Dual Targets

## 🌍 Overview

This project provides state-of-the-art machine learning analysis for understanding the relationship between gender indicators and economic development in low and lower-middle income countries. Using recursive optimization and quantum computing, we identify the best predictive models for business density and GDP growth.

## 🎯 Dual Target Analysis

This project analyzes TWO key economic indicators:

1. **New_Business_Density** - Entrepreneurship and business creation rate
2. **G_GPD_PCAP_SLOPE** - GDP per capita growth trajectory

## 🚀 Advanced Features

### 1. Recursive Model Optimization
- **Intelligent Selection**: Trains all models, ranks them, and recursively optimizes only the top 3
- **Smart Visualization**: Generates detailed plots ONLY for winning models
- **Progressive Refinement**: 3 rounds of hyperparameter optimization for each top model
- **Composite Scoring**: Ranks models by R² + cross-validation robustness

### 2. Advanced ML Algorithms
- **Random Forest** with recursive hyperparameter tuning
- **XGBoost** gradient boosting with adaptive learning
- **Neural Networks** (sklearn MLPRegressor)
- **Deep Neural Networks** (TensorFlow/Keras with regularization)
- **Quantum Machine Learning** (Qiskit VQR - Variational Quantum Regressor)

### 3. Comprehensive Robustness Analysis
- K-Fold Cross-Validation (5 folds)
- Bootstrap Confidence Intervals (95%)
- Residual Analysis & Normality Tests
- Permutation Feature Importance
- Q-Q Plots for residual distribution
- Learning curves and validation metrics

### 4. Interactive Policy Projection Models
- Real-time GDP impact projections
- Country-specific data loading
- Policy recommendations based on top models

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/upocuantitativo/aidevelopment.git
cd aidevelopment
```

2. **Install dependencies**:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow qiskit qiskit-machine-learning plotly openpyxl scipy
```

3. **Run the recursive optimizer**:
```bash
python recursive_optimizer.py
```

### View Results

Open the final dashboard:
```
resultados/modelos_avanzados/best_models_dashboard.html
```

This shows ONLY the top 3 models after recursive optimization for each target variable.

## 📊 Data & Variables

### Dataset
- **File**: `DATA_GHAB2.xlsx`
- **Countries**: 52 low and lower-middle income countries
- **Variables**: 131 gender and development indicators
- **Categories**: Cultural, Demographic, Health, Education, Labour

### Target Variables
1. **New_Business_Density**: New business registrations per 1,000 people (ages 15-64)
2. **G_GPD_PCAP_SLOPE**: Trend line slope of GDP per capita growth

### Top Predictors (Auto-selected)
The recursive optimizer automatically identifies the top 15 most correlated variables for each target using Pearson correlation analysis.

## 🏆 Model Performance

### Recursive Optimization Results
- **Models Trained**: 5 algorithms (Random Forest, XGBoost, Neural Networks, Deep NN, Quantum VQR)
- **Top Models Selected**: Best 3 models per target variable
- **Optimization Rounds**: 3 recursive improvement cycles
- **Validation**: K-fold CV, Bootstrap CI, Residual Analysis
- **Best R² Achieved**: See dashboard for current results

## 🎯 How to Use Policy Projections

1. Select a country from the dropdown or use custom values
2. Adjust the policy variable sliders based on your scenarios
3. View real-time impact on projected GDP growth
4. Review policy recommendations tailored to your inputs

## 📁 Project Structure

```
aidevelopment/
├── DATA_GHAB2.xlsx                     # Main dataset (131 variables, 52 countries)
├── recursive_optimizer.py              # Main recursive optimization script
├── advanced_ml_analysis.py             # Advanced ML analyzer base class
├── resultados/                         # Analysis results
│   ├── modelos_avanzados/             # Best models (TOP 3 only)
│   │   ├── best_models_dashboard.html # Main results dashboard
│   │   ├── top_models_*.csv           # Performance metrics
│   │   └── top_models_comparison_*.html
│   ├── validacion/                    # Validation plots (top models only)
│   │   └── validation_*.html          # Residuals, Q-Q, bootstrap plots
│   └── robustez/                      # Robustness analysis
│       └── robustness_summary_*.csv
├── index.html                         # Legacy interface
└── README.md
```

## 🔬 Methodology

### Recursive Optimization Algorithm

```
1. INITIAL TRAINING
   - Train 5 models: Random Forest, XGBoost, Neural Network, Deep NN, Quantum VQR
   - Calculate R², RMSE, MAE for each

2. RANKING & SELECTION
   - Compute composite score = 0.6×R² + 0.4×CV_mean - 0.2×CV_std
   - Select TOP 3 models

3. RECURSIVE OPTIMIZATION (3 rounds)
   For each top model:
     Round 1: Optimize primary hyperparameters
     Round 2: Optimize secondary hyperparameters
     Round 3: Optimize tertiary hyperparameters
   → Keep model if improved, else stop

4. FINAL RANKING
   - Re-rank optimized models
   - Select final TOP 3

5. VISUALIZATION
   - Generate plots ONLY for final top 3 models
   - Create robustness analysis
   - Build interactive dashboard
```

### Robustness Validation
1. **K-Fold Cross-Validation**: 5-fold CV with shuffling
2. **Bootstrap Analysis**: 100 bootstrap samples for 95% CI
3. **Residual Analysis**: Normality tests (Shapiro-Wilk)
4. **Permutation Tests**: Feature importance validation
5. **Q-Q Plots**: Residual distribution analysis

### Machine Learning Models
- **Random Forest**: Recursive tuning of n_estimators, max_depth, min_samples
- **XGBoost**: Adaptive learning_rate, max_depth, subsample
- **Neural Networks**: Layer optimization, dropout tuning
- **Deep Neural Networks**: Architecture search, regularization
- **Quantum VQR**: Variational quantum circuits with ZZ feature maps

## 🎯 Key Insights

### Model Performance Hierarchy
The recursive optimizer automatically identifies which algorithms work best for:
- **New_Business_Density prediction**
- **G_GPD_PCAP_SLOPE prediction**

Results vary by target - see dashboard for current winners!

### Why Recursive Optimization?
- **Efficiency**: Only optimizes models that show promise
- **Clarity**: Shows only the best results, not cluttered with poor performers
- **Depth**: 3 rounds of refinement ensure optimal performance
- **Robustness**: Composite scoring prevents overfitting

## 📚 Data Sources

All country data is sourced from:
- **World Bank Open Data** (2023-2024 latest indicators)
- **UN Population Division** (World Population Prospects 2024)
- **UNESCO Institute for Statistics** (Education data)

See [DATA_SOURCES.md](DATA_SOURCES.md) for complete indicator codes, validation process, and update schedule.

### Recent Data Updates (October 2024)
- **Sudan**: Updated fertility (4.32), GDP growth (-13.5%), and life expectancy (70 years)
- All data verified against World Bank official indicators
- Conflict-affected countries may have data gaps (noted in documentation)

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+**: Main programming language
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Traditional ML algorithms and validation
- **XGBoost**: Gradient boosting framework
- **TensorFlow/Keras**: Deep learning models
- **Qiskit**: Quantum machine learning
- **Plotly**: Interactive visualizations
- **HTML/CSS/JavaScript**: Web interface

### Dependencies
```bash
pip install pandas numpy scikit-learn xgboost tensorflow qiskit qiskit-machine-learning plotly openpyxl scipy
```

### Hardware Requirements
- **Minimum**: 4GB RAM, 2 CPU cores
- **Recommended**: 8GB+ RAM, 4+ CPU cores for faster optimization
- **Quantum**: Runs on classical simulators (no quantum hardware needed)

## 📝 License

This project is developed for academic and research purposes.

## 👥 Contributors

Development team focused on gender equality and economic development research.

## 🔗 Resources

- [World Bank Gender Data](https://data.worldbank.org/topic/gender)
- [UN Women Data Hub](https://data.unwomen.org/)
- [UNDP Gender Development Index](https://hdr.undp.org/data-center/thematic-composite-indices/gender-development-index)

---

**Note**: This is an analytical tool for research purposes. Policy decisions should consider multiple factors beyond the variables included in this model.
