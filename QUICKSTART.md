# 🚀 Quick Start Guide

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/upocuantitativo/aidevelopment.git
cd aidevelopment
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**Alternative** (if you encounter issues):
```bash
pip install pandas numpy scikit-learn xgboost tensorflow qiskit qiskit-machine-learning plotly openpyxl scipy
```

## Running the Analysis

### Simple method (Recommended)
```bash
python run_analysis.py
```

### Advanced method (Full control)
```bash
python recursive_optimizer.py
```

## What happens during execution?

1. **Data Loading** (1 min)
   - Loads DATA_GHAB2.xlsx (52 countries, 131 variables)
   - Validates target variables

2. **For each target variable** (New_Business_Density, G_GPD_PCAP_SLOPE):

   a. **Initial Training** (3-5 min)
      - Random Forest
      - XGBoost
      - Neural Network (sklearn)
      - Deep Neural Network (TensorFlow)
      - Quantum VQR (Qiskit)

   b. **Ranking** (<1 min)
      - Computes composite scores
      - Selects TOP 3 models

   c. **Recursive Optimization** (5-15 min per model)
      - Round 1: Primary hyperparameters
      - Round 2: Secondary hyperparameters
      - Round 3: Tertiary hyperparameters
      - Stops if no improvement

   d. **Robustness Analysis** (2-3 min per model)
      - K-Fold Cross-Validation
      - Bootstrap CI
      - Residual analysis
      - Permutation tests

   e. **Visualization** (1-2 min)
      - Validation plots for top 3 models only
      - Interactive dashboards

3. **Final Dashboard Creation** (<1 min)

**Total time**: 10-30 minutes (varies by hardware)

## Viewing Results

After completion, open:
```
resultados/modelos_avanzados/best_models_dashboard.html
```

This shows:
- 🥇 **Podium** with top 3 models for each target
- 📊 **Detailed metrics** (R², RMSE, MAE, CV scores)
- 📈 **Interactive comparisons**
- 🔗 **Links to validation plots**

## File Structure

```
aidevelopment/
├── DATA_GHAB2.xlsx                 # Input data
├── run_analysis.py                 # Quick start script
├── recursive_optimizer.py          # Main optimizer
├── advanced_ml_analysis.py         # ML analyzer base
├── requirements.txt                # Dependencies
└── resultados/                     # Output folder
    ├── modelos_avanzados/          # Best models dashboard
    │   ├── best_models_dashboard.html  ← MAIN RESULT
    │   ├── top_models_*.csv
    │   └── top_models_comparison_*.html
    ├── validacion/                 # Validation plots (top 3)
    │   └── validation_*.html
    └── robustez/                   # Robustness analysis
        └── robustness_summary_*.csv
```

## Understanding Results

### Model Rankings
Models are ranked by **composite score**:
```
Composite = 0.6 × R² + 0.4 × CV_mean - 0.2 × CV_std
```

This balances:
- **Accuracy** (R²)
- **Robustness** (CV mean)
- **Stability** (penalizes high variance)

### Validation Plots
Each top model has 6 validation plots:
1. **Predictions vs Actual**: How well predictions match reality
2. **Residual Plot**: Error distribution across predictions
3. **Residual Histogram**: Check for normality
4. **Q-Q Plot**: Theoretical vs actual residual distribution
5. **Bootstrap Distribution**: Confidence in R² score
6. **Cross-Validation Scores**: Consistency across folds

### Interpreting Metrics
- **R²**: 0-1, higher is better (0.75+ is excellent)
- **RMSE**: Lower is better (in target units)
- **MAE**: Average absolute error (in target units)
- **CV Mean ± Std**: Cross-validation consistency

## Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
```

### TensorFlow installation issues
```bash
pip install tensorflow --upgrade
```

### Qiskit installation issues
```bash
pip install qiskit qiskit-machine-learning --upgrade
```

### Out of memory
- Close other applications
- Reduce `top_n` in recursive_optimizer.py (line 1018):
  ```python
  optimizer = RecursiveModelOptimizer(top_n=2, optimization_rounds=2)
  ```

### Too slow
- Reduce optimization rounds:
  ```python
  optimizer = RecursiveModelOptimizer(top_n=3, optimization_rounds=2)
  ```

## Customization

### Change number of top models
Edit `recursive_optimizer.py` line 1018:
```python
optimizer = RecursiveModelOptimizer(top_n=5, optimization_rounds=3)
```

### Change optimization depth
```python
optimizer = RecursiveModelOptimizer(top_n=3, optimization_rounds=5)
```

### Add custom model
Edit `advanced_ml_analysis.py` and add to `analyze_target()` method.

## Need Help?

- Check [README.md](README.md) for methodology details
- See [DATA_SOURCES.md](DATA_SOURCES.md) for data information
- Open an issue on GitHub: https://github.com/upocuantitativo/aidevelopment/issues

## Next Steps

After running the analysis:
1. Review the dashboard
2. Check validation plots for top models
3. Download CSV results for further analysis
4. Share best_models_dashboard.html with stakeholders
