#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUICK START - RECURSIVE MODEL OPTIMIZER
Simple script to run the complete analysis

Just run: python run_analysis.py
"""

import sys
import os

print("""
╔═══════════════════════════════════════════════════════════════╗
║   AI FOR DEVELOPMENT - RECURSIVE MODEL OPTIMIZATION          ║
║   Dual Target Analysis with Quantum ML                       ║
╚═══════════════════════════════════════════════════════════════╝

📊 Analyzing:
   • New_Business_Density
   • G_GPD_PCAP_SLOPE

🤖 Models:
   • Random Forest
   • XGBoost
   • Neural Networks
   • Deep Neural Networks (TensorFlow)
   • Quantum VQR (Qiskit)

🎯 Process:
   1. Train all 5 models
   2. Rank by composite score
   3. Recursively optimize TOP 3
   4. Generate visualizations for winners only

⏱️  Estimated time: 10-30 minutes (depending on hardware)

""")

# Check if data file exists
if not os.path.exists('DATA_GHAB2.xlsx'):
    print("❌ ERROR: DATA_GHAB2.xlsx not found!")
    print("   Please ensure the data file is in the current directory.")
    sys.exit(1)

# Import and run
try:
    from recursive_optimizer import RecursiveModelOptimizer

    print("🚀 Starting recursive optimization...\n")

    # Create optimizer (top 3 models, 3 optimization rounds)
    optimizer = RecursiveModelOptimizer(top_n=3, optimization_rounds=3)

    # Run optimization
    success = optimizer.run_recursive_optimization()

    if success:
        print("\n" + "="*70)
        print("✅ SUCCESS! Analysis completed.")
        print("="*70)
        print("\n📂 View results:")
        print("   Open: resultados/modelos_avanzados/best_models_dashboard.html")
        print("\n📁 Additional outputs:")
        print("   • resultados/validacion/ - Validation plots (top models)")
        print("   • resultados/modelos_avanzados/ - Performance comparisons")
        print("\n🌐 Ready to upload to GitHub!")
        sys.exit(0)
    else:
        print("\n❌ Analysis completed with errors.")
        sys.exit(1)

except ImportError as e:
    print(f"\n❌ ERROR: Missing dependencies!")
    print(f"   {e}")
    print("\n💡 Install required packages:")
    print("   pip install pandas numpy scikit-learn xgboost tensorflow qiskit qiskit-machine-learning plotly openpyxl scipy")
    sys.exit(1)

except KeyboardInterrupt:
    print("\n\n⏹️  Analysis interrupted by user.")
    sys.exit(1)

except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
