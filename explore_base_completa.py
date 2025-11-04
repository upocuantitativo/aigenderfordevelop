#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Explore BASE_COMPLETA.xlsx and identify all G variables
"""

import pandas as pd

# Load BASE_COMPLETA
df_base = pd.read_excel('BASE_COMPLETA.xlsx')
print(f"BASE_COMPLETA shape: {df_base.shape}")
print(f"\nColumns: {df_base.columns.tolist()}")

# Identify G variables
g_cols = [c for c in df_base.columns if c.startswith('G_')]
print(f"\n{'='*80}")
print(f"Total G variables: {len(g_cols)}")
print(f"{'='*80}")

for i, col in enumerate(g_cols, 1):
    print(f"{i:2d}. {col}")

# Check Country code column
print(f"\n{'='*80}")
print("Country code sample:")
print(df_base['Country code'].head(10))

# Load DATA_GHAB2 to see how to merge
df_ghab = pd.read_excel('DATA_GHAB2.xlsx')
print(f"\n{'='*80}")
print(f"DATA_GHAB2 shape: {df_ghab.shape}")
print(f"\nCountry column name in DATA_GHAB2:")
country_cols = [c for c in df_ghab.columns if 'ountry' in c.lower() or 'code' in c.lower()]
print(country_cols)
if country_cols:
    print(f"\nSample from {country_cols[0]}:")
    print(df_ghab[country_cols[0]].head(10))
