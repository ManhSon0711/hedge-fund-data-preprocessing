# Hedge Fund Time-Series Forecasting — Data Preprocessing & Feature Selection

Data preprocessing and temporal-stability-based feature selection pipeline 
for a hedge fund time-series forecasting project.

## Context

This repository contains **my contribution** to a collaborative quantitative 
finance project. It focuses on data preparation and identifying features 
that remain predictive across different time periods — a critical step 
before feeding data into any downstream model.

My teammate built the transformer-based prediction model on top of the 
features selected here.

🔗 **Full project repo (transformer model):** 
https://github.com/NTT1602/Hedge-fund---Time-series-forecasting.git

## My Contributions

### 1. Missing Data Handling
Split features into two groups based on missingness ratio:
- **< 5% missing**: Imputed with column median (missingness assumed random)
- **≥ 5% missing**: Imputed with median **and** augmented with binary 
  missing-indicator flags (missingness potentially non-random and 
  informative for the target)

This approach preserves information that would otherwise be lost with 
naive imputation.

### 2. Categorical Encoding
- One-hot encoded `sub_category` and `horizon`
- Converted `code` and `sub_code` to pandas categorical dtype for memory 
  efficiency and downstream model compatibility

### 3. Memory Optimization
Cast all feature columns to `float32` to reduce memory footprint on the 
large dataset without meaningful precision loss for this use case.

### 4. Temporal Stability Feature Selection (core contribution)

The key challenge in quant ML is that features predictive in one period 
often degrade or invert in others. A feature that looks important on the 
full dataset may be leveraging only a single regime.

**Approach:**
1. Split the dataset into **5 consecutive time segments** based on `ts_index` 
   (each covering ~720 time steps)
2. For each segment:
   - Train a LightGBM regressor (400 estimators, learning rate 0.05) using 
     an 80/20 temporal train/validation split
   - Rank features by gain-based importance
   - Retain the top 40 features
3. **Stable features** = features that appear in the top 40 of **at least 
   3 out of 5 segments**

This selects features that are robustly predictive across multiple market 
periods rather than features that happen to work in a single regime.

**Result:** From 86 raw features, identified a reduced set of stable 
predictors for the downstream model.

## Tech Stack

- Python
- pandas, NumPy
- LightGBM
- scikit-learn
- PyArrow (for parquet I/O)
- matplotlib, seaborn

## Files

- `Data preprocessing.ipynb` — full preprocessing and feature selection 
  pipeline with analysis and code

## Key Design Decisions

- **Why missing flags only for high-missing columns?** Features missing 
  more than 5% of the time often have informative missingness patterns 
  (e.g., data only reported under certain conditions). Below that threshold, 
  the signal-to-noise of a flag column is typically not worth the added 
  dimensionality.

- **Why segment-based importance instead of full-data importance?** 
  Full-data gain importance is dominated by whichever regime contributes 
  most variance. Segmenting forces the selection to favor features that 
  generalize across time — a proxy for future-period robustness.

- **Why LightGBM for selection?** Fast on large tabular data, handles 
  mixed types natively, and gain-based importance is a well-established 
  proxy for predictive value in tree ensembles.

---

*Collaboration with [@NTT1602]— transformer 
model available in the [main project repo]((https://github.com/NTT1602/Hedge-fund---Time-series-forecasting.git)).*
