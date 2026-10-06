# Jiangsu Short-Term GHI Probabilistic Forecasting and GFS Post-Processing

This repository implements a reproducible numerical weather prediction post-processing framework for short-term Global Horizontal Irradiance (GHI) forecasting over Jiangsu, China.

The primary objective is to correct GFS forecasts at 24, 48, and 72 h lead times using statistical, machine-learning, and deep-learning methods, while supporting probabilistic prediction and province-wide spatial generalization.

The entire workflow is configuration-driven and covers data acquisition, preprocessing, feature construction, rolling validation, model fitting, calibration, evaluation, and reproducible artifact management.

---

## 1. Data system

### 1.1 Forecast data

Historical meteorological predictors are derived from GFS products distributed through Open-Meteo.

The currently materialized dataset is organized for the registered 20 sites:

```text
20-site Jiangsu registry
    ↓
site × target time × lead time
    ↓
GFS meteorological predictors
```

The registered forecast lead times are:

```text
24 h
48 h
72 h
```

Previous Runs fields represent forecasts retained at the corresponding offset before each valid time. These lead offsets define forecast availability and causality; they are not assumed to be verified native GFS initialization timestamps.

### 1.2 Irradiance reference

The active reference is satellite-derived GHI from JMA/JAXA Himawari:

```text
Himawari shortwave radiation
→ satellite-derived GHI reference
```

Satellite retrievals are neither NWP forecasts nor reanalysis products. In the
current dataset they are matched to the same 20 requested locations as the
Previous Runs fields.

High-quality ground pyranometer observations can be used as an additional independent point-scale validation source.

### 1.3 Data layers

Local research data follow a single staged structure:

```text
data/
├── raw/
├── clean/<clean_hash>/
├── features/<feature_hash>/
├── registry/
├── catalog.json
└── data_inventory.csv
```

- `raw`: immutable acquired source material;
- `clean`: temporally, spatially, and physically normalized data;
- `features`: model-specific training inputs;
- `registry`: stable spatial and site registries;
- `catalog.json`: dataset registration and resolution;
- `data_inventory.csv`: local data inventory.

Research data products are traced through receipts, content hashes, and dependency fingerprints.

---

## 2. Experimental protocol

The study period is separated into:

```text
Development
2024-02-01 — 2025-08-31

Final test
2025-09-01 — 2026-08-31
```

Model development uses nested purged rolling-origin validation.

| Setting | Value |
|---|---:|
| Outer rolling folds | 5 |
| Inner folds per outer fold | 3 |
| Maximum sequence lookback | 168 h |
| Maximum forecast lead | 72 h |
| Canonical purge | 240 h |
| Purge sensitivity | 7 / 10 / 14 d |

Rolling validation is part of model selection rather than an additional post-selection test.

The final test period remains isolated from iterative model development until the experimental configuration and model-selection process are frozen.

---

## 3. Feature system

The project separates the **raw acquisition contract** from **model-specific feature contracts**.

The acquisition layer preserves meteorological information with long-term research value. Each model family may then construct the feature representation most appropriate to its learning mechanism.

Candidate information includes:

- radiation forecasts;
- cloud state and cloud evolution;
- temperature, humidity, and dew point;
- wind, pressure, and precipitation;
- clear-sky irradiance and transmissivity-related features;
- solar geometry;
- cyclic temporal variables;
- prior forecasts available at issue time;
- spatial GFS-grid coordinates.

Different model families are not required to use the same feature dimensionality.

Typical strategies are:

```text
Linear / Ridge
→ compact tabular representation with regularization
  and dimensionality reduction when appropriate

LightGBM / XGBoost
→ richer engineered tabular features

Deep Learning
→ richer meteorological inputs
  + temporal sequences
  + spatial neighbourhood structure
```

All model inputs must be available at forecast issue time.

Location identities, target variables, and future observations are forbidden as predictive shortcuts. Data-dependent preprocessing is fitted only within the applicable training fold.

---

## 4. Models

The repository contains several levels of forecasting methods.

### Baselines and statistical models

```text
climatology
persistence
smart_persistence
optimal_convex
raw_gfs
bias_correction
linear_mos
ridge_mos
```

### Tree-based machine learning

```text
lgbm
xgboost
```

### Deep-learning architectures

```text
mlp
cnn
tcn
lstm
transformer
autoformer
informer
fedformer
itransformer
patchtst
dlinear
timesnet
tsmixer
pinn
```

Model families may use different feature contracts, while sharing the same temporal protocol, causal constraints, and evaluation rules.

---

## 5. Probabilistic forecasting and evaluation

The registered quantile grid is:

```text
0.05
0.10
0.25
0.50
0.75
0.90
0.95
```

The primary model-selection metric is:

```text
mean pinball loss
```

Deterministic metrics include:

```text
MAE
RMSE
Bias
R²
RMSE skill
```

Probabilistic metrics include:

```text
mean pinball loss
coverage
interval width
```

Probabilistic outputs may subsequently be calibrated on a temporally later and independent calibration block using conformal methods.

Feature-group ablation, grouped permutation importance, and SHAP are used for interpretation rather than model selection.

---

## 6. Spatial organization and generalization

The current 20-site data are organized independently of later experimental sampling choices:

```text
receipt-backed 20-site dataset
        ↓
experimental design
        ├─ training locations
        ├─ spatial hold-out locations
        └─ regional extrapolation locations
```

Province-wide GFS025 acquisition is not part of the current repository state and
must not begin without an explicit instruction.

Spatial evaluation is organized at three levels:

1. stratified spatial hold-out;
2. province-wide unseen-grid evaluation;
3. regional extrapolation stress testing.

---

## 7. Workflow

The unified workflow is:

```text
configuration
    ↓
raw data
    ↓
clean data
    ↓
model-specific features
    ↓
rolling splits
    ↓
hyperparameter tuning
    ↓
model fitting
    ↓
prediction
    ↓
calibration
    ↓
evaluation
    ↓
analysis / figures
```

Scientific configuration is defined by:

```text
config/protocol.yaml
config/data.yaml
config/features.yaml
config/sites.yaml
config/models.yaml
config/experiments.yaml
```

These files are the authoritative source of experimental settings.

The sole workflow entry point is:

```text
python -m nwp
```

---

## 8. Repository layout

```text
config/          scientific and experimental configuration
src/nwp/         Python implementation
tests/           unit, integration, and end-to-end tests
docs/            protocol and technical documentation

data/            local research data
outputs/         isolated experiment runs and results
```

Research data and run outputs are managed separately from source code.

---

## 9. Installation

Base package:

```bash
python -m pip install -e .
```

CPU modeling:

```bash
python -m pip install -e ".[cpu]"
```

Deep-learning development:

```bash
python -m pip install -e ".[cpu,deep]"
```

Full development and testing environment:

```bash
python -m pip install -e ".[cpu,deep,dev]"
```

---

## 10. Usage

Validate a profile:

```bash
python -m nwp validate --profile nanjing_cpu_diagnostic
```

Inventory local data:

```bash
python -m nwp data inventory
```

Run through split construction:

```bash
python -m nwp run \
  --profile nanjing_cpu_diagnostic \
  --to-stage splits
```

Explicitly enable model execution:

```bash
python -m nwp run \
  --profile nanjing_cpu_diagnostic \
  --execute-model-stages
```

Evaluate an existing run:

```bash
python -m nwp evaluate --run-id <run-id>
```

Generate figures:

```bash
python -m nwp figures --run-id <run-id>
```

Inspect registered profiles and runtime paths:

```bash
python -m nwp status
```

Each experiment is isolated under:

```text
outputs/<execution>/<run_id>/
```

A run records resolved configuration, provenance, stage receipts, artifact manifests, fitted models, predictions, evaluation outputs, and figures required for reproducibility.
