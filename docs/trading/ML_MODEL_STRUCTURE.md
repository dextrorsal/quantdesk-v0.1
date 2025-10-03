# 🧠 ML Model Structure Guide

## Overview
This document explains the correct organization of the ML model components to prevent confusion between **features** and **strategies**.

## Directory Structure

```
src/ml/
├── features/                    # Technical Indicators (FEATURES)
│   ├── rsi.py                  # RSI indicator
│   ├── adx.py                  # ADX indicator  
│   ├── cci.py                  # CCI indicator
│   ├── wave_trend.py           # WaveTrend indicator
│   ├── chandelier_exit.py      # Chandelier Exit indicator (13KB, 346 lines)
│   └── base_torch_indicator.py # Base class for indicators
├── models/
│   └── strategy/               # Trading Strategies (STRATEGIES)
│       ├── logistic_regression_torch.py  # Logistic regression strategy
│       ├── lorentzian_classifier.py      # Lorentzian classifier strategy
│       ├── chandelier_exit.py            # Risk management strategy (8KB, 253 lines)
│       └── lag_based_strategy.py         # Lag-based strategy
└── indicators/                 # Base indicator foundations
    └── base_torch_indicator.py
```

## Key Distinction: Features vs Strategies

### 🎯 Features (`src/ml/features/`)
**Purpose**: Technical indicators used as **input features** for ML models
- **What they do**: Calculate technical indicator values (RSI, ADX, CCI, etc.)
- **How they're used**: As input features in ML model training and prediction
- **Example**: RSI value of 65, ADX value of 25, etc.

### 🎯 Strategies (`src/ml/models/strategy/`)
**Purpose**: Trading strategies that generate buy/sell signals
- **What they do**: Use features to generate trading decisions
- **How they're used**: For actual trading decisions and risk management
- **Example**: Buy when RSI < 30, sell when RSI > 70

## Chandelier Exit: The Confusion Point

### ❌ WRONG: Using Chandelier Exit as a Feature
```python
# DON'T DO THIS in ML models
from src.ml.features.chandelier_exit import ChandelierExitIndicator

# Using it as a feature input
chandelier = ChandelierExitIndicator()
ce_feat = chandelier.calculate_signals(df)["long"]  # ❌ WRONG
```

### ✅ CORRECT: Using Chandelier Exit for Risk Management
```python
# DO THIS for risk management
from src.ml.models.strategy.chandelier_exit import ChandelierExit

# Using it for stop losses and position management
chandelier = ChandelierExit()
stops = chandelier.calculate_signals(df)  # ✅ CORRECT
```

## File Size Guidelines

### Features Folder
- **chandelier_exit.py**: ~13KB, ~346 lines (technical indicator implementation)
- **rsi.py**: ~11KB, ~349 lines
- **adx.py**: ~7.8KB, ~247 lines
- **cci.py**: ~8.1KB, ~261 lines
- **wave_trend.py**: ~10KB, ~322 lines

### Strategy Folder  
- **chandelier_exit.py**: ~8KB, ~253 lines (risk management strategy)
- **logistic_regression_torch.py**: ~18KB, ~536 lines
- **lorentzian_classifier.py**: ~21KB, ~603 lines

## Correct Usage Patterns

### 1. ML Models Should Use Features
```python
# ✅ CORRECT: ML models use technical indicators as features
from src.ml.features.rsi import RSIIndicator
from src.ml.features.adx import ADXIndicator
from src.ml.features.cci import CCIIndicator
from src.ml.features.wave_trend import WaveTrendIndicator

# Calculate features for ML input
rsi_feat = rsi.calculate_signals(df)["rsi"]
adx_feat = adx.calculate_signals(df)["adx"]
cci_feat = cci.calculate_signals(df)["cci"]
wt_feat = wavetrend.calculate_signals(df)["wt1"]
```

### 2. Risk Management Should Use Strategies
```python
# ✅ CORRECT: Risk management uses strategy implementations
from src.ml.models.strategy.chandelier_exit import ChandelierExit

# Calculate stop losses and risk management signals
chandelier = ChandelierExit()
risk_signals = chandelier.calculate_signals(df)
```

## Common Mistakes to Avoid

### ❌ Mistake 1: Importing Chandelier Exit as Feature in ML Models
```python
# WRONG - Don't import from features in strategy models
from src.ml.features.chandelier_exit import ChandelierExitIndicator
```

### ❌ Mistake 2: Using Chandelier Exit Values as ML Features
```python
# WRONG - Don't use chandelier exit as input feature
ce_feat = chandelier.calculate_signals(df)["long"]
features = torch.stack([rsi_feat, adx_feat, cci_feat, ce_feat])  # ❌
```

### ❌ Mistake 3: Confusing File Sizes
- If `chandelier_exit.py` in features is >10KB, it's likely the indicator
- If `chandelier_exit.py` in strategy is <10KB, it's likely the strategy wrapper

## Validation Checklist

Before committing changes, verify:

- [ ] ML models only import from `src/ml/features/` for technical indicators
- [ ] ML models do NOT import chandelier exit from features
- [ ] Risk management imports from `src/ml/models/strategy/`
- [ ] File sizes match expected patterns
- [ ] No circular imports between features and strategies

## Recovery Steps

If confusion occurs again:

1. **Check file sizes**: Features should be larger than strategy wrappers
2. **Check imports**: ML models should not import chandelier exit from features
3. **Restore from backup**: Use `/home/dex/ML-MODEL/src/` as backup reference
4. **Follow this guide**: Use the patterns documented above

## Related Documentation

- [ML Model Architecture](ML_MODEL.md) - Overall ML model design
- [Technical Indicators](INDICATORS.md) - Indicator implementations
- [Technical Strategy](TECHNICAL_STRATEGY.md) - Strategy implementations 