# 🎯 Parameter Scaling Corrections: PineScript → PyTorch

*Critical corrections needed to match TradingView exactly*

## 🚨 **Immediate Corrections Required**

### **1. Chandelier Exit - MAJOR MISMATCH**
```python
# CURRENT (WRONG):
atr_period = 3
atr_multiplier = 2.0

# CORRECT (from TradingView):
atr_period = 22
atr_multiplier = 3.0
use_close_for_extremums = True  # Important: use close price for highs/lows
```

### **2. Lorentzian Classifier - Parameter Alignment**
```python
# CORRECT (from TradingView):
LORENTZIAN_CONFIG = {
    'neighbors_count': 8,
    'max_bars_back': 2000,
    'source': 'close',
    
    # Feature Engineering (5 features):
    'features': {
        'rsi_14_1': {'type': 'RSI', 'period': 14, 'smooth': 1},
        'wt_10_11': {'type': 'WT', 'channel': 10, 'average': 11},
        'cci_20_1': {'type': 'CCI', 'period': 20, 'smooth': 1},
        'adx_20_2': {'type': 'ADX', 'period': 20, 'smooth': 2},
        'rsi_9_1': {'type': 'RSI', 'period': 9, 'smooth': 1}
    },
    
    # Filters:
    'use_volatility_filter': True,
    'use_regime_filter': True,
    'regime_threshold': -0.1,
    'use_adx_filter': True,
    'adx_threshold': 20,
    
    # Kernel Settings:
    'trade_with_kernel': True,
    'lookback_window': 8,
    'relative_weighting': 8,
    'regression_level': 25
}
```

### **3. Logistic Regression - Learning Rate Scaling**
```python
# CRITICAL: PineScript learning rates don't directly translate
# TradingView: 0.0009 over 1000 iterations
# PyTorch equivalent analysis needed:

LOGISTIC_CONFIG = {
    'lookback_window': 3,  # Matches TradingView
    'normalization_lookback': 2,  # Matches TradingView
    'learning_rate': 0.001,  # May need adjustment (was 0.0009 in TV)
    'training_iterations': 1000,  # Matches TradingView
    'holding_period': 5,  # Matches TradingView
    
    # PyTorch-specific (not in TradingView):
    'optimizer': 'adam',  # vs SGD in TradingView?
    'batch_size': 32,
    'regularization': 0.01
}
```

## 🔍 **Scaling Differences Analysis**

### **A. Numerical Precision**
```python
# PineScript: 
# - Single precision floats
# - Built-in function implementations

# PyTorch:
# - Double precision available
# - Manual implementations
# - Different rounding behavior
```

### **B. Normalization Methods**
```python
# TradingView Logistic Regression:
def tv_normalize(values, lookback=2):
    """PineScript-style normalization"""
    recent_min = min(values[-lookback:])
    recent_max = max(values[-lookback:])
    return (current_value - recent_min) / (recent_max - recent_min)

# PyTorch Standard:
def pytorch_normalize(values):
    """StandardScaler equivalent"""
    return (values - values.mean()) / values.std()

# These produce DIFFERENT results!
```

### **C. ATR Calculation Differences**
```python
# TradingView Chandelier (PineScript):
# - Uses built-in ta.atr() function
# - Automatically handles close price for extremums
# - Period 22, multiplier 3.0

# Your PyTorch:
# - Custom ATR implementation
# - Period 3 (7x shorter window!)
# - Multiplier 2.0 (25% smaller stops)
# - Result: Completely different stop levels
```

## 🎯 **Action Plan**

### **Phase 1: Fix Critical Mismatches (Do This First)**
```python
# 1. Update Chandelier Exit parameters immediately:
chandelier_config = {
    'atr_period': 22,        # was 3
    'atr_multiplier': 3.0,   # was 2.0
    'use_close_extremums': True
}

# 2. Verify ATR calculation method matches TradingView
# 3. Test signals on recent data
```

### **Phase 2: Normalization Alignment**
```python
# Implement TradingView-style normalization for Logistic Regression:
class TVStyleNormalization:
    def __init__(self, lookback=2):
        self.lookback = lookback
    
    def normalize(self, values):
        normalized = []
        for i in range(len(values)):
            if i < self.lookback:
                normalized.append(0)  # or handle edge case
            else:
                window = values[i-self.lookback:i+1]
                min_val, max_val = min(window), max(window)
                if max_val == min_val:
                    normalized.append(0)
                else:
                    norm_val = (values[i] - min_val) / (max_val - min_val)
                    normalized.append(norm_val)
        return normalized
```

### **Phase 3: Learning Rate Calibration**
```python
# Test different learning rates to match TradingView behavior:
learning_rates_to_test = [0.0009, 0.001, 0.0015, 0.002]

# Compare convergence patterns and final model outputs
```

## ⚠️ **Why This Matters**

### **Impact of Current Mismatches:**
```
Chandelier Exit (Period 3 vs 22):
- Your stops are 7x more sensitive
- Completely different risk management
- May explain poor performance on shorter timeframes

Logistic Regression Normalization:
- Different feature scaling = different model behavior
- Learning rate mismatch = different convergence

Result: Your indicators produce different signals than TradingView
```

## 🧪 **Testing Strategy**

### **1. Parameter-by-Parameter Testing**
```python
# Test each correction individually:
test_chandelier_22_vs_3()    # Compare ATR periods
test_logistic_norm_methods() # Compare normalization
test_lorentzian_features()   # Verify feature calculations
```

### **2. Signal Comparison**
```python
# Compare signals before/after corrections:
old_signals = generate_signals_old_params()
new_signals = generate_signals_tv_params()
signal_diff_analysis(old_signals, new_signals)
```

### **3. TradingView Verification**
```python
# Key timestamps to check after corrections:
verification_dates = [
    '2025-07-15 14:00:00',
    '2025-07-16 09:00:00', 
    '2025-07-16 15:00:00'
]
```

## 🎯 **Expected Results After Corrections**

1. **Chandelier Exit**: Much smoother, less frequent signals
2. **Logistic Regression**: More stable predictions 
3. **Lorentzian**: Better feature alignment
4. **Overall**: Signals should closely match TradingView

---

*Priority: Fix Chandelier Exit first (biggest mismatch), then test signal alignment*