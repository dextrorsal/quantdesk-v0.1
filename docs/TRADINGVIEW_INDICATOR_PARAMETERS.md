# 🎯 TradingView Indicator Parameters Reference

*Generated from your project documentation and configuration files*

This document contains the exact parameters for your 3 indicators to ensure your Python implementations match TradingView.

## 📊 **Lorentzian Classifier Parameters**

Based on `docs/INDICATORS.md` and code analysis:

### Core Parameters
```
n_neighbors: 8          # Number of nearest neighbors
bar_spacing: 4          # Chronological spacing between samples  
threshold: 0.1          # Signal generation threshold
lookback_bars: 50       # Historical data window
```

### Feature Calculation Parameters
```
RSI:
  period: 14
  overbought: 70.0
  oversold: 30.0

Wave Trend:
  channel_length: 10    # ESA calculation period
  avg_length: 11        # Wave trend line averaging
  signal_line: 4        # SMA period for WT2

CCI:
  length: 20            # CCI calculation period
  smooth: 1             # Additional smoothing (1 = no smoothing)
  
ADX:
  period: 14            # ADX calculation period
  threshold: 20.0       # Trend strength threshold
```

### Signal Zones
```
Wave Trend Zones:
  Overbought: > 60
  Oversold: < -60
  Extreme Overbought: > 80
  Extreme Oversold: < -80

CCI Zones:
  Extreme Overbought: > 200 (Strong Sell)
  Overbought: 100 to 200 (Weak Sell)
  Neutral: -100 to 100 (No Signal)
  Oversold: -200 to -100 (Weak Buy)
  Extreme Oversold: < -200 (Strong Buy)
```

## 📈 **Logistic Regression Parameters**

Based on `docs/TRADING_SIGNALS_FIX.md` and testing results:

### Model Configuration
```
lookback: 3             # Feature lookback period
confidence_threshold: 0.3  # Signal generation threshold

Model Output Ranges:
5m model: 0.26-0.40 range
15m model: 0.12-0.35 range
```

### Signal Strategies (from your testing)
```
TradingView-Style:
  - confidence_threshold: 0.3
  - 60 signals over 30 days
  - 63% win rate

Combined Approach (Most Robust):
  - combined_threshold: 0.25
  - 133 signals over 30 days  
  - 59% win rate
  - Best for 60-day performance

Strong Signals:
  - Higher confidence threshold
  - 46 signals over 30 days
  - 70% win rate
```

### Multi-Timeframe Weighting
```
5m model weight: 70%
15m model weight: 30%
```

## 🕯️ **Chandelier Exit Parameters**

Based on code analysis and standard implementations:

### Core Parameters
```
atr_period: 22          # ATR calculation period (from docs/TECHNICAL_STRATEGY.md)
atr_multiplier: 3.0     # ATR multiplier for stop calculation
```

**Note**: Your code shows `atr_period: 3` and `atr_multiplier: 2.0`, but standard Chandelier Exit uses:
- ATR Period: 22
- Multiplier: 3.0

### Calculation Method
```
Long Stop = Highest High (over ATR period) - (ATR * Multiplier)
Short Stop = Lowest Low (over ATR period) + (ATR * Multiplier)
```

### Signal Generation
```
Long Signal: Price closes above Long Stop
Short Signal: Price closes below Short Stop
Exit Signal: Stop level crossover
```

## 🔧 **TradingView Setup Instructions**

### 1. Lorentzian Classifier Settings
```
Neighbors (k): 8
Lookback Bars: 50
Bar Spacing: 4
Threshold: 0.1

Feature Settings:
- RSI Period: 14
- Wave Trend Channel: 10
- Wave Trend Average: 11  
- CCI Period: 20
- ADX Period: 14
```

### 2. Logistic Regression Settings
```
Lookback Period: 3
Confidence Threshold: 0.3 (for TradingView-style signals)

If using multi-timeframe:
- 5m weight: 0.70
- 15m weight: 0.30
```

### 3. Chandelier Exit Settings
```
ATR Period: 22
ATR Multiplier: 3.0
```

## ⚠️ **Parameter Discrepancies Found**

### Chandelier Exit Mismatch
Your Python code currently uses:
- `atr_period: 3` 
- `atr_multiplier: 2.0`

But standard/TradingView Chandelier Exit uses:
- `atr_period: 22`
- `atr_multiplier: 3.0`

**Recommendation**: Update your Python implementation to match TradingView standard parameters.

### Logistic Regression Thresholds
Your documentation shows you've optimized thresholds based on actual model output:
- Original expectation: >0.65 confidence
- Actual model range: 0.12-0.40
- Optimized threshold: 0.3

Make sure TradingView logistic implementation uses the same 0.3 threshold.

## 🎯 **Verification Checklist**

### Step 1: Parameter Verification
- [ ] Lorentzian: 8 neighbors, 50 lookback, threshold 0.1
- [ ] Logistic: Lookback 3, threshold 0.3  
- [ ] Chandelier: ATR 22, multiplier 3.0

### Step 2: Feature Parameter Verification
- [ ] RSI: 14 period
- [ ] Wave Trend: 10 channel, 11 average
- [ ] CCI: 20 period
- [ ] ADX: 14 period, 20.0 threshold

### Step 3: Signal Strategy Verification
- [ ] Test with recent SOL 1H data (July 2025)
- [ ] Compare signal frequency and timing
- [ ] Verify combined signal weighting (70/30)

## 📝 **Configuration Files to Update**

Based on this analysis, you should update:

1. **Chandelier Exit Parameters**: 
   - Update `src/ml/models/strategy/chandelier_exit.py` 
   - Change ATR period from 3 to 22
   - Change multiplier from 2.0 to 3.0

2. **Create Parameter Config File**:
   ```json
   {
     "lorentzian": {
       "n_neighbors": 8,
       "lookback_bars": 50,
       "threshold": 0.1,
       "features": {
         "rsi_period": 14,
         "wt_channel": 10,
         "wt_average": 11,
         "cci_period": 20,
         "adx_period": 14
       }
     },
     "logistic": {
       "lookback": 3,
       "confidence_threshold": 0.3,
       "combined_threshold": 0.25
     },
     "chandelier": {
       "atr_period": 22,
       "atr_multiplier": 3.0
     }
   }
   ```

---

*This parameter reference ensures your Python indicators match TradingView exactly. Focus on the parameter discrepancies found - especially the Chandelier Exit settings.*