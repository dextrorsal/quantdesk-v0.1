# 📊 Custom Technical Indicators

*What is this doc?*  
This guide details all custom technical indicators used in the project, including their math, code, and how they fit into the trading system. It's for developers, quants, and anyone curious about the feature engineering side.

[ML Model](ML_MODEL.md) | [Technical Strategy](TECHNICAL_STRATEGY.md) | [Project README](../README.md)

## Table of Contents
1. [PyTorch Acceleration](#pytorch-acceleration)
2. [Indicator Organization](#indicator-organization)
3. [Lorentzian Classifier](#lorentzian-classifier)
4. [Wave Trend Enhanced](#wave-trend-enhanced)
5. [Custom RSI Implementation](#custom-rsi-implementation)
6. [Custom CCI Implementation](#custom-cci-implementation)
7. [ADX Implementation](#adx-implementation)

## PyTorch Acceleration

All our technical indicators leverage PyTorch for GPU acceleration and automatic differentiation. This enables:

- Up to 100x faster calculation on GPUs
- Automatic differentiation for gradient-based optimization
- Batch processing for efficient backtesting
- Seamless integration with deep learning models

### Base Indicator Implementation

```python
class BaseTorchIndicator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device(config.device)
        self.dtype = config.dtype
        
    def to_tensor(self, data):
        """Convert data to PyTorch tensor"""
        if isinstance(data, pd.Series):
            data = data.values
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return data.to(self.device).to(self.dtype)
        
    def calculate(self, data):
        """Main calculation method with GPU acceleration"""
        with torch.cuda.amp.autocast() if self.config.use_amp else nullcontext():
            signals = self.calculate_signals(data)
            
        return {k: pd.Series(v.cpu().numpy(), index=data.index) 
                for k, v in signals.items()}
```

## Indicator Organization

Our indicators are organized into these key directories:

```
src/
├── features/         # Core technical indicators (RSI, CCI, ADX, WaveTrend)
├── indicators/       # Base indicator foundations
└── models/strategy/  # Strategy indicators (Lorentzian, Logistic Regression, Chandelier)
```

## Lorentzian Classifier

The core of our signal generation system, using a unique distance metric for pattern recognition.

### Mathematical Foundation
```python
def lorentzian_distance(x1, x2):
    return np.log(1 + np.abs(x1 - x2))
```

### Key Components
1. **Distance Calculation**
   ```python
   distances = np.array([
       lorentzian_distance(current_point, historical_point)
       for historical_point in historical_data
   ])
   ```

2. **Neighbor Selection**
   - Uses k=8 nearest neighbors
   - Chronological sampling (4 bar spacing)
   - Dynamic threshold adjustment

3. **Signal Generation**
   ```python
   signal_strength = np.mean(np.sort(distances)[:n_neighbors])
   signal = 1 if signal_strength > upper_threshold else -1 if signal_strength < lower_threshold else 0
   ```

### Optimization Parameters
| Parameter | Default | Range | Description |
|-----------|---------|--------|-------------|
| n_neighbors | 8 | 4-12 | Number of neighbors |
| bar_spacing | 4 | 2-8 | Chronological spacing |
| threshold | 0.1 | 0.05-0.2 | Signal threshold |

## Wave Trend Enhanced

Custom implementation of the WaveTrend indicator with enhanced sensitivity.

### Calculation
```python
def calculate_wave_trend(close, channel_length=10, avg_length=11):
    # Step 1: Calculate ESA (Exponential Moving Average)
    esa = ema(close, channel_length)
    
    # Step 2: Calculate absolute distance
    d = ema(abs(close - esa), channel_length)
    
    # Step 3: Calculate CI (Raw Wave Trend)
    ci = (close - esa) / (0.015 * d)
    
    # Step 4: Calculate Wave Trend Lines
    wt1 = ema(ci, avg_length)      # Wave Trend Line 1
    wt2 = sma(wt1, 4)             # Wave Trend Line 2
    
    return wt1, wt2
```

### Signal Zones
- **Overbought**: > 60
- **Oversold**: < -60
- **Extreme Overbought**: > 80
- **Extreme Oversold**: < -80

### Cross Signals
```python
def wave_trend_cross(wt1, wt2):
    # Bullish Cross
    bullish = (wt1 > wt2) & (wt1.shift(1) <= wt2.shift(1))
    
    # Bearish Cross
    bearish = (wt1 < wt2) & (wt1.shift(1) >= wt2.shift(1))
    
    return bullish, bearish
```

## Custom RSI Implementation

Enhanced RSI with GPU acceleration and advanced signal processing.

### Features
1. **PyTorch Implementation**
   ```python
   class RSIIndicator(BaseTorchIndicator):
       def __init__(self, period=14, overbought=70.0, oversold=30.0, device=None, dtype=None):
           config = RSIConfig(
               period=period,
               overbought=overbought,
               oversold=oversold,
               device=device,
               dtype=dtype
           )
           super().__init__(config)
           
       def forward(self, close):
           # Calculate price changes
           price_diff = torch.diff(close, dim=0)
           padding = torch.zeros(1, device=self.device, dtype=self.dtype)
           price_diff = torch.cat([padding, price_diff])
   
           # Separate gains and losses
           gains = torch.where(price_diff > 0, price_diff, torch.zeros_like(price_diff))
           losses = torch.where(price_diff < 0, -price_diff, torch.zeros_like(price_diff))
   
           # Calculate average gains and losses
           avg_gains = self.torch_ema(gains, self.alpha.item())
           avg_losses = self.torch_ema(losses, self.alpha.item())
   
           # Calculate relative strength and RSI
           rs = avg_gains / (avg_losses + 1e-10)
           rsi = 100.0 - (100.0 / (1.0 + rs))
           
           return {
               'rsi': rsi,
               'buy_signals': (rsi < self.config.oversold).to(self.dtype),
               'sell_signals': (rsi > self.config.overbought).to(self.dtype)
           }
   ```

2. **Configuration**
   ```python
   @dataclass
   class RSIConfig(TorchIndicatorConfig):
       period: int = 14
       overbought: float = 70.0
       oversold: float = 30.0
   ```

### Signal Generation
- **Strong Buy**: RSI < 30 (oversold condition)
- **Strong Sell**: RSI > 70 (overbought condition)
- **Neutral**: 30 ≤ RSI ≤ 70

### Usage Example
```python
# Initialize indicator
rsi = RSIIndicator(period=14, device="cuda" if torch.cuda.is_available() else "cpu")

# Calculate signals
results = rsi.calculate(price_data)

# Access results
rsi_values = results['rsi']
buy_signals = results['buy_signals']
sell_signals = results['sell_signals']
```

## Custom CCI Implementation

Modified CCI with enhanced sensitivity and noise reduction.

### Calculation
```python
def enhanced_cci(high, low, close, length=20, smooth=1):
    # Calculate Typical Price
    tp = (high + low + close) / 3
    
    # Calculate SMA of Typical Price
    sma_tp = talib.SMA(tp, timeperiod=length)
    
    # Calculate Mean Deviation
    mean_dev = mean_deviation(tp, sma_tp, length)
    
    # Calculate Raw CCI
    cci = (tp - sma_tp) / (0.015 * mean_dev)
    
    # Apply smoothing
    if smooth > 1:
        cci = talib.EMA(cci, timeperiod=smooth)
    
    return cci
```

### Signal Thresholds
| Zone | Range | Signal Strength |
|------|-------|----------------|
| Extreme Overbought | > 200 | Strong Sell |
| Overbought | 100 to 200 | Weak Sell |
| Neutral | -100 to 100 | No Signal |
| Oversold | -200 to -100 | Weak Buy |
| Extreme Oversold | < -200 | Strong Buy |

### Divergence Detection
```python
def detect_cci_divergence(price, cci, window=14):
    # Price making higher highs
    price_hh = price > price.shift(window)
    
    # CCI making lower highs
    cci_lh = cci < cci.shift(window)
    
    # Bearish divergence
    bearish_div = price_hh & cci_lh
    
    # Similar for bullish divergence
    return bearish_div
```

## Integration with ML Pipeline

### Feature Engineering
```python
def calculate_all_features(ohlcv_data):
    features = {}
    
    # Add Wave Trend
    features['wt1'], features['wt2'] = calculate_wave_trend(
        ohlcv_data['close']
    )
    
    # Add RSI
    features['rsi'] = smooth_rsi(
        ohlcv_data['close']
    )
    
    # Add CCI
    features['cci'] = enhanced_cci(
        ohlcv_data['high'],
        ohlcv_data['low'],
        ohlcv_data['close']
    )
    
    return features
```

### Signal Combination
```python
def combine_indicator_signals(features):
    signals = {
        'wave_trend': get_wt_signal(features['wt1'], features['wt2']),
        'rsi': get_rsi_signal(features['rsi']),
        'cci': get_cci_signal(features['cci'])
    }
    
    # Weight and combine signals
    final_signal = weighted_signal_combination(signals)
    
    return final_signal
```

---

*This documentation provides detailed insights into our custom technical indicators and their implementation. Each indicator has been optimized for our specific trading strategy and market conditions.* 

## See Also
- [Project README](../README.md) — Project overview and structure
- [ML Model Architecture](ML_MODEL.md) — How indicators are used in the model
- [Technical Strategy](TECHNICAL_STRATEGY.md) — How indicators drive trading logic
- [src/features/](../src/features/) — Core indicator code ([rsi.py](../src/features/rsi.py), [cci.py](../src/features/cci.py), [adx.py](../src/features/adx.py), [wave_trend.py](../src/features/wave_trend.py))
- [src/indicators/base_torch_indicator.py](../src/indicators/base_torch_indicator.py) — PyTorch base class for indicators
- [src/models/strategy/](../src/models/strategy/) — Strategy code using these indicators 