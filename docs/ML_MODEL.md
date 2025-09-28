# 🧠 Machine Learning Model Architecture

*What is this doc?*  
This guide explains the structure, training, and deployment of the project's main ML model. It's for anyone wanting to understand, extend, or use the model—whether you're a developer, researcher, or curious trader.

[Project Structure](../README.md#project-structure) | [Indicators](INDICATORS.md) | [Technical Strategy](TECHNICAL_STRATEGY.md) | [Model Training](MODEL_TRAINING.md) | [ML Structure Guide](ML_MODEL_STRUCTURE.md)

## Overview
Our trading model uses PyTorch to implement a hybrid architecture combining traditional technical analysis with deep learning. This document explains the model's structure, training process, and deployment.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Model Architecture](#model-architecture)
3. [Training Pipeline](#training-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Performance Optimization](#performance-optimization)
6. [Deployment](#deployment)

## Project Structure

The project is organized into the following directory structure:

```
src/
├── features/           # Core technical indicators
│   ├── rsi.py          # Relative Strength Index
│   ├── cci.py          # Commodity Channel Index
│   ├── adx.py          # Average Directional Index
│   └── wave_trend.py   # WaveTrend Oscillator
├── indicators/         # Base indicator foundations
│   └── base_torch_indicator.py  # PyTorch base indicator class
├── models/
│   ├── strategy/       # Trading strategies
│   │   ├── lorentzian_classifier.py
│   │   ├── logistic_regression_torch.py
│   │   └── chandelier_exit.py
│   └── training/       # Model training utilities
└── pattern-recognition/ # Pattern detection algorithms
```

## Model Architecture

### Base Model Structure
```python
class TradingModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(config.input_channels, 64, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        
        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4
        )
        
        # Technical indicator integration
        self.technical_layer = nn.Linear(
            config.n_indicators,
            64
        )
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config.output_dim)
        )
```

### Model Components

1. **Feature Extraction**
   - Convolutional layers for pattern recognition
   - Batch normalization for training stability
   - Dropout for regularization

2. **Temporal Processing**
   - LSTM layers for sequence modeling
   - Attention mechanism for important pattern focus
   - Residual connections for gradient flow

3. **Technical Integration**
   - Direct input of technical indicators
   - Learned feature combination
   - Adaptive weighting mechanism

## Training Pipeline

### 1. Data Loading
```python
class TradingDataset(Dataset):
    def __init__(self, data, lookback=100):
        self.data = data
        self.lookback = lookback
        
    def __getitem__(self, idx):
        # Get sequence of price data
        sequence = self.data[idx:idx + self.lookback]
        
        # Get technical indicators
        indicators = self.get_indicators(sequence)
        
        # Get target (future price movement)
        target = self.get_target(idx + self.lookback)
        
        return {
            'sequence': sequence,
            'indicators': indicators,
            'target': target
        }
```

### 2. Training Loop
```python
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    
    for batch in dataloader:
        # Move data to device
        sequences = batch['sequence'].to(device)
        indicators = batch['indicators'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        predictions = model(sequences, indicators)
        loss = criterion(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        update_metrics(predictions, targets)
```

### 3. Validation
```python
@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    metrics = defaultdict(list)
    
    for batch in val_loader:
        # Generate predictions
        predictions = model(
            batch['sequence'].to(device),
            batch['indicators'].to(device)
        )
        
        # Calculate metrics
        metrics['loss'].append(
            criterion(predictions, batch['target'].to(device))
        )
        metrics['accuracy'].append(
            calculate_accuracy(predictions, batch['target'])
        )
        
    return {k: np.mean(v) for k, v in metrics.items()}
```

## Feature Engineering

### 1. Price Features
```python
def calculate_price_features(df):
    features = {}
    
    # Price changes
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        features[f'ma_{window}'] = df['close'].rolling(window).mean()
        
    # Volatility
    features['volatility'] = features['returns'].rolling(20).std()
    
    return features
```

### 2. GPU-Accelerated Technical Indicators

We use PyTorch-based implementations of technical indicators for improved performance:

```python
# Initialize RSI with GPU support
rsi_indicator = RSIIndicator(
    period=14,
    overbought=70.0,
    oversold=30.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float32
)

# Calculate RSI values and signals
rsi_results = rsi_indicator.calculate(data)

# Access RSI values and signals
rsi_values = rsi_results['rsi']
buy_signals = rsi_results['buy_signals']
sell_signals = rsi_results['sell_signals']
```

### 3. Indicator Integration

Our PyTorch indicators provide a unified interface:

```python
def create_feature_matrix(data):
    # Initialize indicators
    indicators = {
        'rsi': RSIIndicator(period=14),
        'cci': CCIIndicator(period=20),
        'adx': ADXIndicator(period=14),
        'wavetrend': WaveTrendIndicator(channel_length=10, average_length=21)
    }
    
    # Calculate all indicator values
    features = {}
    for name, indicator in indicators.items():
        results = indicator.calculate(data)
        features.update({f"{name}_{k}": v for k, v in results.items()})
    
    return pd.DataFrame(features)
```

### 4. Feature Normalization
```python
def normalize_features(features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
    else:
        features_normalized = scaler.transform(features)
        
    return features_normalized, scaler
```

## Performance Optimization

### 1. GPU Acceleration
```python
def setup_training(model, device='cuda'):
    # Move model to GPU if available
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
        
    # Enable automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    return model, scaler
```

### 2. Memory Management
```python
def optimize_memory(dataloader):
    # Pin memory for faster GPU transfer
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        pin_memory=True,
        num_workers=4
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    return dataloader
```

### 3. Training Optimization
```python
def optimize_training():
    # Use mixed precision training
    with torch.cuda.amp.autocast():
        predictions = model(sequences, indicators)
        loss = criterion(predictions, targets)
    
    # Scale loss and backpropagate
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Deployment

### 1. Model Export
```python
def export_model(model, path):
    # Save model architecture and weights
    torch.save({
        'model_state': model.state_dict(),
        'config': model.config,
        'scaler': scaler,
    }, path)
```

### 2. Inference Pipeline
```python
def setup_inference(model_path):
    # Load model for inference
    checkpoint = torch.load(model_path)
    model = TradingModel(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    return model

def predict(model, data):
    with torch.no_grad():
        predictions = model(
            data['sequence'],
            data['indicators']
        )
    return predictions
```

### 3. Real-time Processing
```python
async def process_realtime(model, data_stream):
    async for data in data_stream:
        # Preprocess data
        features = calculate_features(data)
        
        # Generate prediction
        prediction = predict(model, features)
        
        # Update trading signals
        await update_signals(prediction)
```

---

*This documentation provides a comprehensive overview of our machine learning model architecture. For implementation details or specific questions, please refer to the relevant sections or contact the development team.* 

## See Also
- [Project README](../README.md) — Project overview, directory map, and quick start
- [Technical Indicators](INDICATORS.md) — Details on all custom indicators
- [Technical Strategy](TECHNICAL_STRATEGY.md) — How the model fits into the trading system
- [Model Training Guide](MODEL_TRAINING.md) — How to train and evaluate models
- **[ML Structure Guide](ML_MODEL_STRUCTURE.md) — CRITICAL: Features vs Strategies organization**
- [Neon Data Pipeline](NEON_PIPELINE.md) — Data ingestion and feature engineering
- [src/features/](../src/features/) — Core indicator code (e.g., [rsi.py](../src/features/rsi.py), [cci.py](../src/features/cci.py))
- [src/models/strategy/](../src/models/strategy/) — Strategy/model code (e.g., [lorentzian_classifier.py](../src/models/strategy/lorentzian_classifier.py)) 