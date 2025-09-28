#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Training Script for SOL Trading Strategy

USE CASES:
- **Lorentzian Classifier training**: Train the main Lorentzian-inspired ML model
- **Feature engineering**: Comprehensive technical indicator calculation
- **Model evaluation**: Backtesting and performance metrics calculation
- **Hyperparameter tuning**: Configurable training parameters
- **Model persistence**: Save trained models and metrics
- **Data preprocessing**: Automated data cleaning and feature creation
- **Performance visualization**: Generate training plots and metrics

DIFFERENCES FROM OTHER TRAINING SCRIPTS:
- train_model.py: Main Lorentzian classifier training with comprehensive features
- train_model_walkforward.py: Walk-forward optimization with time-based validation
- train_logistic_regression.py: Simple logistic regression baseline
- train_logistic_regression_fast.py: Fast logistic regression for quick experiments
- extended_training.py: Extended training with additional features and models

WHEN TO USE:
- For training the main Lorentzian classifier model
- When you need comprehensive feature engineering
- For production model training with full evaluation
- When you want detailed performance metrics and visualization
- For hyperparameter experimentation

FEATURES:
- Lorentzian-inspired neural network architecture
- 20+ technical indicators (RSI, MACD, Bollinger Bands, ADX, etc.)
- Automated data preprocessing and feature engineering
- Comprehensive model evaluation with trading metrics
- Model persistence and metrics saving
- GPU acceleration support
- Configurable training parameters

EXAMPLES:
    # Basic training
    python scripts/train_model.py --data-days 90 --epochs 50
    
    # Custom parameters
    python scripts/train_model.py --timeframe 15m --batch-size 128 --lr 0.0005
    
    # Different model type
    python scripts/train_model.py --model-type logistic --epochs 100
"""

import os
import logging
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import project modules
from src.data.collectors.sol_data_collector import SOLDataCollector
from src.ml.models.strategy.lorentzian_classifier import LorentzianANN

# Create a simple PyTorch neural network for training
class LorentzianClassifier(nn.Module):
    """Simple neural network classifier for Lorentzian-inspired trading signals"""
    
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
        super(LorentzianClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate the Lorentzian ML model')
    parser.add_argument('--data-days', type=int, default=30, 
                        help='Number of days of historical data to use')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate for training')
    parser.add_argument('--save-path', type=str, default='models/trained', 
                        help='Path to save the trained model')
    parser.add_argument('--timeframe', type=str, default='5m', choices=['5m', '15m', '1h', '4h'],
                        help='Timeframe to use for training data (default: 5m)')
    parser.add_argument('--model-type', type=str, default='lorentzian', choices=['lorentzian', 'logistic'],
                        help='Type of model to train (default: lorentzian)')
    return parser.parse_args()

async def fetch_training_data(days=90):
    """Fetch historical data for training"""
    logger.info(f"Fetching {days} days of historical data for training")
    
    # Initialize data collector
    collector = SOLDataCollector()
    
    # Fetch data for all timeframes
    data = await collector.fetch_all_timeframes(lookback_days=days)
    
    logger.info(f"Successfully fetched data for timeframes: {', '.join(data.keys())}")
    
    return data

def preprocess_data(data, timeframe='5m'):
    """Preprocess data for model training"""
    logger.info(f"Preprocessing data for training using {timeframe} timeframe")
    
    # Use selected timeframe as base
    df = data[timeframe].copy()
    
    logger.info(f"Starting with {len(df)} rows of {timeframe} data")
    
    # Make sure we have data
    if len(df) == 0:
        logger.error("No data available for preprocessing")
        raise ValueError("Empty dataset provided for preprocessing")
    
    try:
        # Calculate technical indicators optimized for crypto trading
        
        # Volume indicators
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['volume_ma50'] = df['volume'].rolling(window=50).mean()
        df['relative_volume'] = df['volume'] / df['volume_ma20']
        
        # Price action indicators
        df['price_change'] = df['close'].pct_change()
        df['price_volatility'] = df['price_change'].rolling(window=20).std()
        df['high_low_diff'] = (df['high'] - df['low']) / df['low']
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Momentum
        df['mom_1'] = df['close'].pct_change(1)
        df['mom_5'] = df['close'].pct_change(5)
        df['mom_10'] = df['close'].pct_change(10)
        
        # RSI (14 period)
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-7)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI divergence
        df['rsi_slope'] = pd.Series(np.gradient(df['rsi'].values), index=df.index)
        df['price_slope'] = pd.Series(np.gradient(df['close'].values), index=df.index)
        df['rsi_div'] = (df['rsi_slope'] < 0) & (df['price_slope'] > 0)  # Bearish divergence
        df['rsi_div'] = df['rsi_div'].astype(int)
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_slope'] = pd.Series(np.gradient(df['macd'].values), index=df.index)
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Moving average crossovers
        df['ma_cross_5_20'] = (df['sma_5'] > df['sma_20']).astype(int)
        df['ma_cross_10_50'] = (df['sma_10'] > df['sma_50']).astype(int)
        
        # ADX (simplified)
        high_delta = df['high'].diff()
        low_delta = df['low'].diff()
        df['tr'] = np.maximum(
            np.maximum(
                df['high'] - df['low'],
                np.abs(df['high'] - df['close'].shift())
            ),
            np.abs(df['low'] - df['close'].shift())
        )
        df['atr'] = df['tr'].rolling(window=14).mean()
        
        # Calculate directional movement
        df['dmplus'] = np.where((high_delta > 0) & (high_delta > low_delta.abs()), high_delta, 0)
        df['dmminus'] = np.where((low_delta < 0) & (low_delta.abs() > high_delta), low_delta.abs(), 0)
        
        # Normalize DM by ATR
        df['diplus'] = 100 * df['dmplus'].rolling(window=14).mean() / df['atr']
        df['diminus'] = 100 * df['dmminus'].rolling(window=14).mean() / df['atr']
        
        # Calculate ADX
        df['dx'] = 100 * np.abs(df['diplus'] - df['diminus']) / (df['diplus'] + df['diminus']).replace(0, 1e-7)
        df['adx'] = df['dx'].rolling(window=14).mean()
        
        # Stochastic oscillator
        df['stoch_k'] = 100 * ((df['close'] - df['low'].rolling(window=14).min()) / 
                             (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()).replace(0, 1e-7))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Stochastic RSI
        df['stoch_rsi'] = 100 * ((df['rsi'] - df['rsi'].rolling(window=14).min()) / 
                               (df['rsi'].rolling(window=14).max() - df['rsi'].rolling(window=14).min()).replace(0, 1e-7))
        
        logger.info(f"Calculated indicators successfully")
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        # Fall back to minimal indicator set
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        logger.info("Using minimal indicator set due to calculation error")
    
    # Drop rows with NaN values from indicator calculations
    # Make sure to do this before creating labels
    original_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {original_len - len(df)} rows with NaN values from indicators")
    
    # Make sure we still have data
    if len(df) < 25:  # Need at least some data for meaningful training
        logger.error(f"Too few rows ({len(df)}) after dropping NaNs")
        raise ValueError("Insufficient data for training after preprocessing")
    
    # Create labels based on future price movement
    # For 5-minute timeframe, look ahead 12 periods (1 hour)
    # For 15-minute timeframe, look ahead 4 periods (1 hour)
    # For 1-hour timeframe, look ahead 4 periods (4 hours)
    timeframe_lookahead = {
        '5m': 12,
        '15m': 4,
        '1h': 4,
        '4h': 6
    }
    
    future_periods = timeframe_lookahead.get(timeframe, 12)  # Default to 12 periods
    threshold_pct = {
        '5m': 0.005,  # 0.5% for 5m
        '15m': 0.008,  # 0.8% for 15m
        '1h': 0.01,    # 1% for 1h
        '4h': 0.015    # 1.5% for 4h
    }
    
    threshold = threshold_pct.get(timeframe, 0.005)  # Default to 0.5%
    
    logger.info(f"Creating labels with {future_periods} periods look-ahead, threshold: {threshold*100}%")
    
    # Use future returns for label
    df['future_price'] = df['close'].shift(-future_periods)
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    df['label'] = (df['future_return'] > threshold).astype(int)
    
    # Drop rows with NaN in labels (last few rows will have NaN future values)
    df = df.dropna(subset=['label'])
    logger.info(f"After dropping rows with NaN labels: {len(df)} rows remaining")
    logger.info(f"Label distribution - Positive: {df['label'].sum()} ({df['label'].mean()*100:.2f}%), Negative: {len(df)-df['label'].sum()}")
    
    # Select features - use whatever columns are available
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove future-related columns and target from features
    feature_columns = [col for col in numerical_columns 
                      if col not in ['future_price', 'future_return', 'label']]
    
    logger.info(f"Selected {len(feature_columns)} features: {', '.join(feature_columns[:5])}...")
    
    # Create feature matrix and labels
    X = df[feature_columns].values
    y = df['label'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data - use chronological split rather than random
    test_size = min(0.2, 1.0/3.0)  # Use at most 1/3rd of data for testing
    split_idx = int(len(X_scaled) * (1 - test_size))
    
    X_train = X_scaled[:split_idx]
    X_test = X_scaled[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    logger.info(f"Preprocessed data shape: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Save feature names for later use
    feature_names = feature_columns
    
    return X_train, X_test, y_train, y_test, scaler, feature_names

def train_model(X_train, y_train, X_test, y_test, args):
    """Train the model on preprocessed data"""
    logger.info("Training model")
    
    # Create PyTorch datasets
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True
    )
    
    # Create model
    input_size = X_train.shape[1]
    logger.info(f"Creating model with input size: {input_size}")
    model = LorentzianClassifier(input_size=input_size)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    losses = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            test_accuracy = (test_predictions.squeeze() == y_test_tensor).float().mean()
        
        logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_path = args.save_path
    if '.' not in os.path.basename(model_path):
        # If save_path is a directory
        os.makedirs(model_path, exist_ok=True)
        model_path = os.path.join(model_path, f"lorentzian_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
    else:
        # If save_path includes a filename
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    logger.info(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved successfully")
    
    return model, losses, model_path

def evaluate_trading_performance(model, X_test, y_test, feature_names):
    """Evaluate the model's trading performance using various metrics"""
    logger.info("Evaluating trading performance")
    
    # Generate predictions
    X_test_tensor = torch.FloatTensor(X_test)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    
    # Create trade signals (1 for buy, -1 for sell, 0 for hold)
    # Use more sensitive thresholds to generate more signals
    buy_threshold = 0.55  # Lower threshold for buys (was 0.6)
    sell_threshold = 0.45  # Higher threshold for sells (was 0.4)
    
    signals = np.zeros(len(predictions))
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1
    
    logger.info(f"Generated {np.sum(signals != 0)} signals out of {len(signals)} predictions")
    logger.info(f"Buy signals: {np.sum(signals == 1)}, Sell signals: {np.sum(signals == -1)}")
    
    # For backtesting, simulate trade outcomes
    returns = []
    positions = []
    current_position = 0
    entry_price = None
    entry_index = None
    
    # Add parameter to limit holding period (exit after N periods if no signal)
    max_holding_periods = 10  # Maximum number of periods to hold a position
    
    # Simple backtesting logic - using ground truth as the "future" price movement
    for i in range(len(signals)):
        signal = signals[i]
        actual = y_test[i]
        
        # Record the current position
        positions.append(current_position)
        
        # Check if we need to exit due to max holding period
        if (current_position != 0 and entry_index is not None and 
            i - entry_index >= max_holding_periods):
            # Force exit after max holding period
            return_val = 0.01 if ((current_position == 1 and actual == 1) or 
                                 (current_position == -1 and actual == 0)) else -0.01
            returns.append(return_val)
            logger.info(f"Forced exit at index {i} after max holding period, return: {return_val:.4f}")
            current_position = 0
            entry_price = None
            entry_index = None
            continue
        
        # No position and signal to buy
        if current_position == 0 and signal == 1:
            current_position = 1
            entry_price = 1.0  # Normalized price
            entry_index = i
            returns.append(0)
        
        # No position and signal to sell
        elif current_position == 0 and signal == -1:
            current_position = -1
            entry_price = 1.0  # Normalized price
            entry_index = i
            returns.append(0)
        
        # Long position
        elif current_position == 1:
            # Exit signal
            if signal == -1:
                # If actual label is 1, we made money; if 0, we lost
                return_val = 0.01 if actual == 1 else -0.01
                returns.append(return_val)
                logger.info(f"Exit long at index {i}, return: {return_val:.4f}")
                current_position = 0
                entry_price = None
                entry_index = None
            else:
                returns.append(0)
        
        # Short position
        elif current_position == -1:
            # Exit signal
            if signal == 1:
                # If actual label is 0, we made money; if 1, we lost
                return_val = 0.01 if actual == 0 else -0.01
                returns.append(return_val)
                logger.info(f"Exit short at index {i}, return: {return_val:.4f}")
                current_position = 0
                entry_price = None
                entry_index = None
            else:
                returns.append(0)
    
    # Close any remaining positions at the end
    if current_position != 0:
        last_return = 0.01 if (current_position == 1 and y_test[-1] == 1) or (current_position == -1 and y_test[-1] == 0) else -0.01
        returns.append(last_return)
        logger.info(f"Closed final position with return: {last_return:.4f}")
    
    # Calculate comprehensive metrics
    returns = np.array(returns)
    
    # Basic trading metrics
    total_returns = np.sum(returns)
    winning_trades = np.sum(returns > 0)
    losing_trades = np.sum(returns < 0)
    total_trades = winning_trades + losing_trades
    
    # Avoid division by zero
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    avg_win = np.mean(returns[returns > 0]) if winning_trades > 0 else 0
    avg_loss = np.mean(returns[returns < 0]) if losing_trades > 0 else 0
    profit_factor = -np.sum(returns[returns > 0]) / np.sum(returns[returns < 0]) if np.sum(returns[returns < 0]) != 0 else float('inf')
    
    # Risk metrics
    if len(returns) > 0:
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = np.max(np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns))
    else:
        sharpe_ratio = 0
        max_drawdown = 0
    
    metrics = {
        'total_returns': float(total_returns),
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': float(win_rate),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'profit_factor': float(profit_factor),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'signal_ratio': float(np.sum(signals != 0) / len(signals)) # Percentage of predictions that generated signals
    }
    
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")
    
    return metrics

async def main():
    """Main function to train and evaluate the model"""
    args = parse_arguments()
    
    # Fetch training data
    data = await fetch_training_data(days=args.data_days)
    
    # Preprocess data using the selected timeframe
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data, timeframe=args.timeframe)
    
    # Train model
    model, losses, model_path = train_model(X_train, y_train, X_test, y_test, args)
    
    # Evaluate trading performance
    metrics = evaluate_trading_performance(model, X_test, y_test, feature_names)
    
    # Save metrics
    metrics_file = os.path.splitext(model_path)[0] + "_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Add metadata to metrics
    metrics['timeframe'] = args.timeframe
    metrics['data_days'] = args.data_days
    metrics['feature_count'] = len(feature_names)
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f'Training Loss ({args.timeframe} timeframe)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.splitext(model_path)[0] + "_loss_curve.png"
    plt.savefig(loss_plot_path)
    
    logger.info(f"Training completed. Metrics saved to {metrics_file}")
    logger.info(f"Loss curve saved to {loss_plot_path}")
    
    return {
        'model_path': model_path,
        'metrics': metrics,
        'losses': losses
    }

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 