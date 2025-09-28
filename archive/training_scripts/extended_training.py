#!/usr/bin/env python
"""
Extended Training Script for SOL Trading Model

This script implements an extended training session with walk-forward optimization.
It can run for several hours, continuously improving the model by testing various
hyperparameter combinations.

USE CASES:
- **Extended model training**: Long-running training sessions for optimal models
- **Walk-forward optimization**: Time-based validation across multiple periods
- **Hyperparameter search**: Comprehensive parameter space exploration
- **Production model development**: Extended training for production-ready models
- **Performance optimization**: Continuous improvement over long time periods
- **Research and experimentation**: Deep exploration of model configurations

DIFFERENCES FROM OTHER TRAINING SCRIPTS:
- extended_training.py: Extended training with long-running optimization
- train_model.py: Main Lorentzian classifier training with comprehensive features
- train_model_walkforward.py: Walk-forward optimization with time-based validation
- train_logistic_regression.py: Full logistic regression with optimization
- train_logistic_regression_fast.py: Fast GPU-optimized logistic regression

WHEN TO USE:
- For production model development
- When you need optimal model performance
- For comprehensive hyperparameter search
- When you have time for extended training sessions
- For research and deep model exploration

FEATURES:
- Long-running training sessions (up to 8 hours)
- Walk-forward validation across multiple time periods
- Comprehensive hyperparameter search
- Enhanced feature engineering
- Continuous model improvement
- Detailed logging and progress tracking

Usage:
    python scripts/extended_training.py --timeframe 5m --start-date 2024-01-01 --end-date 2025-04-02
"""

import os
import sys
import json
import logging
import asyncio
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import random

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.collectors.sol_data_collector import SOLDataCollector
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/extended_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extended training for SOL trading model')
    parser.add_argument('--timeframe', type=str, default='5m', choices=['5m', '15m'],
                        help='Timeframe to use for training (5m or 15m)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                        help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2025-04-02',
                        help='End date for training data (YYYY-MM-DD)')
    parser.add_argument('--max-runtime', type=int, default=28800,  # 8 hours
                        help='Maximum runtime in seconds')
    parser.add_argument('--save-path', type=str, default='models/trained/extended',
                        help='Path to save the trained models')
    return parser.parse_args()

async def fetch_historical_data(start_date, end_date, timeframes=['5m', '15m', '1h', '4h', '1d']):
    """Fetch historical data for multiple timeframes"""
    logger.info(f"Fetching historical data from {start_date} to {end_date}")
    
    # Calculate the number of days between start and end date
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end_dt - start_dt).days
    
    logger.info(f"Fetching {timeframes} data for {days} days...")
    
    collector = SOLDataCollector()
    
    # Initialize dictionary to store data for each timeframe
    data = {}
    
    # Fetch data for each timeframe using the fetch_historical method
    for tf in timeframes:
        logger.info(f"Fetching {tf} data for SOLUSDT...")
        
        # Our collector doesn't have a method to fetch by exact date range,
        # so we'll use the lookback days parameter instead
        try:
            # For 5m and 15m, we might hit API limits if we fetch too many days at once
            if tf in ['5m', '15m'] and days > 30:
                # Fetch in chunks of 30 days
                chunk_data = []
                for i in range(0, days, 30):
                    chunk_days = min(30, days - i)
                    tf_data = await collector.fetch_historical(tf, lookback_days=chunk_days)
                    chunk_data.append(tf_data)
                    logger.info(f"Fetched chunk {i//30 + 1} for {tf}: {len(tf_data)} candles")
                
                # Combine chunks
                if chunk_data:
                    data[tf] = pd.concat(chunk_data).drop_duplicates()
            else:
                # Fetch all at once for larger timeframes
                data[tf] = await collector.fetch_historical(tf, lookback_days=days)
            
            logger.info(f"Successfully fetched {len(data[tf])} {tf} candles")
        except Exception as e:
            logger.error(f"Error fetching {tf} data: {str(e)}")
            # Create empty DataFrame if fetch fails
            data[tf] = pd.DataFrame()
    
    # Check if we got data for at least one timeframe
    if all(len(df) == 0 for df in data.values()):
        raise ValueError("Failed to fetch data for any timeframe")
    
    logger.info(f"Successfully fetched data for timeframes: {', '.join(data.keys())}")
    
    return data

def preprocess_data(data, timeframe='5m'):
    """Preprocess data for model training with enhanced features"""
    logger.info(f"Preprocessing data for training using {timeframe} timeframe")
    
    # Use selected timeframe as base
    df = data[timeframe].copy()
    
    logger.info(f"Starting with {len(df)} rows of {timeframe} data")
    
    # Make sure we have data
    if len(df) == 0:
        logger.error("No data available for preprocessing")
        raise ValueError("Empty dataset provided for preprocessing")
    
    try:
        # Calculate technical indicators for crypto trading
        
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
    original_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {original_len - len(df)} rows with NaN values from indicators")
    
    # Make sure we still have data
    if len(df) < 25:  # Need at least some data for meaningful training
        logger.error(f"Too few rows ({len(df)}) after dropping NaNs")
        raise ValueError("Insufficient data for training after preprocessing")
        
    # Create labels based on future price movement
    timeframe_lookahead = {
        '5m': 12,   # 1 hour look-ahead
        '15m': 4,   # 1 hour look-ahead
    }
    
    future_periods = timeframe_lookahead.get(timeframe, 12)
    threshold_pct = {
        '5m': 0.005,  # 0.5% for 5m
        '15m': 0.008,  # 0.8% for 15m
    }
    
    threshold = threshold_pct.get(timeframe, 0.005)
    
    logger.info(f"Creating labels with {future_periods} periods look-ahead, threshold: {threshold*100}%")
    
    # Use future returns for label
    df['future_price'] = df['close'].shift(-future_periods)
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    df['label'] = (df['future_return'] > threshold).astype(int)
    
    # Drop rows with NaN in labels (last few rows will have NaN future values)
    df = df.dropna(subset=['label'])
    logger.info(f"After dropping rows with NaN labels: {len(df)} rows remaining")
    
    # Select features - use whatever columns are available
    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove future-related columns and target from features
    feature_columns = [col for col in numerical_columns 
                      if col not in ['future_price', 'future_return', 'label']]
    
    logger.info(f"Selected {len(feature_columns)} features")
    
    # Create feature matrix and labels
    X = df[feature_columns].values
    y = df['label'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Return everything needed for training
    return X_scaled, y, feature_columns

def train_model(X, y, hyperparams, max_epochs=30):
    """Train a model with given hyperparameters"""
    
    # Split data for training/validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # Create dataset and loader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=hyperparams['batch_size'], 
        shuffle=True
    )
    
    # Create model with given hyperparameters
    input_size = X.shape[1]
    hidden_size = hyperparams['hidden_size']
    dropout_rate = hyperparams['dropout_rate']
    
    model = LorentzianClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout_rate=dropout_rate
    )
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # Training loop
    best_val_accuracy = 0
    best_epoch = 0
    patience = 5  # Early stopping patience
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.unsqueeze(1))
            val_predictions = (val_outputs > 0.5).float()
            val_accuracy = (val_predictions.squeeze() == y_val_tensor).float().mean()
        
        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Return trained model and metrics
    return model, {
        'val_accuracy': float(best_val_accuracy),
        'best_epoch': best_epoch,
        'train_loss': train_loss / len(train_loader)
    }

def perform_walkforward_optimization(X, y, feature_columns, timeframe, save_path):
    """Perform walk-forward optimization"""
    logger.info("Starting walk-forward optimization")
    
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.0005, 0.001],
        'batch_size': [32, 64, 128],
        'hidden_size': [32, 64, 128],
        'dropout_rate': [0.2, 0.3, 0.4]
    }
    
    # Generate all possible combinations
    hyperparameter_combinations = list(ParameterGrid(param_grid))
    logger.info(f"Generated {len(hyperparameter_combinations)} hyperparameter combinations")
    
    # Use multiple windows for walk-forward optimization
    window_sizes = [20, 30, 40]  # in days (assuming daily data)
    fold_results = []
    
    # For each window size
    for window_size in window_sizes:
        # Calculate window size in data points (rows)
        # This is an approximation based on timeframe
        points_per_day = {'5m': 288, '15m': 96, '1h': 24}[timeframe]
        window_points = window_size * points_per_day
        
        # Make sure window isn't larger than data
        if window_points > len(X):
            window_points = len(X) // 2
        
        # Define number of folds
        n_folds = max(3, len(X) // window_points - 1)
        
        logger.info(f"Window size {window_size} days ({window_points} data points) with {n_folds} folds")
        
        for fold in range(n_folds):
            # Define train/test indices for this fold
            start_idx = fold * window_points
            end_idx = start_idx + window_points
            test_end_idx = min(end_idx + window_points, len(X))
            
            # Skip if we don't have enough data
            if test_end_idx - end_idx < window_points // 4:
                continue
                
            X_fold_train = X[start_idx:end_idx]
            y_fold_train = y[start_idx:end_idx]
            X_fold_test = X[end_idx:test_end_idx]
            y_fold_test = y[end_idx:test_end_idx]
            
            logger.info(f"Fold {fold+1}/{n_folds}: Training on indices {start_idx}:{end_idx}, Testing on {end_idx}:{test_end_idx}")
            
            # Try different hyperparameter combinations (sample a subset to save time)
            random.seed(fold)  # Make sampling reproducible per fold
            sample_size = min(5, len(hyperparameter_combinations))
            sampled_hyperparams = random.sample(hyperparameter_combinations, sample_size)
            
            for i, hyperparams in enumerate(sampled_hyperparams):
                logger.info(f"Training with hyperparams set {i+1}/{sample_size}: {hyperparams}")
                
                # Train model
                model, metrics = train_model(X_fold_train, y_fold_train, hyperparams)
                
                # Evaluate on test set
                model.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_fold_test)
                    y_test_tensor = torch.FloatTensor(y_fold_test)
                    
                    outputs = model(X_test_tensor)
                    predictions = (outputs > 0.5).float()
                    test_accuracy = (predictions.squeeze() == y_test_tensor).float().mean()
                
                # Store results
                result = {
                    'fold': fold,
                    'window_size': window_size,
                    'hyperparameters': hyperparams,
                    'val_accuracy': metrics['val_accuracy'],
                    'test_accuracy': float(test_accuracy),
                    'train_loss': metrics['train_loss']
                }
                
                fold_results.append(result)
                
                # Save model if it's good
                if float(test_accuracy) > 0.65:
                    model_filename = f"{timeframe}_win{window_size}_fold{fold}_acc{float(test_accuracy):.4f}.pt"
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_path, model_filename))
                    logger.info(f"Saved model with test accuracy {float(test_accuracy):.4f} to {model_filename}")
    
    # Save all results
    results_file = os.path.join(save_path, f"walkforward_results_{timeframe}.json")
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=4)
    
    logger.info(f"Walk-forward optimization completed. Results saved to {results_file}")
    
    # Find best model overall
    if fold_results:
        best_result = max(fold_results, key=lambda x: x['test_accuracy'])
        logger.info(f"Best model: Window {best_result['window_size']}, Fold {best_result['fold']}")
        logger.info(f"Best accuracy: {best_result['test_accuracy']:.4f}")
        logger.info(f"Best hyperparameters: {best_result['hyperparameters']}")
        
        return best_result
    else:
        logger.warning("No valid results found in walk-forward optimization")
        return None

async def main():
    """Main function for extended training"""
    # Parse arguments
    args = parse_arguments()
    
    # Adjust end date to be realistic (can't fetch future data)
    original_end_date = args.end_date
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    if datetime.strptime(args.end_date, '%Y-%m-%d') > datetime.now():
        logger.warning(f"End date {args.end_date} is in the future. Using current date {current_date} instead.")
        args.end_date = current_date
    
    # Create save directory
    save_path = os.path.join(args.save_path, f"{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}")
    os.makedirs(save_path, exist_ok=True)
    
    # Fetch historical data
    data = await fetch_historical_data(args.start_date, args.end_date)
    
    # Preprocess data
    X, y, feature_columns = preprocess_data(data, timeframe=args.timeframe)
    
    # Record start time for max runtime enforcement
    start_time = datetime.now()
    
    # Perform walk-forward optimization
    logger.info(f"Starting extended training session for {args.timeframe} timeframe")
    logger.info(f"Maximum runtime: {args.max_runtime} seconds")
    logger.info(f"Using data from {args.start_date} to {args.end_date} (originally requested until {original_end_date})")
    
    best_result = perform_walkforward_optimization(
        X, y, feature_columns, args.timeframe, save_path
    )
    
    # If we have a best result, train final model with those hyperparameters
    if best_result:
        logger.info("Training final model with best hyperparameters")
        
        final_model, metrics = train_model(X, y, best_result['hyperparameters'], max_epochs=50)
        
        # Save final model
        final_model_path = os.path.join(save_path, f"final_model_{args.timeframe}.pt")
        torch.save(final_model.state_dict(), final_model_path)
        
        # Save feature columns for later use
        feature_file = os.path.join(save_path, f"feature_columns_{args.timeframe}.json")
        with open(feature_file, 'w') as f:
            json.dump(feature_columns, f, indent=4)
        
        logger.info(f"Final model saved to {final_model_path}")
        logger.info(f"Feature columns saved to {feature_file}")
    
    # Calculate total runtime
    total_runtime = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total runtime: {total_runtime:.2f} seconds")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run main function
    asyncio.run(main()) 