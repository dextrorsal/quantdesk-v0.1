#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logistic Regression Model Training Script for SOL Trading Strategy

This script trains and optimizes the TradingView-style Logistic Regression model
for trading signals, saves the model state, and generates performance metrics.

USE CASES:
- **Logistic regression baseline**: Simple, interpretable model for trading signals
- **TradingView-style signals**: Replicate TradingView's logistic regression approach
- **Model comparison**: Baseline model for comparing against more complex models
- **Hyperparameter optimization**: Grid search for optimal parameters
- **Signal generation**: Generate buy/sell signals based on technical indicators
- **Performance benchmarking**: Establish baseline performance metrics

DIFFERENCES FROM OTHER TRAINING SCRIPTS:
- train_logistic_regression.py: Full logistic regression with optimization
- train_logistic_regression_fast.py: Fast logistic regression for quick experiments
- train_model.py: Main Lorentzian classifier training with comprehensive features
- train_model_walkforward.py: Walk-forward optimization with time-based validation
- extended_training.py: Extended training with additional features and models

WHEN TO USE:
- For establishing baseline model performance
- When you want interpretable trading signals
- For comparing against more complex models
- When you need TradingView-style signal generation
- For quick model experimentation

FEATURES:
- TradingView-style logistic regression implementation
- Hyperparameter optimization with grid search
- Technical indicator-based feature engineering
- Signal generation and performance evaluation
- Model persistence and metrics saving
- Comprehensive performance visualization

EXAMPLES:
    # Basic training
    python scripts/train_logistic_regression.py --data-days 90
    
    # With hyperparameter optimization
    python scripts/train_logistic_regression.py --optimize --timeframe 15m
    
    # Custom test split
    python scripts/train_logistic_regression.py --test-size 0.3
"""

import os
import logging
import argparse
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# Import project modules
from src.data.collectors.sol_data_collector import SOLDataCollector
from src.ml.models.strategy.logistic_regression_torch import LogisticRegression, LogisticConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and optimize Logistic Regression model')
    parser.add_argument('--data-days', type=int, default=90, 
                        help='Number of days of historical data to use')
    parser.add_argument('--timeframe', type=str, default='5m', choices=['5m', '15m', '1h', '4h'],
                        help='Timeframe to use for training data (default: 5m)')
    parser.add_argument('--save-path', type=str, default='models/trained', 
                        help='Path to save the trained model')
    parser.add_argument('--optimize', action='store_true',
                        help='Run hyperparameter optimization')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
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
        # Calculate technical indicators for the logistic regression model
        
        # ATR for volatility filtering
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
        df['atr1'] = df['tr'].rolling(window=1).mean()
        df['atr10'] = df['tr'].rolling(window=10).mean()
        
        # Volume RSI for volume filtering
        volume_delta = df['volume'].diff()
        volume_gain = volume_delta.clip(lower=0)
        volume_loss = -volume_delta.clip(upper=0)
        avg_volume_gain = volume_gain.rolling(window=14).mean()
        avg_volume_loss = volume_loss.rolling(window=14).mean()
        rs_volume = avg_volume_gain / avg_volume_loss.replace(0, 1e-7)
        df['volume_rsi'] = 100 - (100 / (1 + rs_volume))
        
        # Additional features that might help
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        logger.info(f"Calculated indicators successfully")
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        # Fall back to minimal indicator set
        df['atr'] = df['high'] - df['low']  # Simple ATR approximation
        df['atr1'] = df['atr']
        df['atr10'] = df['atr'].rolling(window=10).mean()
        df['volume_rsi'] = 50  # Neutral volume RSI
        logger.info("Using minimal indicator set due to calculation error")
    
    # Drop rows with NaN values from indicator calculations
    original_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {original_len - len(df)} rows with NaN values from indicators")
    
    # Make sure we still have data
    if len(df) < 50:  # Need at least some data for meaningful training
        logger.error(f"Too few rows ({len(df)}) after dropping NaNs")
        raise ValueError("Insufficient data for training after preprocessing")
    
    return df

def evaluate_model_performance(model, df, test_start_idx):
    """Evaluate model performance on test data"""
    logger.info("Evaluating model performance on test data")
    
    # Use only test data
    test_df = df.iloc[test_start_idx:].copy()
    
    # Generate signals using the model
    signals = model.calculate_signals(test_df)
    
    # Get metrics from the model
    metrics = model.get_metrics()
    
    # Calculate additional metrics
    signal_df = signals['dataframe']
    
    # Calculate returns based on signals
    returns = []
    positions = []
    current_position = 0
    entry_price = None
    
    for i in range(len(signal_df)):
        signal = signal_df['signal'].iloc[i]
        price = signal_df['close'].iloc[i]
        
        positions.append(current_position)
        
        # No position and signal to buy
        if current_position == 0 and signal == 1:
            current_position = 1
            entry_price = price
            returns.append(0)
        
        # No position and signal to sell
        elif current_position == 0 and signal == -1:
            current_position = -1
            entry_price = price
            returns.append(0)
        
        # Long position
        elif current_position == 1:
            # Exit signal
            if signal == -1:
                return_val = (price - entry_price) / entry_price if entry_price else 0
                returns.append(return_val)
                current_position = 0
                entry_price = None
            else:
                returns.append(0)
        
        # Short position
        elif current_position == -1:
            # Exit signal
            if signal == 1:
                return_val = (entry_price - price) / entry_price if entry_price else 0
                returns.append(return_val)
                current_position = 0
                entry_price = None
            else:
                returns.append(0)
    
    # Close any remaining positions at the end
    if current_position != 0 and entry_price:
        last_price = signal_df['close'].iloc[-1]
        if current_position == 1:
            last_return = (last_price - entry_price) / entry_price
        else:
            last_return = (entry_price - last_price) / entry_price
        returns.append(last_return)
    
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
    
    # Combine model metrics with calculated metrics
    combined_metrics = {
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
        'model_win_rate': float(metrics.get('win_rate', 0)),
        'model_total_trades': int(metrics.get('total_trades', 0)),
        'model_cumulative_return': float(metrics.get('cumulative_return', 0))
    }
    
    logger.info("Performance Metrics:")
    for key, value in combined_metrics.items():
        logger.info(f"  {key}: {value}")
    
    return combined_metrics, signals

def optimize_hyperparameters(df, test_start_idx):
    """Optimize hyperparameters using grid search"""
    logger.info("Running hyperparameter optimization")
    
    # Define parameter grid (reduced for faster optimization)
    param_grid = {
        'lookback': [3, 5, 7],
        'learning_rate': [0.0009, 0.001],
        'iterations': [1000],
        'holding_period': [5, 7],
        'volatility_filter': [True, False],
        'volume_filter': [True, False]
    }
    
    best_score = -np.inf
    best_params = None
    best_model = None
    best_metrics = None
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    logger.info(f"Testing {len(param_combinations)} parameter combinations")
    
    for i, params in enumerate(param_combinations):
        logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
        
        try:
            # Create model with current parameters
            config = LogisticConfig(**params)
            model = LogisticRegression(config)
            
            # Evaluate model
            metrics, _ = evaluate_model_performance(model, df, test_start_idx)
            
            # Use Sharpe ratio as optimization metric
            score = metrics['sharpe_ratio']
            
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
                best_metrics = metrics
                logger.info(f"New best score: {score:.4f} with params: {params}")
        
        except Exception as e:
            logger.warning(f"Failed to evaluate parameters {params}: {str(e)}")
            continue
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best Sharpe ratio: {best_score:.4f}")
    
    return best_model, best_params, best_metrics

def save_model(model, config, metrics, save_path, timeframe):
    """Save the trained model and configuration"""
    logger.info(f"Saving model to {save_path}")
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model configuration
    config_dict = {
        'lookback': config.lookback,
        'learning_rate': config.learning_rate,
        'iterations': config.iterations,
        'norm_lookback': config.norm_lookback,
        'use_amp': config.use_amp,
        'device': config.device,
        'dtype': str(config.dtype),
        'volatility_filter': config.volatility_filter,
        'volume_filter': config.volume_filter,
        'threshold': config.threshold,
        'use_price_data': config.use_price_data,
        'holding_period': config.holding_period
    }
    
    config_file = os.path.join(save_path, f'logistic_regression_config_{timeframe}_{timestamp}.json')
    with open(config_file, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    # Save metrics
    metrics_file = os.path.join(save_path, f'logistic_regression_metrics_{timeframe}_{timestamp}.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save model state (if the model has any state to save)
    # Note: This LogisticRegression model doesn't have traditional weights to save,
    # but we save the configuration which is what's needed to recreate the model
    model_file = os.path.join(save_path, f'logistic_regression_model_{timeframe}_{timestamp}.json')
    with open(model_file, 'w') as f:
        json.dump({
            'model_type': 'logistic_regression',
            'config': config_dict,
            'training_date': timestamp,
            'timeframe': timeframe,
            'metrics': metrics
        }, f, indent=4)
    
    logger.info(f"Model saved successfully:")
    logger.info(f"  Config: {config_file}")
    logger.info(f"  Metrics: {metrics_file}")
    logger.info(f"  Model: {model_file}")
    
    return {
        'config_file': config_file,
        'metrics_file': metrics_file,
        'model_file': model_file
    }

async def main():
    """Main function to train and optimize the Logistic Regression model"""
    args = parse_arguments()
    
    # Fetch training data
    data = await fetch_training_data(days=args.data_days)
    
    # Preprocess data
    df = preprocess_data(data, timeframe=args.timeframe)
    
    logger.info(f"Data shape: {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")
    
    # Split data into train and test
    test_start_idx = int(len(df) * (1 - args.test_size))
    train_df = df.iloc[:test_start_idx]
    test_df = df.iloc[test_start_idx:]
    
    logger.info(f"Training data: {len(train_df)} rows, Test data: {len(test_df)} rows")
    
    if args.optimize:
        # Run hyperparameter optimization
        best_model, best_params, best_metrics = optimize_hyperparameters(df, test_start_idx)
        
        # Create config with best parameters
        config = LogisticConfig(**best_params)
    else:
        # Use default configuration
        config = LogisticConfig()
        logger.info("Using default configuration")
        
        # Create and evaluate model
        model = LogisticRegression(config)
        best_metrics, signals = evaluate_model_performance(model, df, test_start_idx)
        best_model = model
    
    # Save the model
    saved_files = save_model(best_model, config, best_metrics, args.save_path, args.timeframe)
    
    # Create a summary report
    summary = {
        'model_type': 'logistic_regression',
        'timeframe': args.timeframe,
        'data_days': args.data_days,
        'training_date': datetime.now().isoformat(),
        'performance': best_metrics,
        'files': saved_files
    }
    
    summary_file = os.path.join(args.save_path, f'logistic_regression_summary_{args.timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Training completed successfully!")
    logger.info(f"Summary saved to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 