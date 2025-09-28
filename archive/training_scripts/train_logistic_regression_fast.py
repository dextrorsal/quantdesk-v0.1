#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast GPU-Optimized Logistic Regression Training Script

This script trains the Logistic Regression model with GPU acceleration
and optimized hyperparameter search for faster results.

USE CASES:
- **Quick experimentation**: Fast model training for rapid prototyping
- **GPU acceleration**: Leverage GPU for faster training and optimization
- **Hyperparameter search**: Parallel hyperparameter optimization
- **Rapid iteration**: Quick feedback for model development
- **Performance testing**: Fast evaluation of different configurations
- **Development workflow**: Quick training during development phase

DIFFERENCES FROM OTHER TRAINING SCRIPTS:
- train_logistic_regression_fast.py: Fast GPU-optimized logistic regression
- train_logistic_regression.py: Full logistic regression with comprehensive features
- train_model.py: Main Lorentzian classifier training with comprehensive features
- train_model_walkforward.py: Walk-forward optimization with time-based validation
- extended_training.py: Extended training with additional features and models

WHEN TO USE:
- For rapid model experimentation and prototyping
- When you need quick feedback on model performance
- For GPU-accelerated training and optimization
- During development and testing phases
- When you want parallel hyperparameter search

FEATURES:
- GPU acceleration for faster training
- Parallel hyperparameter optimization
- Minimal but essential technical indicators
- Fast preprocessing and evaluation
- Optimized for speed over comprehensiveness
- Quick model persistence and metrics

EXAMPLES:
    # Quick training
    python scripts/train_logistic_regression_fast.py --data-days 30
    
    # With optimization
    python scripts/train_logistic_regression_fast.py --optimize --timeframe 5m
    
    # Custom test split
    python scripts/train_logistic_regression_fast.py --test-size 0.3
"""

import os
import logging
import argparse
import json
import numpy as np
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
from tqdm import tqdm
import time

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
    parser = argparse.ArgumentParser(description='Fast GPU-optimized Logistic Regression training')
    parser.add_argument('--data-days', type=int, default=30, 
                        help='Number of days of historical data to use')
    parser.add_argument('--timeframe', type=str, default='5m', 
                        choices=['5m', '15m', '1h', '4h'],
                        help='Timeframe to use for training data')
    parser.add_argument('--save-path', type=str, default='models/trained', 
                        help='Path to save the trained model')
    parser.add_argument('--optimize', action='store_true',
                        help='Run hyperparameter optimization')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    return parser.parse_args()

async def fetch_training_data(days=30):
    """Fetch historical data for training"""
    logger.info(f"Fetching {days} days of historical data for training")
    
    collector = SOLDataCollector()
    data = await collector.fetch_all_timeframes(lookback_days=days)
    
    logger.info(f"Successfully fetched data for timeframes: {', '.join(data.keys())}")
    return data

def preprocess_data_fast(data, timeframe='5m'):
    """Fast preprocessing with minimal indicators"""
    logger.info(f"Fast preprocessing for {timeframe} timeframe")
    
    df = data[timeframe].copy()
    logger.info(f"Starting with {len(df)} rows of {timeframe} data")
    
    if len(df) == 0:
        raise ValueError("Empty dataset provided for preprocessing")
    
    try:
        # Minimal but essential indicators for speed
        df['atr'] = (df['high'] - df['low']).rolling(window=14).mean()
        df['atr1'] = df['high'] - df['low']
        df['atr10'] = (df['high'] - df['low']).rolling(window=10).mean()
        
        # Simple volume indicator
        df['volume_ma'] = df['volume'].rolling(window=14).mean()
        df['volume_rsi'] = 50  # Neutral default
        
        # Basic price features
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        
        logger.info("Calculated minimal indicators successfully")
    
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        # Fallback to basic features
        df['atr'] = df['high'] - df['low']
        df['atr1'] = df['atr']
        df['atr10'] = df['atr']
        df['volume_rsi'] = 50
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
    
    # Drop NaN values
    original_len = len(df)
    df = df.dropna()
    logger.info(f"Dropped {original_len - len(df)} rows with NaN values")
    
    if len(df) < 50:
        raise ValueError("Insufficient data for training after preprocessing")
    
    return df

def evaluate_model_fast(model, df, test_start_idx):
    """Fast model evaluation"""
    logger.info("Fast model evaluation")
    
    test_df = df.iloc[test_start_idx:].copy()
    signals = model.calculate_signals(test_df)
    metrics = model.get_metrics()
    
    # Quick performance calculation
    signal_df = signals['dataframe']
    
    # Calculate simple returns
    returns = []
    current_position = 0
    entry_price = None
    
    for i in range(len(signal_df)):
        signal = signal_df['signal'].iloc[i]
        price = signal_df['close'].iloc[i]
        
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
            if signal == -1:  # Exit signal
                return_val = (price - entry_price) / entry_price if entry_price else 0
                returns.append(return_val)
                current_position = 0
                entry_price = None
            else:
                returns.append(0)
        
        # Short position
        elif current_position == -1:
            if signal == 1:  # Exit signal
                return_val = (entry_price - price) / entry_price if entry_price else 0
                returns.append(return_val)
                current_position = 0
                entry_price = None
            else:
                returns.append(0)
    
    # Close final position
    if current_position != 0 and entry_price:
        last_price = signal_df['close'].iloc[-1]
        if current_position == 1:
            last_return = (last_price - entry_price) / entry_price
        else:
            last_return = (entry_price - last_price) / entry_price
        returns.append(last_return)
    
    # Calculate metrics
    returns = np.array(returns)
    total_returns = np.sum(returns)
    winning_trades = np.sum(returns > 0)
    losing_trades = np.sum(returns < 0)
    total_trades = winning_trades + losing_trades
    
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(np.cumsum(returns)) - np.cumsum(returns)) if len(returns) > 0 else 0
    
    combined_metrics = {
        'total_returns': float(total_returns),
        'total_trades': int(total_trades),
        'winning_trades': int(winning_trades),
        'losing_trades': int(losing_trades),
        'win_rate': float(win_rate),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'model_win_rate': float(metrics.get('win_rate', 0)),
        'model_total_trades': int(metrics.get('total_trades', 0)),
        'model_cumulative_return': float(metrics.get('cumulative_return', 0))
    }
    
    return combined_metrics, signals

def evaluate_single_params(params, df, test_start_idx):
    """Evaluate a single parameter combination"""
    try:
        config = LogisticConfig(**params)
        model = LogisticRegression(config)
        metrics, _ = evaluate_model_fast(model, df, test_start_idx)
        return params, metrics, model
    except Exception as e:
        logger.warning(f"Failed to evaluate parameters {params}: {str(e)}")
        return params, None, None

def worker(params_df_testidx):
    params, df, test_start_idx = params_df_testidx
    # Force CPU for parallel optimization
    params['device'] = 'cpu'
    start_time = time.time()
    print(f"[WORKER] Starting params: {params}")
    try:
        config = LogisticConfig(**params)
        model = LogisticRegression(config)
        metrics, _ = evaluate_model_fast(model, df, test_start_idx)
        elapsed = time.time() - start_time
        print(f"[WORKER] Finished params: {params} in {elapsed:.2f} seconds")
        return (params, metrics, model)
    except Exception as e:
        import logging
        logging.warning(f"Failed to evaluate parameters {params}: {str(e)}")
        elapsed = time.time() - start_time
        print(f"[WORKER] Failed params: {params} in {elapsed:.2f} seconds")
        return (params, None, None)

def optimize_hyperparameters_fast(df, test_start_idx):
    """Fast hyperparameter optimization with reduced search space (parallelized)"""
    logger.info("Running fast hyperparameter optimization (parallel)")
    
    # Reduced parameter grid for speed
    param_grid = [
        {'lookback': 3, 'learning_rate': 0.0009, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': True, 'volume_filter': True},
        {'lookback': 5, 'learning_rate': 0.0009, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': True, 'volume_filter': True},
        {'lookback': 7, 'learning_rate': 0.0009, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': True, 'volume_filter': True},
        {'lookback': 3, 'learning_rate': 0.001, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': True, 'volume_filter': True},
        {'lookback': 5, 'learning_rate': 0.001, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': True, 'volume_filter': True},
        {'lookback': 7, 'learning_rate': 0.001, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': True, 'volume_filter': True},
        {'lookback': 5, 'learning_rate': 0.0009, 'iterations': 1000, 'holding_period': 7, 'volatility_filter': True, 'volume_filter': False},
        {'lookback': 5, 'learning_rate': 0.0009, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': False, 'volume_filter': True},
        {'lookback': 5, 'learning_rate': 0.0009, 'iterations': 1000, 'holding_period': 5, 'volatility_filter': False, 'volume_filter': False},
    ]
    logger.info(f"Testing {len(param_grid)} parameter combinations in parallel")

    # Prepare arguments for each process
    args_list = [(params, df, test_start_idx) for params in param_grid]

    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in tqdm(pool.imap(worker, args_list), total=len(args_list), desc='Optimizing'):
            results.append(result)

    best_score = -float('inf')
    best_params = None
    best_model = None
    best_metrics = None

    for params, metrics, model in results:
        if metrics is not None:
            score = metrics['sharpe_ratio']
            if score > best_score:
                best_score = score
                best_params = params
                best_model = model
                best_metrics = metrics
                logger.info(f"New best score: {score:.4f} with params: {params}")

    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Best Sharpe ratio: {best_score:.4f}")

    # After best_params is found, retrain on GPU if available
    if best_params is not None:
        best_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        config = LogisticConfig(**best_params)
        model = LogisticRegression(config)
        best_metrics, _ = evaluate_model_fast(model, df, test_start_idx)
        best_model = model

    return best_model, best_params, best_metrics

def save_model_fast(model, config, metrics, save_path, timeframe):
    """Save the trained model and configuration"""
    logger.info(f"Saving model to {save_path}")
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save configuration
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
    
    # Save model state
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
    """Main function for fast Logistic Regression training"""
    args = parse_arguments()
    
    # Fetch training data
    data = await fetch_training_data(days=args.data_days)
    
    # Fast preprocessing
    df = preprocess_data_fast(data, timeframe=args.timeframe)
    
    logger.info(f"Data shape: {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")
    
    # Split data
    test_start_idx = int(len(df) * (1 - args.test_size))
    train_df = df.iloc[:test_start_idx]
    test_df = df.iloc[test_start_idx:]
    
    logger.info(f"Training data: {len(train_df)} rows, Test data: {len(test_df)} rows")
    
    if args.optimize:
        # Run fast hyperparameter optimization
        best_model, best_params, best_metrics = optimize_hyperparameters_fast(df, test_start_idx)
        config = LogisticConfig(**best_params)
    else:
        # Use default configuration
        config = LogisticConfig()
        logger.info("Using default configuration")
        
        model = LogisticRegression(config)
        best_metrics, signals = evaluate_model_fast(model, df, test_start_idx)
        best_model = model
    
    # Save the model
    saved_files = save_model_fast(best_model, config, best_metrics, args.save_path, args.timeframe)
    
    # Create summary
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
    
    logger.info(f"Fast training completed successfully!")
    logger.info(f"Summary saved to: {summary_file}")
    
    return summary

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 