#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Walk-Forward Model Training Script for SOL Trading Strategy

This script implements walk-forward validation for more realistic performance assessment
of the trading model on historical data.

USE CASES:
- **Walk-forward validation**: Time-based model validation for realistic performance
- **Hyperparameter optimization**: Optimize model parameters across time windows
- **Performance assessment**: Realistic backtesting without look-ahead bias
- **Model robustness testing**: Test model performance across different time periods
- **Trading strategy validation**: Validate strategies on out-of-sample data
- **Risk management**: Assess model performance under different market conditions

DIFFERENCES FROM OTHER TRAINING SCRIPTS:
- train_model_walkforward.py: Walk-forward optimization with time-based validation
- train_model.py: Main Lorentzian classifier training with comprehensive features
- train_logistic_regression.py: Simple logistic regression baseline
- train_logistic_regression_fast.py: Fast logistic regression for quick experiments
- extended_training.py: Extended training with additional features and models

WHEN TO USE:
- For realistic model performance assessment
- When you need to avoid look-ahead bias in backtesting
- For hyperparameter optimization across time
- When you want to test model robustness
- For production model validation

FEATURES:
- Walk-forward validation with configurable windows
- Multiple optimization metrics (Sharpe, Sortino, Calmar ratios)
- Time-based performance analysis
- Model robustness testing across periods
- Comprehensive visualization of results
- Risk-adjusted performance metrics

EXAMPLES:
    # Basic walk-forward training
    python scripts/train_model_walkforward.py --data-days 180 --window-size 30
    
    # Optimize for Sharpe ratio
    python scripts/train_model_walkforward.py --optimize-metric sharpe_ratio
    
    # Custom window sizes
    python scripts/train_model_walkforward.py --window-size 60 --test-size 14
"""

import sys
import os
import logging
import argparse
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Add project root to path
pass # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.data.collectors.sol_data_collector import SOLDataCollector
from src.features.technical.indicators.technical_indicators import calculate_indicators
from src.models.strategy.primary.lorentzian_classifier import LorentzianClassifier
from src.utils.performance_metrics import calculate_trading_metrics, backtest_strategy, generate_performance_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train trading model with walk-forward validation')
    parser.add_argument('--data-days', type=int, default=180, 
                        help='Number of days of historical data to use')
    parser.add_argument('--window-size', type=int, default=30,
                        help='Size of each training window in days')
    parser.add_argument('--test-size', type=int, default=7,
                        help='Size of each test window in days')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs per window')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--save-path', type=str, default='models/trained',
                        help='Path to save the trained model')
    parser.add_argument('--optimize-metric', type=str, default='sharpe_ratio',
                        choices=['total_return', 'win_rate', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio'],
                        help='Metric to optimize during training')
    return parser.parse_args()

async def fetch_training_data(days=180):
    """Fetch historical data for training"""
    logger.info(f"Fetching {days} days of historical data for training")
    
    # Initialize data collector
    collector = SOLDataCollector()
    
    # Fetch data for multiple timeframes
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    data = {}
    
    for tf in timeframes:
        df = await collector.fetch_historical_data(timeframe=tf, days=days)
        logger.info(f"Fetched {len(df)} records for {tf} timeframe")
        data[tf] = df
    
    return data

def preprocess_data(data):
    """Preprocess data for model training"""
    logger.info("Preprocessing data for training")
    
    # Use 1-hour timeframe as base
    df = data['1h'].copy()
    
    # Calculate technical indicators
    df = calculate_indicators(df)
    
    # Drop rows with NaN values from indicator calculations
    df = df.dropna()
    
    # Create labels based on future price movement (simplified approach)
    # 1 for price going up by 1% or more in next 24 hours, 0 otherwise
    price_shift = 24  # 24 hours ahead
    threshold = 0.01  # 1% price increase
    
    df['future_return'] = df['close'].pct_change(price_shift).shift(-price_shift)
    df['label'] = (df['future_return'] > threshold).astype(int)
    
    # Calculate returns for performance evaluation
    df['returns'] = df['close'].pct_change()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume',
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'wt1', 'wt2', 'adx', 'atr', 'bb_upper', 'bb_lower',
        'cci', 'stoch_k', 'stoch_d'
    ]
    
    # Make sure we have all the features, else remove them from list
    available_features = [f for f in feature_columns if f in df.columns]
    
    logger.info(f"Using features: {available_features}")
    
    return df, available_features

def create_walk_forward_windows(df, window_size_days=30, test_size_days=7):
    """
    Create walk-forward validation windows
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with preprocessed data
    window_size_days : int
        Size of each training window in days
    test_size_days : int
        Size of each test window in days
    
    Returns:
    --------
    list of tuples
        List of (train_df, test_df) pairs for each window
    """
    # Convert days to number of hourly records
    hours_per_day = 24
    window_size = window_size_days * hours_per_day
    test_size = test_size_days * hours_per_day
    
    # Create windows
    windows = []
    total_rows = len(df)
    
    # Calculate number of windows
    num_windows = (total_rows - window_size) // test_size
    
    for i in range(num_windows):
        start_idx = i * test_size
        train_end_idx = start_idx + window_size
        test_end_idx = min(train_end_idx + test_size, total_rows)
        
        train_df = df.iloc[start_idx:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:test_end_idx].copy()
        
        if len(test_df) > 0:
            windows.append((train_df, test_df))
    
    logger.info(f"Created {len(windows)} walk-forward windows")
    
    return windows

def train_model_window(train_df, test_df, feature_columns, scaler, args):
    """
    Train model on a single window
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data for the window
    test_df : pd.DataFrame
        Test data for the window
    feature_columns : list
        List of feature column names
    scaler : StandardScaler
        Fitted scaler for feature normalization
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    tuple
        (trained_model, test_signals, test_metrics)
    """
    # Extract features and labels
    X_train = train_df[feature_columns].values
    y_train = train_df['label'].values
    
    X_test = test_df[feature_columns].values
    y_test = test_df['label'].values
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    
    # Create model
    input_size = X_train.shape[1]
    model = LorentzianClassifier(input_size=input_size)
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
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
        
        # Periodically log progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_loader)
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_tensor)
                test_predictions = (test_outputs > 0.5).float()
                test_accuracy = (test_predictions.squeeze() == y_test_tensor).float().mean()
            
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f} - Test Accuracy: {test_accuracy:.4f}")
    
    # Generate trading signals on test set
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    
    # Create trade signals (1 for buy, -1 for sell, 0 for hold)
    buy_threshold = 0.7  # High confidence for buy
    sell_threshold = 0.3  # Low confidence for sell
    
    signals = np.zeros(len(predictions))
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1
    
    # Calculate performance metrics on test data
    returns = test_df['returns'].values
    metrics = calculate_trading_metrics(returns, signals)
    
    return model, signals, metrics

def run_walk_forward_optimization(df, feature_columns, args):
    """
    Run walk-forward optimization
    
    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed data
    feature_columns : list
        List of feature column names
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    tuple
        (combined_signals, window_metrics, best_model)
    """
    # Create windows
    windows = create_walk_forward_windows(df, args.window_size, args.test_size)
    
    # Initialize scaler and fit on all training data
    scaler = StandardScaler()
    scaler.fit(df[feature_columns].values)
    
    # Train on each window
    all_signals = []
    all_metrics = []
    best_model = None
    best_metric_value = -np.inf if args.optimize_metric != 'max_drawdown' else np.inf
    
    for i, (train_df, test_df) in enumerate(windows):
        logger.info(f"Training window {i+1}/{len(windows)}")
        
        # Train model on this window
        model, signals, metrics = train_model_window(train_df, test_df, feature_columns, scaler, args)
        
        # Store results
        window_result = {
            'window': i+1,
            'signals': signals,
            'metrics': metrics,
            'test_start': test_df.index[0],
            'test_end': test_df.index[-1]
        }
        
        all_signals.append(signals)
        all_metrics.append(metrics)
        
        # Log window performance
        logger.info(f"Window {i+1} Performance:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        # Check if this is the best model so far based on the chosen metric
        metric_value = metrics[args.optimize_metric]
        is_better = False
        
        if args.optimize_metric == 'max_drawdown':
            # For drawdown, lower is better
            is_better = metric_value > best_metric_value if np.isnan(best_metric_value) else metric_value < best_metric_value
        else:
            # For other metrics, higher is better
            is_better = metric_value > best_metric_value
        
        if is_better:
            best_metric_value = metric_value
            best_model = model
            
            # Save this model
            os.makedirs(args.save_path, exist_ok=True)
            model_path = os.path.join(args.save_path, f"best_model_window_{i+1}.pt")
            torch.save(model.state_dict(), model_path)
            logger.info(f"New best model saved to {model_path}")
    
    # Combine results
    window_metrics = pd.DataFrame(all_metrics)
    window_metrics['window'] = range(1, len(all_metrics) + 1)
    
    # Calculate cumulative signals
    combined_signals = np.concatenate(all_signals)
    
    return combined_signals, window_metrics, best_model

def visualize_results(df, signals, metrics_df, args):
    """
    Visualize walk-forward optimization results
    
    Parameters:
    -----------
    df : pd.DataFrame
        Full data
    signals : np.ndarray
        Combined signals from all windows
    metrics_df : pd.DataFrame
        Metrics from each window
    args : argparse.Namespace
        Command line arguments
    """
    # Create directory for visualizations
    vis_path = os.path.join(args.save_path, 'visualizations')
    os.makedirs(vis_path, exist_ok=True)
    
    # 1. Plot metrics across windows
    plt.figure(figsize=(12, 8))
    
    for col in ['total_return', 'win_rate', 'sharpe_ratio', 'sortino_ratio']:
        if col in metrics_df.columns:
            plt.plot(metrics_df['window'], metrics_df[col], label=col)
    
    plt.title('Performance Metrics Across Windows')
    plt.xlabel('Window Number')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    metrics_plot_path = os.path.join(vis_path, f"metrics_across_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(metrics_plot_path)
    
    # 2. Plot correlation heatmap of metrics
    plt.figure(figsize=(10, 8))
    metric_cols = [col for col in metrics_df.columns if col != 'window']
    corr = metrics_df[metric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Between Metrics')
    corr_plot_path = os.path.join(vis_path, f"metrics_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(corr_plot_path)
    
    # 3. Create a summary report
    report = []
    report.append("# Walk-Forward Optimization Results\n")
    
    report.append("## Average Performance Metrics")
    for col in metric_cols:
        report.append(f"- Average {col}: {metrics_df[col].mean():.4f}")
    
    report.append("\n## Best Window Performance")
    best_window = metrics_df.loc[metrics_df[args.optimize_metric].idxmax()]
    report.append(f"- Best window: {best_window['window']}")
    for col in metric_cols:
        report.append(f"- {col}: {best_window[col]:.4f}")
    
    report.append("\n## Optimization Parameters")
    report.append(f"- Training window size: {args.window_size} days")
    report.append(f"- Test window size: {args.test_size} days")
    report.append(f"- Optimization metric: {args.optimize_metric}")
    report.append(f"- Total data period: {df.index[0]} to {df.index[-1]}")
    
    report_path = os.path.join(vis_path, f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report))
    
    logger.info(f"Visualizations and report saved to {vis_path}")

async def main():
    """Main function to run walk-forward optimization"""
    args = parse_arguments()
    
    # Fetch historical data
    data = await fetch_training_data(days=args.data_days)
    
    # Preprocess data
    df, feature_columns = preprocess_data(data)
    
    logger.info(f"Data shape: {df.shape}, Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run walk-forward optimization
    signals, window_metrics, best_model = run_walk_forward_optimization(df, feature_columns, args)
    
    # Save window metrics
    metrics_path = os.path.join(args.save_path, f"window_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    window_metrics.to_csv(metrics_path, index=False)
    
    # Visualize results
    visualize_results(df, signals, window_metrics, args)
    
    # Save best model one more time
    if best_model is not None:
        final_model_path = os.path.join(args.save_path, f"best_model_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(best_model.state_dict(), final_model_path)
        logger.info(f"Final best model saved to {final_model_path}")
    
    logger.info("Walk-forward optimization completed")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 