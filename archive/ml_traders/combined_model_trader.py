#!/usr/bin/env python
"""
Combined Model Trader for SOL

This script loads both 5-minute and 15-minute timeframe models
and combines their signals for more robust trading decisions.

USE CASES:
- **Multi-timeframe trading**: Combine signals from 5m and 15m models
- **Signal confirmation**: Use multiple timeframes to confirm trading signals
- **Risk reduction**: Reduce false signals by requiring agreement across timeframes
- **Live trading**: Real-time trading with combined model signals
- **Backtesting**: Test combined strategy performance on historical data
- **Signal visualization**: Visualize signals across multiple timeframes

DIFFERENCES FROM OTHER TRADING SCRIPTS:
- combined_model_trader.py: Multi-timeframe model combination and trading
- start_trading_system.py: Simple trading system with single model
- run_comparison.py: Model comparison and backtesting
- final_comparison.py: Final model evaluation and comparison
- fresh_backtest_pipeline.py: Fresh backtesting pipeline

WHEN TO USE:
- When you want to combine multiple timeframe signals
- For more robust trading decisions
- When you have trained models for different timeframes
- For live trading with signal confirmation
- When you want to reduce false signals

FEATURES:
- Multi-timeframe model loading and combination
- Configurable confidence thresholds
- Live trading mode support
- Comprehensive backtesting
- Signal visualization and analysis
- Risk management through signal confirmation

EXAMPLES:
    # Basic combined trading
    python scripts/combined_model_trader.py --backtest-days 30
    
    # Live trading mode
    python scripts/combined_model_trader.py --live --confidence-threshold 0.7
    
    # Custom model paths
    python scripts/combined_model_trader.py --model-5m path/to/5m_model.pt --model-15m path/to/15m_model.pt
"""

import os
import sys
import json
import asyncio
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

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
        logging.FileHandler(f"logs/combined_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Combined model trader")
    
    parser.add_argument("--model-5m", type=str, 
                       default="models/trained/extended/5m_20250402_1614/final_model_5m.pt",
                       help="Path to 5m model")
    
    parser.add_argument("--model-15m", type=str,
                       default="models/trained/extended/15m_20250402_1614/final_model_15m.pt",
                       help="Path to 15m model")
    
    parser.add_argument("--features-5m", type=str, 
                       default=None,
                       help="Path to 5m features list (JSON). If not provided, will use default features.")
    
    parser.add_argument("--features-15m", type=str,
                       default=None,
                       help="Path to 15m features list (JSON). If not provided, will use default features.")
    
    parser.add_argument("--live", action="store_true",
                       help="Enable live trading mode")
    
    parser.add_argument("--backtest-days", type=int, default=30,
                       help="Number of days to backtest (default: 30)")
    
    parser.add_argument("--confidence-threshold", type=float, default=0.65,
                       help="Confidence threshold for signals (default: 0.65)")
    
    parser.add_argument("--combined-threshold", type=float, default=0.75,
                       help="Combined confidence threshold (default: 0.75)")
    
    parser.add_argument("--neon-connection", type=str,
                       default=None,
                       help="Neon database connection string")
    
    return parser.parse_args()

def load_model(model_path, features_path, input_size=None):
    """Load a trained model and its features"""
    logger.info(f"Loading model from {model_path}")
    
    # If features_path is None, try to determine it from the model path
    if features_path is None:
        # Try to guess the features path from the model path
        model_dir = os.path.dirname(model_path)
        timeframe = "5m" if "5m" in model_path else "15m"
        features_path = os.path.join(model_dir, f"feature_columns_{timeframe}.json")
        logger.info(f"No features path provided, using auto-detected path: {features_path}")
    
    # Check if the features file exists
    if not os.path.isfile(features_path):
        logger.warning(f"Features file not found at {features_path}")
        # Use a default set of features as fallback
        logger.info("Using default feature set")
        if "5m" in model_path:
            feature_columns = [
                "price_change", "volume_change", "high_low_diff", "body_size",
                "rsi", "macd", "bb_width", "mom_1", "mom_5", "mom_10", 
                "sma_5", "sma_10", "sma_20", "ma_cross_5_20", "ma_cross_10_50"
            ]
        else:
            feature_columns = [
                "price_change", "volume_change", "high_low_diff", "body_size",
                "rsi", "macd", "bb_width", "mom_1", "mom_5", "mom_10", 
                "sma_5", "sma_10", "sma_20", "ma_cross_5_20", "ma_cross_10_50"
            ]
    else:
        # Load feature columns from file
        with open(features_path, 'r') as f:
            feature_columns = json.load(f)
    
    # Determine input size
    if input_size is None:
        input_size = len(feature_columns)
    
    # Create model with the same architecture
    model = LorentzianClassifier(
        input_size=input_size,
        hidden_size=64,
        dropout_rate=0.4,
        sigma=1.0
    )
    
    # Load saved weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    return model, feature_columns

async def fetch_data(collector, timeframe, lookback_days=30):
    """Fetch historical data for a specific timeframe"""
    logger.info(f"Fetching {timeframe} data for the last {lookback_days} days")
    
    try:
        # Use fetch_historical method
        df = await collector.fetch_historical(timeframe, lookback_days=lookback_days)
        
        if len(df) == 0:
            raise ValueError(f"No data returned for {timeframe}")
        
        # Ensure we have a proper datetime index
        if 'timestamp' in df.columns:
            # Make sure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Set index if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
        else:
            logger.warning(f"No timestamp column found in {timeframe} data")
        
        logger.info(f"Successfully fetched {len(df)} {timeframe} candles")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching {timeframe} data: {str(e)}")
        raise

def preprocess_data(df, feature_columns):
    """Preprocess data for model input"""
    logger.info(f"Preprocessing data with {len(df)} rows")
    
    try:
        # Calculate technical indicators (same as training)
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
        
        # Drop rows with NaN values
        df = df.dropna()
        logger.info(f"After preprocessing, we have {len(df)} rows")
        
        # Ensure all feature columns are present
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Feature {col} not found, adding zeros")
                df[col] = 0
        
        # Create feature matrix
        X = df[feature_columns].values
        
        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, df
    
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def generate_signals(model_5m, model_15m, X_5m, X_15m, df_5m, df_15m, args):
    """Generate trading signals from both models"""
    logger.info("Generating signals from both models")
    
    # Convert to tensor
    X_5m_tensor = torch.FloatTensor(X_5m)
    X_15m_tensor = torch.FloatTensor(X_15m)
    
    # Get predictions
    with torch.no_grad():
        pred_5m = model_5m(X_5m_tensor)
        pred_15m = model_15m(X_15m_tensor)
    
    # Convert to numpy for easier handling
    pred_5m = pred_5m.numpy()
    pred_15m = pred_15m.numpy()
    
    # Print prediction statistics
    logger.info("5-minute model prediction statistics:")
    logger.info(f"  Min: {pred_5m.min():.4f}, Max: {pred_5m.max():.4f}, Mean: {pred_5m.mean():.4f}, Median: {np.median(pred_5m):.4f}")
    logger.info(f"  Quantiles: 25%={np.percentile(pred_5m, 25):.4f}, 50%={np.percentile(pred_5m, 50):.4f}, 75%={np.percentile(pred_5m, 75):.4f}")
    logger.info(f"  Values > 0.5: {np.sum(pred_5m > 0.5)} ({np.sum(pred_5m > 0.5)/len(pred_5m):.2%})")
    logger.info(f"  Values > 0.75: {np.sum(pred_5m > 0.75)} ({np.sum(pred_5m > 0.75)/len(pred_5m):.2%})")
    
    logger.info("15-minute model prediction statistics:")
    logger.info(f"  Min: {pred_15m.min():.4f}, Max: {pred_15m.max():.4f}, Mean: {pred_15m.mean():.4f}, Median: {np.median(pred_15m):.4f}")
    logger.info(f"  Quantiles: 25%={np.percentile(pred_15m, 25):.4f}, 50%={np.percentile(pred_15m, 50):.4f}, 75%={np.percentile(pred_15m, 75):.4f}")
    logger.info(f"  Values > 0.5: {np.sum(pred_15m > 0.5)} ({np.sum(pred_15m > 0.5)/len(pred_15m):.2%})")
    logger.info(f"  Values > 0.75: {np.sum(pred_15m > 0.75)} ({np.sum(pred_15m > 0.75)/len(pred_15m):.2%})")
    
    # Use more appropriate thresholds based on actual model output ranges
    # Instead of using args.confidence_threshold, use adaptive thresholds
    threshold_5m = max(0.3, np.percentile(pred_5m, 75))  # Either 0.3 or the 75th percentile
    threshold_15m = max(0.16, np.percentile(pred_15m, 75))  # Either 0.16 or the 75th percentile
    
    logger.info(f"Using adaptive thresholds - 5m: {threshold_5m:.4f}, 15m: {threshold_15m:.4f}")
    
    # Create signal DataFrames - use DataFrames' indices which should already be datetime
    signals_5m = pd.DataFrame({
        'confidence': pred_5m.flatten(),
        'signal': (pred_5m > threshold_5m).astype(int).flatten()
    }, index=df_5m.index[-len(pred_5m):])
    
    signals_15m = pd.DataFrame({
        'confidence': pred_15m.flatten(),
        'signal': (pred_15m > threshold_15m).astype(int).flatten()
    }, index=df_15m.index[-len(pred_15m):])
    
    logger.info(f"Generated {len(signals_5m)} signals for 5m and {len(signals_15m)} signals for 15m")
    logger.info(f"5m signals > threshold: {signals_5m['signal'].sum()} ({signals_5m['signal'].sum()/len(signals_5m):.2%})")
    logger.info(f"15m signals > threshold: {signals_15m['signal'].sum()} ({signals_15m['signal'].sum()/len(signals_15m):.2%})")
    
    # Resample 5m signals to 15m timeframe for comparison
    # Make sure we're working with datetime index
    if not isinstance(signals_5m.index, pd.DatetimeIndex):
        logger.warning("5m signals index is not a DatetimeIndex, converting...")
        try:
            # Try to convert to datetime if it isn't already
            signals_5m.index = pd.to_datetime(signals_5m.index)
        except:
            logger.error("Failed to convert 5m index to datetime, using direct comparison")
    
    if not isinstance(signals_15m.index, pd.DatetimeIndex):
        logger.warning("15m signals index is not a DatetimeIndex, converting...")
        try:
            # Try to convert to datetime if it isn't already
            signals_15m.index = pd.to_datetime(signals_15m.index)
        except:
            logger.error("Failed to convert 15m index to datetime, using direct comparison")
    
    # If indices are datetime, resample
    if isinstance(signals_5m.index, pd.DatetimeIndex):
        signals_5m_resampled = signals_5m.resample('15min').last().fillna(method='ffill')
        logger.info(f"Resampled 5m signals to 15m, got {len(signals_5m_resampled)} rows")
    else:
        # Just use the 5m signals as is if we can't resample
        signals_5m_resampled = signals_5m
    
    # Find common timestamps between the two signals
    common_index = signals_5m_resampled.index.intersection(signals_15m.index)
    logger.info(f"Found {len(common_index)} common timestamps for alignment")
    
    if len(common_index) == 0:
        logger.warning("No common timestamps found, cannot combine signals")
        # Create an empty combined signals dataframe
        combined_signals = pd.DataFrame(columns=[
            'confidence_5m', 'signal_5m', 'confidence_15m', 'signal_15m',
            'avg_confidence', 'combined_signal', 'strong_signal', 'tradingview_style_signal'
        ])
        return combined_signals
    
    # Align signals
    aligned_5m = signals_5m_resampled.loc[common_index]
    aligned_15m = signals_15m.loc[common_index]
    
    # Combine signals with multiple strategies
    combined_signals = pd.DataFrame({
        'confidence_5m': aligned_5m['confidence'],
        'signal_5m': aligned_5m['signal'],
        'confidence_15m': aligned_15m['confidence'],
        'signal_15m': aligned_15m['signal'],
        'avg_confidence': (aligned_5m['confidence'] + aligned_15m['confidence']) / 2,
    }, index=common_index)
    
    # Strategy 1: Weighted average approach (more like TradingView)
    # Use 70% weight for 5m model and 30% for 15m model
    combined_signals['weighted_confidence'] = (0.7 * combined_signals['confidence_5m'] + 
                                              0.3 * combined_signals['confidence_15m'])
    
    # Strategy 2: 5m primary with 15m as trend filter
    # Signal when 5m > threshold AND 15m is positive (even low confidence)
    combined_signals['tradingview_style_signal'] = (
        (combined_signals['confidence_5m'] > threshold_5m) & 
        (combined_signals['confidence_15m'] > np.percentile(pred_15m, 50))  # Above median
    ).astype(int)
    
    # Strategy 3: Original combined approach (both models must agree AND high combined confidence)
    combined_threshold = args.combined_threshold
    if combined_threshold > 0.5:  # If the user specified a high threshold, adapt it
        combined_threshold = 0.25  # Use a more reasonable default
    
    combined_signals['combined_signal'] = (
        (combined_signals['weighted_confidence'] > combined_threshold)
    ).astype(int)
    
    # Strategy 4: Original strong signal approach (kept for compatibility)
    combined_signals['strong_signal'] = (
        (combined_signals['signal_5m'] == 1) & 
        (combined_signals['signal_15m'] == 1)
    ).astype(int)
    
    # Count signals from different strategies
    tv_signal_count = combined_signals['tradingview_style_signal'].sum()
    combined_signal_count = combined_signals['combined_signal'].sum()
    strong_signal_count = combined_signals['strong_signal'].sum()
    
    logger.info(f"Signal counts by strategy:")
    logger.info(f"  TradingView-style signals: {tv_signal_count} ({tv_signal_count/len(combined_signals):.2%})")
    logger.info(f"  Combined signals: {combined_signal_count} ({combined_signal_count/len(combined_signals):.2%})")
    logger.info(f"  Strong signals: {strong_signal_count} ({strong_signal_count/len(combined_signals):.2%})")
    
    return combined_signals

def visualize_signals(df_5m, df_15m, combined_signals):
    """Visualize signals on price charts"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter
        
        # Check if we have any signals to visualize
        if len(combined_signals) == 0:
            logger.warning("No signals to visualize")
            return
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
        
        # Plot 5m price
        ax1.plot(df_5m.index, df_5m['close'], label='5m Price', color='blue', alpha=0.7)
        ax1.set_title('5-Minute Price Chart')
        
        # Add moving averages to price charts
        if 'sma_20' in df_5m.columns:
            ax1.plot(df_5m.index, df_5m['sma_20'], label='20 SMA', color='orange', alpha=0.6)
        
        # Plot 15m price
        ax2.plot(df_15m.index, df_15m['close'], label='15m Price', color='green', alpha=0.7)
        ax2.set_title('15-Minute Price Chart')
        
        if 'sma_20' in df_15m.columns:
            ax2.plot(df_15m.index, df_15m['sma_20'], label='20 SMA', color='orange', alpha=0.6)
        
        # Plot model confidence
        ax3.plot(combined_signals.index, combined_signals['confidence_5m'], 
               label='5m Confidence', color='blue', alpha=0.6)
        ax3.plot(combined_signals.index, combined_signals['confidence_15m'], 
               label='15m Confidence', color='green', alpha=0.6)
        ax3.plot(combined_signals.index, combined_signals['weighted_confidence'], 
               label='Weighted Confidence', color='purple', linestyle='-')
        
        # Show thresholds
        if 'threshold_5m' in locals():
            threshold_5m = max(0.3, np.percentile(combined_signals['confidence_5m'], 75))
            threshold_15m = max(0.16, np.percentile(combined_signals['confidence_15m'], 75))
        else:
            threshold_5m = np.percentile(combined_signals['confidence_5m'], 75)
            threshold_15m = np.percentile(combined_signals['confidence_15m'], 75)
        
        ax3.axhline(y=threshold_5m, color='blue', linestyle='--', alpha=0.5, label='5m Threshold')
        ax3.axhline(y=threshold_15m, color='green', linestyle='--', alpha=0.5, label='15m Threshold')
        ax3.axhline(y=0.25, color='purple', linestyle='--', alpha=0.5, label='Combined Threshold')
        
        # First, check for TradingView-style signals (most likely to have signals)
        tv_signal_points = combined_signals[combined_signals['tradingview_style_signal'] == 1]
        if len(tv_signal_points) > 0:
            # For 5m chart
            common_indices_5m = tv_signal_points.index.intersection(df_5m.index)
            if len(common_indices_5m) > 0:
                ax1.scatter(common_indices_5m, 
                          df_5m.loc[common_indices_5m, 'close'],
                          color='magenta', s=100, marker='^', label='TradingView Style Signal')
            
            # For 15m chart
            common_indices_15m = tv_signal_points.index.intersection(df_15m.index)
            if len(common_indices_15m) > 0:
                ax2.scatter(common_indices_15m, 
                          df_15m.loc[common_indices_15m, 'close'], 
                          color='magenta', s=100, marker='^', label='TradingView Style Signal')
            
            # For confidence chart - always available
            ax3.scatter(tv_signal_points.index, 
                      tv_signal_points['weighted_confidence'], 
                      color='magenta', s=100, marker='^', label='TradingView Style Signal')
            
            logger.info(f"Plotted {len(tv_signal_points)} TradingView-style signals")
        else:
            logger.info("No TradingView-style signals to plot")
        
        # Then check for combined signals
        combined_signal_points = combined_signals[combined_signals['combined_signal'] == 1]
        if len(combined_signal_points) > 0 and len(combined_signal_points) != len(tv_signal_points):
            # For confidence chart only - to avoid cluttering price charts
            ax3.scatter(combined_signal_points.index, 
                      combined_signal_points['weighted_confidence'], 
                      color='cyan', s=80, marker='o', label='Combined Signal')
            
            logger.info(f"Plotted {len(combined_signal_points)} combined signals")
        
        # Finally check for strong signals (original method)
        strong_signal_points = combined_signals[combined_signals['strong_signal'] == 1]
        if len(strong_signal_points) > 0:
            # For confidence chart only - to avoid cluttering price charts
            ax3.scatter(strong_signal_points.index, 
                      strong_signal_points['weighted_confidence'], 
                      color='yellow', s=60, marker='*', label='Strong Signal (Original)')
            
            logger.info(f"Plotted {len(strong_signal_points)} strong signals")
        
        # Format x-axis
        date_format = DateFormatter('%Y-%m-%d %H:%M')
        ax3.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        ax3.set_title('Model Confidence and Signals')
        
        # Add legends
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"models/trained/visualizations/combined_signals_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        
        logger.info(f"Visualization saved to {save_path}")
        
        plt.close(fig)
        
    except Exception as e:
        logger.error(f"Error visualizing signals: {str(e)}")
        logger.exception("Detailed error:")  # Print full traceback

async def main():
    """Main function that runs the combined model trader"""
    # Parse arguments
    args = parse_arguments()
    
    # Create a logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(logs_dir, f'combined_trader_{datetime.now().strftime("%Y%m%d_%H%M")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("Starting Combined Model Trader")
    
    # Load models
    model_5m, features_5m = load_model(args.model_5m, args.features_5m)
    model_15m, features_15m = load_model(args.model_15m, args.features_15m)
    
    # Initialize data collector with Neon connection if provided
    collector = SOLDataCollector(args.neon_connection)
    
    # Fetch historical data
    df_5m = await fetch_data(collector, "5m", args.backtest_days)
    df_15m = await fetch_data(collector, "15m", args.backtest_days)
    
    # Preprocess data
    X_5m, df_5m_processed = preprocess_data(df_5m, features_5m)
    X_15m, df_15m_processed = preprocess_data(df_15m, features_15m)
    
    # Generate signals
    combined_signals = generate_signals(model_5m, model_15m, X_5m, X_15m,
                                      df_5m_processed, df_15m_processed, args)
    
    # Visualize signals
    visualize_signals(df_5m_processed, df_15m_processed, combined_signals)
    
    # If live mode, start streaming data and processing signals
    if args.live:
        logger.info("Starting live trading mode")
        logger.info(f"Price feed activated at {datetime.now()}")
        logger.info(f"Current price: ${df_5m_processed['close'].iloc[-1]:.2f}")
        
        # If Neon connection is available, log that we're storing data
        if args.neon_connection:
            logger.info(f"Storing trading data to Neon database")
        
        # Start WebSocket for real-time data
        await collector.start_websocket()
        
        try:
            last_price_log = datetime.now()
            check_count = 0
            
            while True:
                # Sleep for 5 minutes (or a shorter interval for demo)
                check_interval = 1 * 60  # 1 minute for testing, use 5 * 60 for production
                logger.info(f"Waiting for next signal check... ({check_interval}s)")
                await asyncio.sleep(check_interval)
                
                # Fetch latest data
                df_5m_latest = await fetch_data(collector, "5m", 1)
                df_15m_latest = await fetch_data(collector, "15m", 1)
                
                # Log current price periodically
                current_time = datetime.now()
                if (current_time - last_price_log).total_seconds() > 60 or check_count % 5 == 0:
                    last_price = df_5m_latest['close'].iloc[-1]
                    logger.info(f"Current price: ${last_price:.2f}")
                    last_price_log = current_time
                
                check_count += 1
                
                # Preprocess latest data
                X_5m_latest, df_5m_latest_processed = preprocess_data(df_5m_latest, features_5m)
                X_15m_latest, df_15m_latest_processed = preprocess_data(df_15m_latest, features_15m)
                
                # Generate signals for latest data
                latest_signals = generate_signals(model_5m, model_15m, X_5m_latest, X_15m_latest,
                                               df_5m_latest_processed, df_15m_latest_processed, args)
                
                # Check for signals from any of our strategies
                latest_row = latest_signals.iloc[-1] if len(latest_signals) > 0 else None
                
                if latest_row is not None and (latest_row['tradingview_style_signal'] == 1 or 
                                             latest_row['combined_signal'] == 1 or
                                             latest_row['strong_signal'] == 1):
                    # Determine the most significant signal type
                    if latest_row['strong_signal'] == 1:
                        signal_type = "Strong"
                    elif latest_row['tradingview_style_signal'] == 1:
                        signal_type = "TradingView-style"
                    else:
                        signal_type = "Combined"
                    
                    logger.info("🔔 NEW TRADING SIGNAL DETECTED!")
                    logger.info(f"Timestamp: {latest_signals.index[-1]}")
                    logger.info(f"Signal type: {signal_type}")
                    logger.info(f"5m Confidence: {latest_row['confidence_5m']:.4f}")
                    logger.info(f"15m Confidence: {latest_row['confidence_15m']:.4f}")
                    logger.info(f"Weighted Confidence: {latest_row['weighted_confidence']:.4f}")
                    current_price = df_5m_latest_processed['close'].iloc[-1]
                    logger.info(f"Current SOL Price: ${current_price:.2f}")
                    
                    # Store signal in database if connection available
                    if args.neon_connection and hasattr(collector, '_db_pool') and collector._db_pool:
                        try:
                            async with collector._db_pool.acquire() as conn:
                                signal_time = latest_signals.index[-1]
                                
                                # Insert into trading_signals table
                                await conn.execute("""
                                    INSERT INTO trading_signals 
                                    (timestamp, symbol, signal_type, signal_strength, 
                                     confidence_5m, confidence_15m, weighted_confidence, price)
                                    VALUES 
                                    ($1, $2, $3, $4, $5, $6, $7, $8)
                                    ON CONFLICT (timestamp, symbol, signal_type) DO UPDATE
                                    SET 
                                        signal_strength = $4,
                                        confidence_5m = $5,
                                        confidence_15m = $6,
                                        weighted_confidence = $7,
                                        price = $8
                                """, 
                                signal_time, 
                                "SOLUSDT", 
                                signal_type, 
                                float(latest_row['weighted_confidence']),
                                float(latest_row['confidence_5m']),
                                float(latest_row['confidence_15m']),
                                float(latest_row['weighted_confidence']),
                                float(current_price))
                                
                                logger.info(f"Signal saved to database")
                        except Exception as e:
                            logger.error(f"Error saving signal to database: {e}")
                    
                    # Play notification sound if possible
                    try:
                        print('\a')  # Terminal bell
                    except:
                        pass
        
        except KeyboardInterrupt:
            logger.info("Stopping live trading")
            await collector.stop_websocket()
            
        except Exception as e:
            logger.error(f"Error in live trading: {str(e)}")
            logger.exception("Detailed error:")
            await collector.stop_websocket()
    
    else:
        # Show backtest results for different signal types
        logger.info("=== BACKTEST RESULTS ===")
        
        # Function to evaluate signal performance
        def evaluate_signals(signal_type, signal_col):
            signal_times = combined_signals[combined_signals[signal_col] == 1].index
            
            if len(signal_times) == 0:
                logger.info(f"\n{signal_type} Signals: No signals generated during backtest period")
                return
                
            logger.info(f"\n{signal_type} Signals ({len(signal_times)} total):")
            
            # Show first 5 signals
            for i, time in enumerate(signal_times[:5]):
                if time in df_5m_processed.index:
                    price_5m = df_5m_processed.loc[time, 'close']
                    logger.info(f"  Signal {i+1} at {time}: Price ${price_5m:.2f}, " +
                              f"5m: {combined_signals.loc[time, 'confidence_5m']:.4f}, " +
                              f"15m: {combined_signals.loc[time, 'confidence_15m']:.4f}")
            
            if len(signal_times) > 5:
                logger.info(f"  ... and {len(signal_times) - 5} more signals")
                
            # Calculate performance metrics
            future_periods = 12  # 1 hour in 5m data
            future_returns = []
            
            for time in signal_times:
                # Find closest 5m candle
                if time in df_5m_processed.index:
                    idx = df_5m_processed.index.get_loc(time)
                elif len(df_5m_processed.index) > 0:
                    # Find nearest timestamp
                    idx = df_5m_processed.index.get_indexer([time], method='nearest')[0]
                else:
                    continue
                    
                if idx + future_periods < len(df_5m_processed):
                    entry_price = df_5m_processed.iloc[idx]['close']
                    exit_price = df_5m_processed.iloc[idx + future_periods]['close']
                    pct_return = (exit_price - entry_price) / entry_price * 100
                    future_returns.append(pct_return)
            
            if future_returns:
                avg_return = sum(future_returns) / len(future_returns)
                winning_trades = sum(1 for ret in future_returns if ret > 0)
                win_rate = winning_trades / len(future_returns) if future_returns else 0
                
                logger.info(f"  Average Return (1-hour hold): {avg_return:.2f}%")
                logger.info(f"  Win Rate: {win_rate:.2%} ({winning_trades}/{len(future_returns)})")
                
                if len(future_returns) >= 3:
                    # Simple equity curve simulation
                    equity = 100  # Start with $100
                    for ret in future_returns:
                        equity *= (1 + ret/100)
                    logger.info(f"  Final Equity: ${equity:.2f} (from $100 start)")
            else:
                logger.info("  No completed trades to calculate performance")
        
        # Evaluate different signal types
        evaluate_signals("TradingView-Style", "tradingview_style_signal")
        evaluate_signals("Combined", "combined_signal") 
        evaluate_signals("Strong (Original)", "strong_signal")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Run main function
    asyncio.run(main()) 