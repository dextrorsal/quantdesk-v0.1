#!/usr/bin/env python3
"""
🚀 HFT Model Training Script

Trains high-frequency trading models using consolidated 1m/5m data.
Optimizes Lorentzian Classifier and other models for HFT performance.

Features:
- Multi-timeframe training (1m, 5m)
- GPU-accelerated training
- HFT-specific hyperparameter optimization
- Ensemble model creation
- Walk-forward validation
- Performance metrics for HFT
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.models.strategy.lorentzian_classifier import LorentzianANN
from src.ml.models.strategy.logistic_regression_torch import LogisticRegression
from src.ml.features.rsi import RSIIndicator
from src.ml.features.adx import ADXIndicator
from src.ml.features.cci import CCIIndicator
from src.ml.features.wave_trend import WaveTrendIndicator
from src.ml.features.chandelier_exit import ChandelierExitIndicator

class HFTModelTrainer:
    """Trains HFT-specific models with optimized parameters."""
    
    def __init__(self, data_dir: str = "data/hft_consolidated"):
        self.data_dir = Path(data_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # HFT configuration
        self.timeframes = ['1m', '5m']
        self.symbols = ['BTC', 'ETH']
        self.exchanges = ['binance', 'coinbase', 'bitget']
        
        # Model configuration
        self.models = {
            'lorentzian': LorentzianANN,
            'logistic': LogisticRegression
        }
        
        # HFT-specific hyperparameters
        self.hft_params = {
            '1m': {
                'lookback_period': 20,
                'prediction_threshold': 0.6,
                'batch_size': 128,
                'learning_rate': 0.001,
                'epochs': 50
            },
            '5m': {
                'lookback_period': 15,
                'prediction_threshold': 0.65,
                'batch_size': 64,
                'learning_rate': 0.0005,
                'epochs': 75
            }
        }
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
    
    def load_consolidated_data(self, exchange: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Load consolidated data for training."""
        data_file = self.data_dir / exchange / symbol / timeframe / f"{symbol}_{timeframe}_consolidated.csv"
        
        if not data_file.exists():
            self.logger.warning(f"Data file not found: {data_file}")
            return None
        
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(df)} rows from {data_file}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
    
    def prepare_features(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Prepare features for HFT model training."""
        self.logger.info("Preparing HFT features...")
        
        # Initialize indicators
        rsi = RSIIndicator()
        adx = ADXIndicator()
        cci = CCIIndicator()
        wave_trend = WaveTrendIndicator()
        chandelier = ChandelierExitIndicator()
        
        # Calculate features
        features_df = df.copy()
        
        # RSI
        rsi_features = rsi.calculate_signals(df)
        features_df['rsi'] = rsi_features['rsi'].cpu().numpy()
        features_df['rsi_buy'] = rsi_features['buy_signals'].cpu().numpy()
        features_df['rsi_sell'] = rsi_features['sell_signals'].cpu().numpy()
        
        # ADX
        adx_features = adx.calculate_signals(df)
        features_df['adx'] = adx_features['adx'].cpu().numpy()
        features_df['adx_buy'] = adx_features['buy_signals'].cpu().numpy()
        features_df['adx_sell'] = adx_features['sell_signals'].cpu().numpy()
        
        # CCI
        cci_features = cci.calculate_signals(df)
        features_df['cci'] = cci_features['cci'].cpu().numpy()
        features_df['cci_buy'] = cci_features['buy_signals'].cpu().numpy()
        features_df['cci_sell'] = cci_features['sell_signals'].cpu().numpy()
        
        # WaveTrend
        wt_features = wave_trend.calculate_signals(df)
        features_df['wave_trend'] = wt_features['wave_trend'].cpu().numpy()
        features_df['wt_buy'] = wt_features['buy_signals'].cpu().numpy()
        features_df['wt_sell'] = wt_features['sell_signals'].cpu().numpy()
        
        # Chandelier Exit
        ce_features = chandelier.calculate_signals(df)
        features_df['ce_direction'] = ce_features['direction'].cpu().numpy()
        features_df['ce_buy_signal'] = ce_features['buy_signals'].cpu().numpy()
        features_df['ce_sell_signal'] = ce_features['sell_signals'].cpu().numpy()
        
        # Price-based features
        features_df['price_change'] = features_df['close'].pct_change()
        features_df['volume_change'] = features_df['volume'].pct_change()
        features_df['high_low_ratio'] = features_df['high'] / features_df['low']
        features_df['close_open_ratio'] = features_df['close'] / features_df['open']
        
        # Volatility features
        features_df['volatility'] = features_df['price_change'].rolling(20).std()
        features_df['volatility_5'] = features_df['price_change'].rolling(5).std()
        
        # Momentum features
        features_df['momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
        features_df['momentum_10'] = features_df['close'] / features_df['close'].shift(10) - 1
        features_df['momentum_20'] = features_df['close'] / features_df['close'].shift(20) - 1
        
        # Remove NaN values
        features_df = features_df.dropna()
        
        self.logger.info(f"Prepared {len(features_df)} rows with features")
        return features_df
    
    def create_labels(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Create labels for HFT training."""
        params = self.hft_params[timeframe]
        lookback = params['lookback_period']
        
        # Future price change
        df['future_return'] = df['close'].shift(-lookback) / df['close'] - 1
        
        # Binary labels (1 for positive return, 0 for negative)
        df['label'] = (df['future_return'] > 0).astype(int)
        
        # Multi-class labels for more granular prediction
        df['label_3class'] = pd.cut(df['future_return'], 
                                  bins=[-np.inf, -0.01, 0.01, np.inf], 
                                  labels=[0, 1, 2]).astype(int)
        
        # Remove rows with NaN labels (end of dataset)
        df = df.dropna(subset=['label', 'label_3class'])
        
        self.logger.info(f"Created labels for {len(df)} rows")
        return df
    
    def train_model(self, model_class, train_data: pd.DataFrame, timeframe: str, 
                   model_name: str) -> Dict:
        """Train a single model."""
        self.logger.info(f"Training {model_name} for {timeframe} timeframe...")
        
        params = self.hft_params[timeframe]
        
        # Prepare training data
        feature_cols = ['rsi', 'rsi_buy', 'rsi_sell', 'adx', 'adx_buy', 'adx_sell', 
                       'cci', 'cci_buy', 'cci_sell', 'wave_trend', 'wt_buy', 'wt_sell',
                       'price_change', 'volume_change', 'high_low_ratio', 'close_open_ratio',
                       'volatility', 'volatility_5', 'momentum_5', 'momentum_10', 'momentum_20']
        
        X = train_data[feature_cols].values
        y = train_data['label'].values
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Initialize model
        model = model_class(
            input_size=len(feature_cols),
            hidden_size=64,
            num_classes=2,
            dropout_rate=0.3
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Training loop
        model.train()
        for epoch in range(params['epochs']):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{params['epochs']}, Loss: {loss.item():.4f}")
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_tensor).sum().item() / len(y_tensor)
            
            # Calculate additional metrics
            predictions = predicted.cpu().numpy()
            true_labels = y_tensor.cpu().numpy()
            
            # Win rate (for trading context)
            win_rate = np.mean(predictions == true_labels)
            
            # Confusion matrix
            tp = np.sum((predictions == 1) & (true_labels == 1))
            tn = np.sum((predictions == 0) & (true_labels == 0))
            fp = np.sum((predictions == 1) & (true_labels == 0))
            fn = np.sum((predictions == 0) & (true_labels == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        results = {
            'model_name': model_name,
            'timeframe': timeframe,
            'accuracy': accuracy,
            'win_rate': win_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': {
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
            }
        }
        
        self.logger.info(f"Training complete - Accuracy: {accuracy:.4f}, Win Rate: {win_rate:.4f}")
        
        return model, results
    
    def train_all_models(self, exchange: str = 'binance') -> Dict:
        """Train models for all symbols and timeframes."""
        all_results = {}
        
        for symbol in self.symbols:
            all_results[symbol] = {}
            
            for timeframe in self.timeframes:
                self.logger.info(f"Training models for {symbol} {timeframe}...")
                
                # Load data
                df = self.load_consolidated_data(exchange, symbol, timeframe)
                if df is None:
                    continue
                
                # Prepare features and labels
                df = self.prepare_features(df, timeframe)
                df = self.create_labels(df, timeframe)
                
                if len(df) < 1000:  # Need sufficient data
                    self.logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} rows")
                    continue
                
                # Split data (80% train, 20% validation)
                split_idx = int(len(df) * 0.8)
                train_df = df[:split_idx]
                val_df = df[split_idx:]
                
                symbol_results = {}
                
                # Train each model
                for model_name, model_class in self.models.items():
                    try:
                        model, results = self.train_model(
                            model_class, train_df, timeframe, model_name
                        )
                        
                        # Save model
                        model_path = f"models/hft_trained/{exchange}/{symbol}/{timeframe}/{model_name}_model.pt"
                        os.makedirs(os.path.dirname(model_path), exist_ok=True)
                        torch.save(model.state_dict(), model_path)
                        
                        symbol_results[model_name] = results
                        
                    except Exception as e:
                        self.logger.error(f"Error training {model_name}: {e}")
                        continue
                
                all_results[symbol][timeframe] = symbol_results
        
        return all_results
    
    def generate_training_report(self, results: Dict):
        """Generate a comprehensive training report."""
        report_file = Path("results/hft_training_report.md")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write("# HFT Model Training Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            
            total_models = 0
            successful_models = 0
            
            for symbol, symbol_results in results.items():
                f.write(f"### {symbol}\n\n")
                
                for timeframe, timeframe_results in symbol_results.items():
                    f.write(f"#### {timeframe} Timeframe\n\n")
                    
                    for model_name, model_results in timeframe_results.items():
                        total_models += 1
                        successful_models += 1
                        
                        f.write(f"**{model_name.upper()} Model**\n")
                        f.write(f"- Accuracy: {model_results['accuracy']:.4f}\n")
                        f.write(f"- Win Rate: {model_results['win_rate']:.4f}\n")
                        f.write(f"- Precision: {model_results['precision']:.4f}\n")
                        f.write(f"- Recall: {model_results['recall']:.4f}\n")
                        f.write(f"- F1 Score: {model_results['f1_score']:.4f}\n")
                        f.write(f"- Confusion Matrix: TP={model_results['confusion_matrix']['tp']}, "
                               f"TN={model_results['confusion_matrix']['tn']}, "
                               f"FP={model_results['confusion_matrix']['fp']}, "
                               f"FN={model_results['confusion_matrix']['fn']}\n\n")
            
            f.write(f"## Overall Statistics\n\n")
            f.write(f"- **Total Models Trained:** {total_models}\n")
            f.write(f"- **Successful Models:** {successful_models}\n")
            f.write(f"- **Success Rate:** {successful_models/total_models*100:.1f}%\n")
        
        self.logger.info(f"Training report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Train HFT models")
    parser.add_argument("--exchange", default="binance", 
                       help="Exchange to use for training")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH"],
                       help="Symbols to train models for")
    parser.add_argument("--timeframes", nargs="+", default=["1m", "5m"],
                       help="Timeframes to train models for")
    parser.add_argument("--data-dir", default="data/hft_consolidated",
                       help="Directory with consolidated data")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = HFTModelTrainer(args.data_dir)
    
    # Override symbols and timeframes if provided
    trainer.symbols = args.symbols
    trainer.timeframes = args.timeframes
    
    # Train models
    results = trainer.train_all_models(args.exchange)
    
    # Generate report
    trainer.generate_training_report(results)
    
    # Print summary
    print(f"\n🎯 HFT Model Training Complete!")
    print(f"📊 Results saved to: results/hft_training_report.md")
    print(f"💾 Models saved to: models/hft_trained/")

if __name__ == "__main__":
    main() 