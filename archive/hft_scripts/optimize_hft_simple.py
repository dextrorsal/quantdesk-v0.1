#!/usr/bin/env python3
"""
🚀 Simple HFT Model Optimization

Optimizes HFT models using existing indicators without parameter changes.
Focuses on hyperparameter tuning and feature selection.

Current Performance: 53.37% win rate
Target Performance: >60% win rate
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.models.strategy.lorentzian_classifier import LorentzianANN
from src.ml.features.rsi import RSIIndicator
from src.ml.features.adx import ADXIndicator
from src.ml.features.cci import CCIIndicator
from src.ml.features.wave_trend import WaveTrendIndicator

class SimpleHFTOptimizer:
    """Simple HFT optimizer using existing indicators."""
    
    def __init__(self, data_dir: str = "data/hft_consolidated"):
        self.data_dir = Path(data_dir)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Check GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Optimization parameters (Lorentzian model only)
        self.optimization_params = {
            'lookback_periods': [10, 15, 20, 25, 30],
            'prediction_bars': [2, 4, 6, 8],
            'k_neighbors': [10, 15, 20, 25, 30]
        }
        
        # Feature combinations to test
        self.feature_combinations = [
            ['rsi', 'rsi_buy', 'rsi_sell'],
            ['adx', 'adx_buy', 'adx_sell'],
            ['cci', 'cci_buy', 'cci_sell'],
            ['wave_trend', 'wt_buy', 'wt_sell'],
            ['price_change', 'volume_change'],
            ['volatility', 'volatility_5'],
            ['momentum_5', 'momentum_10', 'momentum_20'],
            ['high_low_ratio', 'close_open_ratio']
        ]
    
    def load_data(self, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load and prepare data for optimization."""
        data_file = self.data_dir / exchange / symbol / timeframe / f"{symbol}_{timeframe}_consolidated.csv"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        self.logger.info(f"Loaded {len(df)} rows from {data_file}")
        self.logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare all features using existing indicators."""
        self.logger.info("Preparing features...")
        
        features_df = df.copy()
        
        # Initialize indicators (using default parameters)
        rsi = RSIIndicator()
        adx = ADXIndicator()
        cci = CCIIndicator()
        wave_trend = WaveTrendIndicator()
        
        # Calculate features
        rsi_features = rsi.calculate_signals(df)
        features_df['rsi'] = rsi_features['rsi'].cpu().numpy()
        features_df['rsi_buy'] = rsi_features['buy_signals'].cpu().numpy()
        features_df['rsi_sell'] = rsi_features['sell_signals'].cpu().numpy()
        
        adx_features = adx.calculate_signals(df)
        features_df['adx'] = adx_features['adx'].cpu().numpy()
        features_df['adx_buy'] = adx_features['buy_signals'].cpu().numpy()
        features_df['adx_sell'] = adx_features['sell_signals'].cpu().numpy()
        
        cci_features = cci.calculate_signals(df)
        features_df['cci'] = cci_features['cci'].cpu().numpy()
        features_df['cci_buy'] = cci_features['buy_signals'].cpu().numpy()
        features_df['cci_sell'] = cci_features['sell_signals'].cpu().numpy()
        
        wt_features = wave_trend.calculate_signals(df)
        features_df['wave_trend'] = wt_features['wave_trend'].cpu().numpy()
        features_df['wt_buy'] = wt_features['buy_signals'].cpu().numpy()
        features_df['wt_sell'] = wt_features['sell_signals'].cpu().numpy()
        
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
    
    def create_labels(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Create labels for training."""
        # Future price change
        df['future_return'] = df['close'].shift(-lookback) / df['close'] - 1
        
        # Binary labels (1 for positive return, 0 for negative)
        df['label'] = (df['future_return'] > 0).astype(int)
        
        # Remove rows with NaN labels
        df = df.dropna(subset=['label'])
        
        return df
    
    def evaluate_model(self, model, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
        """Evaluate model performance."""
        predictions = model.predict(X_test)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0).long()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test.cpu(), binary_predictions.cpu())
        precision = precision_score(y_test.cpu(), binary_predictions.cpu(), zero_division=0)
        recall = recall_score(y_test.cpu(), binary_predictions.cpu(), zero_division=0)
        f1 = f1_score(y_test.cpu(), binary_predictions.cpu(), zero_division=0)
        
        # Win rate (same as accuracy for binary classification)
        win_rate = accuracy
        
        return {
            'accuracy': accuracy,
            'win_rate': win_rate,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def cross_validate_model(self, df: pd.DataFrame, params: Dict, 
                           feature_cols: List[str]) -> Dict:
        """Perform time series cross-validation."""
        # Prepare features and labels
        features_df = self.prepare_features(df)
        features_df = self.create_labels(features_df, params['lookback_period'])
        
        # Prepare data
        X = features_df[feature_cols].values
        y = features_df['label'].values
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        cv_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.LongTensor(y_train).to(self.device)
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_tensor = torch.LongTensor(y_test).to(self.device)
            
            # Train model
            model = LorentzianANN(
                lookback_bars=params['lookback_period'],
                prediction_bars=params['prediction_bars'],
                k_neighbors=params['k_neighbors']
            )
            
            # Fit model
            prices = torch.tensor(features_df['close'].iloc[train_idx].values, 
                                dtype=torch.float32)
            model.fit(X_train_tensor, prices)
            
            # Evaluate
            scores = self.evaluate_model(model, X_test_tensor, y_test_tensor)
            cv_scores.append(scores)
        
        # Average scores
        avg_scores = {}
        for metric in ['accuracy', 'win_rate', 'precision', 'recall', 'f1_score']:
            avg_scores[metric] = np.mean([score[metric] for score in cv_scores])
        
        return avg_scores
    
    def optimize_hyperparameters(self, df: pd.DataFrame, 
                               target_win_rate: float = 0.60) -> Dict:
        """Optimize hyperparameters to achieve target win rate."""
        self.logger.info("Starting hyperparameter optimization...")
        
        best_score = 0.0
        best_params = {}
        best_features = []
        
        # Test different feature combinations
        for n_features in range(1, len(self.feature_combinations) + 1):
            for feature_combo in itertools.combinations(self.feature_combinations, n_features):
                feature_cols = []
                for combo in feature_combo:
                    feature_cols.extend(combo)
                
                # Test hyperparameter combinations
                param_combinations = itertools.product(
                    self.optimization_params['lookback_periods'],
                    self.optimization_params['prediction_bars'],
                    self.optimization_params['k_neighbors']
                )
                
                for params_tuple in param_combinations:
                    params = {
                        'lookback_period': params_tuple[0],
                        'prediction_bars': params_tuple[1],
                        'k_neighbors': params_tuple[2]
                    }
                    
                    try:
                        scores = self.cross_validate_model(df, params, feature_cols)
                        
                        if scores['win_rate'] > best_score:
                            best_score = scores['win_rate']
                            best_params = params.copy()
                            best_features = feature_cols.copy()
                            
                            self.logger.info(f"New best score: {best_score:.4f}")
                            self.logger.info(f"Params: {best_params}")
                            self.logger.info(f"Features: {len(best_features)} features")
                            
                            # Early stopping if target achieved
                            if best_score >= target_win_rate:
                                self.logger.info(f"Target win rate {target_win_rate} achieved!")
                                return {
                                    'best_params': best_params,
                                    'best_features': best_features,
                                    'best_score': best_score,
                                    'target_achieved': True
                                }
                    
                    except Exception as e:
                        self.logger.warning(f"Error with params {params}: {e}")
                        continue
        
        return {
            'best_params': best_params,
            'best_features': best_features,
            'best_score': best_score,
            'target_achieved': best_score >= target_win_rate
        }
    
    def train_optimized_model(self, df: pd.DataFrame, optimization_result: Dict) -> Tuple[LorentzianANN, Dict]:
        """Train the optimized model on full dataset."""
        self.logger.info("Training optimized model...")
        
        # Prepare data
        features_df = self.prepare_features(df)
        features_df = self.create_labels(features_df, optimization_result['best_params']['lookback_period'])
        
        # Prepare training data
        X = features_df[optimization_result['best_features']].values
        y = features_df['label'].values
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Train model
        model = LorentzianANN(
            lookback_bars=optimization_result['best_params']['lookback_period'],
            prediction_bars=optimization_result['best_params']['prediction_bars'],
            k_neighbors=optimization_result['best_params']['k_neighbors']
        )
        
        # Fit model
        prices = torch.tensor(features_df['close'].values, dtype=torch.float32)
        model.fit(X_tensor, prices)
        
        # Final evaluation
        final_scores = self.evaluate_model(model, X_tensor, y_tensor)
        
        return model, final_scores
    
    def generate_optimization_report(self, optimization_result: Dict, 
                                   final_scores: Dict, symbol: str, timeframe: str):
        """Generate optimization report."""
        report_file = Path(f"results/hft_optimization_report_{symbol}_{timeframe}.md")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w') as f:
            f.write(f"# HFT Model Optimization Report - {symbol} {timeframe}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Optimization Results\n\n")
            f.write(f"**Target Win Rate:** 60%\n")
            f.write(f"**Achieved Win Rate:** {optimization_result['best_score']:.4f} ({optimization_result['best_score']*100:.2f}%)\n")
            f.write(f"**Target Achieved:** {'✅ Yes' if optimization_result['target_achieved'] else '❌ No'}\n\n")
            
            f.write("## Best Parameters\n\n")
            for param, value in optimization_result['best_params'].items():
                f.write(f"- **{param}:** {value}\n")
            
            f.write(f"\n## Best Features ({len(optimization_result['best_features'])} features)\n\n")
            for feature in optimization_result['best_features']:
                f.write(f"- {feature}\n")
            
            f.write(f"\n## Final Model Performance\n\n")
            f.write(f"- **Accuracy:** {final_scores['accuracy']:.4f}\n")
            f.write(f"- **Win Rate:** {final_scores['win_rate']:.4f}\n")
            f.write(f"- **Precision:** {final_scores['precision']:.4f}\n")
            f.write(f"- **Recall:** {final_scores['recall']:.4f}\n")
            f.write(f"- **F1 Score:** {final_scores['f1_score']:.4f}\n")
        
        self.logger.info(f"Optimization report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Optimize HFT models")
    parser.add_argument("--exchange", default="binance", help="Exchange to use")
    parser.add_argument("--symbol", default="BTC", help="Symbol to optimize")
    parser.add_argument("--timeframe", default="5m", help="Timeframe to optimize")
    parser.add_argument("--target-win-rate", type=float, default=0.60, 
                       help="Target win rate (default: 0.60)")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = SimpleHFTOptimizer()
    
    # Load data
    df = optimizer.load_data(args.exchange, args.symbol, args.timeframe)
    
    # Optimize hyperparameters
    optimization_result = optimizer.optimize_hyperparameters(df, args.target_win_rate)
    
    # Train optimized model
    model, final_scores = optimizer.train_optimized_model(df, optimization_result)
    
    # Generate report
    optimizer.generate_optimization_report(optimization_result, final_scores, 
                                         args.symbol, args.timeframe)
    
    # Save optimized model
    model_path = f"models/hft_optimized/{args.exchange}/{args.symbol}/{args.timeframe}/lorentzian_optimized.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    # Print summary
    print(f"\n🎯 HFT Model Optimization Complete!")
    print(f"📊 Achieved Win Rate: {optimization_result['best_score']:.4f} ({optimization_result['best_score']*100:.2f}%)")
    print(f"🎯 Target Achieved: {'✅ Yes' if optimization_result['target_achieved'] else '❌ No'}")
    print(f"📄 Report: results/hft_optimization_report_{args.symbol}_{args.timeframe}.md")
    print(f"💾 Model: {model_path}")

if __name__ == "__main__":
    main() 