#!/usr/bin/env python3
"""
🧠 Real Data ML-Optimized Meme Coin HFT System

Uses real CSV data from multiple exchanges to train ML models on actual market conditions.
This will be MUCH more powerful than synthetic data!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timedelta
import time
import random
from dataclasses import dataclass
from collections import deque
import glob
from pathlib import Path

from src.ml.features.market_structure import MarketStructureDetector, MarketStructureConfig
from src.trading.leverage_manager import LeverageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingParams:
    """Trading parameters optimized for real data."""
    entry_threshold: float = 0.15
    exit_threshold: float = 0.05
    bos_weight: float = 0.35
    choch_weight: float = 0.25
    order_block_weight: float = 0.25
    fvg_weight: float = 0.15
    max_leverage: float = 25.0
    position_size_pct: float = 0.1
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15

class RealDataLoader:
    """Load and process real CSV data from multiple exchanges."""
    
    def __init__(self, data_dir: str = "data/historical/processed"):
        self.data_dir = data_dir
        self.exchanges = ['binance', 'bitget', 'coinbase', 'kraken', 'kucoin', 'mexc']
        self.meme_coins = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA', 'BOME', 'PEPE', 'SHIB', 'FLOKI']
        
        logger.info(f"📊 Real Data Loader initialized for {len(self.exchanges)} exchanges")
        logger.info(f"🎯 Targeting {len(self.meme_coins)} meme coins")
    
    def find_csv_files(self) -> Dict[str, List[str]]:
        """Find all CSV files for meme coins across exchanges."""
        csv_files = {}
        
        for exchange in self.exchanges:
            exchange_files = []
            exchange_path = os.path.join(self.data_dir, exchange)
            
            if os.path.exists(exchange_path):
                # Look for meme coin directories
                for coin in self.meme_coins:
                    coin_path = os.path.join(exchange_path, coin)
                    if os.path.exists(coin_path):
                        # Find CSV files in coin directory
                        csv_pattern = os.path.join(coin_path, "**/*.csv")
                        files = glob.glob(csv_pattern, recursive=True)
                        exchange_files.extend(files)
                
                if exchange_files:
                    csv_files[exchange] = exchange_files
                    logger.info(f"📁 {exchange}: Found {len(exchange_files)} CSV files")
        
        total_files = sum(len(files) for files in csv_files.values())
        logger.info(f"📊 Total CSV files found: {total_files}")
        
        return csv_files
    
    def load_csv_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Load and standardize a single CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Standardize column names
            column_mapping = {
                'timestamp': ['timestamp', 'time', 'date', 'datetime'],
                'open': ['open', 'open_price'],
                'high': ['high', 'high_price'],
                'low': ['low', 'low_price'],
                'close': ['close', 'close_price', 'price'],
                'volume': ['volume', 'vol', 'volume_usd', 'volume_btc']
            }
            
            # Map columns
            for standard_name, possible_names in column_mapping.items():
                for col in possible_names:
                    if col in df.columns:
                        df = df.rename(columns={col: standard_name})
                        break
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.warning(f"⚠️ Missing columns in {file_path}: {missing_cols}")
                return None
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with NaN values
            df = df.dropna()
            
            if len(df) < 50:  # Need minimum data points
                logger.warning(f"⚠️ Insufficient data in {file_path}: {len(df)} rows")
                return None
            
            logger.info(f"✅ Loaded {file_path}: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            return None
    
    def load_all_data(self, max_files_per_coin: int = 10) -> Dict[str, List[pd.DataFrame]]:
        """Load all available CSV data for training."""
        csv_files = self.find_csv_files()
        all_data = {}
        
        for exchange, files in csv_files.items():
            exchange_data = []
            
            # Group files by coin
            coin_files = {}
            for file_path in files:
                # Extract coin name from path
                path_parts = file_path.split(os.sep)
                for coin in self.meme_coins:
                    if coin in path_parts:
                        if coin not in coin_files:
                            coin_files[coin] = []
                        coin_files[coin].append(file_path)
                        break
            
            # Load data for each coin
            for coin, coin_file_list in coin_files.items():
                # Limit files per coin to avoid memory issues
                files_to_load = coin_file_list[:max_files_per_coin]
                
                for file_path in files_to_load:
                    df = self.load_csv_data(file_path)
                    if df is not None:
                        # Add metadata
                        df['exchange'] = exchange
                        df['symbol'] = coin
                        exchange_data.append(df)
            
            if exchange_data:
                all_data[exchange] = exchange_data
                total_rows = sum(len(df) for df in exchange_data)
                logger.info(f"📊 {exchange}: Loaded {len(exchange_data)} datasets, {total_rows} total rows")
        
        return all_data
    
    def create_training_dataset(self, all_data: Dict[str, List[pd.DataFrame]], 
                               min_data_points: int = 1000) -> List[Dict]:
        """Create training dataset from real data."""
        training_data = []
        
        for exchange, datasets in all_data.items():
            for df in datasets:
                if len(df) < min_data_points:
                    continue
                
                # Create sliding windows for training
                window_size = 100
                step_size = 10
                
                for i in range(0, len(df) - window_size, step_size):
                    window_df = df.iloc[i:i+window_size].copy()
                    
                    # Create training sample
                    sample = {
                        'exchange': exchange,
                        'symbol': df['symbol'].iloc[0],
                        'data': window_df,
                        'start_time': window_df['timestamp'].iloc[0],
                        'end_time': window_df['timestamp'].iloc[-1],
                        'price_change': (window_df['close'].iloc[-1] - window_df['close'].iloc[0]) / window_df['close'].iloc[0],
                        'volatility': window_df['close'].pct_change().std(),
                        'volume_trend': window_df['volume'].iloc[-10:].mean() / window_df['volume'].iloc[:10].mean()
                    }
                    
                    training_data.append(sample)
        
        logger.info(f"🎯 Created {len(training_data)} training samples from real data")
        return training_data

class AdvancedSignalPredictor(nn.Module):
    """Advanced neural network for signal prediction using real data."""
    
    def __init__(self, input_size: int = 13, hidden_size: int = 128):
        super(AdvancedSignalPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 4),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_size // 4, 3),  # buy, sell, hold
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class RealDataMLTrader:
    """ML-optimized trader using real CSV data."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        # Initialize components
        self.market_detector = MarketStructureDetector()
        self.leverage_manager = LeverageManager()
        self.data_loader = RealDataLoader()
        
        # ML components
        self.signal_predictor = AdvancedSignalPredictor().to(self.device)
        self.optimizer = optim.Adam(self.signal_predictor.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training data
        self.training_data = []
        self.validation_data = []
        
        logger.info("🧠 Real Data ML Trader initialized!")
    
    def load_real_data(self):
        """Load real CSV data for training."""
        logger.info("📊 Loading real CSV data...")
        
        # Load all available data
        all_data = self.data_loader.load_all_data(max_files_per_coin=5)
        
        if not all_data:
            logger.error("❌ No CSV data found! Please check data directory.")
            return False
        
        # Create training dataset
        self.training_data = self.data_loader.create_training_dataset(all_data)
        
        if not self.training_data:
            logger.error("❌ No training data created!")
            return False
        
        # Split into training and validation
        random.shuffle(self.training_data)
        split_idx = int(len(self.training_data) * 0.8)
        self.validation_data = self.training_data[split_idx:]
        self.training_data = self.training_data[:split_idx]
        
        logger.info(f"✅ Loaded {len(self.training_data)} training samples, {len(self.validation_data)} validation samples")
        return True
    
    def extract_advanced_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract advanced features from real market data."""
        features = []
        
        # Market structure signals
        signals = self.market_detector.calculate_signals(df)
        
        # Basic market structure features
        for key in ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
                   'order_block_bullish', 'order_block_bearish', 'fvg_bullish', 'fvg_bearish']:
            tensor = signals.get(key, torch.tensor([]))
            if len(tensor) > 0:
                features.append(tensor[-1].item())
            else:
                features.append(0.0)
        
        # Price action features
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volumes = df['volume'].values
        
        # Price momentum
        price_change_1h = (close_prices[-1] - close_prices[-60]) / close_prices[-60] if len(close_prices) > 60 else 0
        price_change_4h = (close_prices[-1] - close_prices[-240]) / close_prices[-240] if len(close_prices) > 240 else 0
        
        # Volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Volume features
        volume_ma = np.mean(volumes[-20:]) if len(volumes) > 20 else np.mean(volumes)
        volume_ratio = volumes[-1] / volume_ma if volume_ma > 0 else 1
        
        # RSI-like momentum
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) > 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) > 14 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Add all features
        features.extend([
            price_change_1h,
            price_change_4h,
            volatility,
            volume_ratio,
            rsi / 100  # Normalize to 0-1
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def train_on_real_data(self, epochs: int = 100, batch_size: int = 32):
        """Train the neural network on real data."""
        logger.info(f"🧠 Training on real data for {epochs} epochs...")
        
        if not self.training_data:
            logger.error("❌ No training data available!")
            return False
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            # Shuffle training data
            random.shuffle(self.training_data)
            
            # Process in batches
            for i in range(0, len(self.training_data), batch_size):
                batch = self.training_data[i:i+batch_size]
                
                batch_features = []
                batch_targets = []
                
                for sample in batch:
                    try:
                        # Extract features
                        features = self.extract_advanced_features(sample['data'])
                        batch_features.append(features)
                        
                        # Create target based on price movement
                        price_change = sample['price_change']
                        if price_change > 0.05:  # 5% gain = buy
                            target = torch.tensor([1], dtype=torch.long)  # Buy
                        elif price_change < -0.05:  # 5% loss = sell
                            target = torch.tensor([2], dtype=torch.long)  # Sell
                        else:
                            target = torch.tensor([0], dtype=torch.long)  # Hold
                        
                        batch_targets.append(target)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error processing sample: {e}")
                        continue
                
                if not batch_features:
                    continue
                
                # Convert to tensors
                features_tensor = torch.stack(batch_features).to(self.device)
                targets_tensor = torch.stack(batch_targets).to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.signal_predictor(features_tensor)
                loss = self.criterion(outputs, targets_tensor.squeeze())
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = torch.argmax(outputs, dim=1)
                correct_predictions += (predicted == targets_tensor.squeeze()).sum().item()
                total_predictions += len(batch_features)
            
            # Validation
            if (epoch + 1) % 20 == 0:
                val_accuracy = self.validate_model()
                avg_loss = total_loss / (len(self.training_data) // batch_size)
                train_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, "
                          f"Train Acc = {train_accuracy:.2f}, Val Acc = {val_accuracy:.2f}")
        
        logger.info("✅ Training on real data complete!")
        return True
    
    def validate_model(self) -> float:
        """Validate model on validation data."""
        if not self.validation_data:
            return 0.0
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sample in self.validation_data[:100]:  # Use subset for speed
                try:
                    features = self.extract_advanced_features(sample['data'])
                    features = features.unsqueeze(0).to(self.device)
                    
                    output = self.signal_predictor(features)
                    predicted = torch.argmax(output, dim=1)
                    
                    # Create target
                    price_change = sample['price_change']
                    if price_change > 0.05:
                        target = 1
                    elif price_change < -0.05:
                        target = 2
                    else:
                        target = 0
                    
                    if predicted.item() == target:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    continue
        
        return correct / total if total > 0 else 0.0
    
    def optimize_parameters_real_data(self) -> TradingParams:
        """Optimize parameters using real data patterns."""
        logger.info("🧬 Optimizing parameters based on real data...")
        
        # Analyze real data patterns
        entry_thresholds = []
        exit_thresholds = []
        signal_strengths = []
        
        for sample in self.training_data[:100]:  # Analyze subset
            try:
                signals = self.market_detector.calculate_signals(sample['data'])
                
                # Calculate signal strengths
                long_strength = (
                    signals.get('bos_bullish', torch.tensor([0]))[-1].item() * 0.35 +
                    signals.get('choch_bullish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('order_block_bullish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('fvg_bullish', torch.tensor([0]))[-1].item() * 0.15
                )
                
                short_strength = (
                    signals.get('bos_bearish', torch.tensor([0]))[-1].item() * 0.35 +
                    signals.get('choch_bearish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('order_block_bearish', torch.tensor([0]))[-1].item() * 0.25 +
                    signals.get('fvg_bearish', torch.tensor([0]))[-1].item() * 0.15
                )
                
                signal_strengths.extend([long_strength, short_strength])
                
                # Analyze successful trades
                if sample['price_change'] > 0.05:  # Successful long
                    entry_thresholds.append(long_strength)
                elif sample['price_change'] < -0.05:  # Successful short
                    entry_thresholds.append(short_strength)
                
            except Exception as e:
                continue
        
        # Calculate optimal thresholds
        if entry_thresholds:
            optimal_entry = np.percentile(entry_thresholds, 75)  # 75th percentile
            optimal_exit = np.percentile(signal_strengths, 25)   # 25th percentile
        else:
            optimal_entry = 0.15
            optimal_exit = 0.05
        
        # Create optimized parameters
        params = TradingParams(
            entry_threshold=optimal_entry,
            exit_threshold=optimal_exit,
            bos_weight=0.35,
            choch_weight=0.25,
            order_block_weight=0.25,
            fvg_weight=0.15,
            max_leverage=25.0,
            position_size_pct=0.1,
            stop_loss_pct=0.05,
            take_profit_pct=0.15
        )
        
        logger.info(f"🎯 Real data optimized parameters:")
        logger.info(f"   Entry Threshold: {optimal_entry:.3f}")
        logger.info(f"   Exit Threshold: {optimal_exit:.3f}")
        
        return params
    
    def run_real_data_trading(self, cycles: int = 5):
        """Run trading using real data trained models."""
        logger.info("🚀 Starting Real Data ML Trading...")
        
        # Step 1: Load real data
        if not self.load_real_data():
            logger.error("❌ Failed to load real data!")
            return
        
        # Step 2: Train on real data
        if not self.train_on_real_data(epochs=50):
            logger.error("❌ Failed to train on real data!")
            return
        
        # Step 3: Optimize parameters
        optimized_params = self.optimize_parameters_real_data()
        
        # Step 4: Run trading
        logger.info("📊 Running real data optimized trading...")
        
        results = []
        for cycle in range(cycles):
            logger.info(f"📊 Cycle {cycle + 1}/{cycles}")
            
            cycle_result = self._run_real_data_cycle(optimized_params)
            results.append(cycle_result)
            
            # Print summary
            summary = cycle_result['portfolio_summary']
            logger.info(
                f"💰 Balance: ${summary['current_balance']:.2f} | "
                f"Equity: ${summary['total_equity']:.2f} | "
                f"PnL: ${summary['daily_pnl']:.2f} | "
                f"Positions: {summary['active_positions']}"
            )
        
        # Final performance report
        self._print_real_data_performance_report(results, optimized_params)
        
        return results, optimized_params
    
    def _run_real_data_cycle(self, params: TradingParams) -> Dict:
        """Run one trading cycle with real data optimized parameters."""
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': [],
            'trades': [],
            'portfolio_summary': {}
        }
        
        for symbol in self.symbols:
            try:
                # Use real data sample for this symbol
                symbol_data = [s for s in self.training_data if s['symbol'] == symbol]
                
                if symbol_data:
                    # Use most recent data sample
                    sample = random.choice(symbol_data)
                    df = sample['data']
                    
                    # Get market structure signals
                    signals = self.market_detector.calculate_signals(df)
                    
                    # Use ML predictor
                    features = self.extract_advanced_features(df)
                    with torch.no_grad():
                        prediction = self.signal_predictor(features.unsqueeze(0).to(self.device))
                        action_probs = prediction.cpu().numpy()[0]
                        action = np.argmax(action_probs)
                        confidence = np.max(action_probs)
                    
                    # Calculate signal strength
                    long_strength = (
                        signals.get('bos_bullish', torch.tensor([0]))[-1].item() * params.bos_weight +
                        signals.get('choch_bullish', torch.tensor([0]))[-1].item() * params.choch_weight +
                        signals.get('order_block_bullish', torch.tensor([0]))[-1].item() * params.order_block_weight +
                        signals.get('fvg_bullish', torch.tensor([0]))[-1].item() * params.fvg_weight
                    )
                    
                    short_strength = (
                        signals.get('bos_bearish', torch.tensor([0]))[-1].item() * params.bos_weight +
                        signals.get('choch_bearish', torch.tensor([0]))[-1].item() * params.choch_weight +
                        signals.get('order_block_bearish', torch.tensor([0]))[-1].item() * params.order_block_weight +
                        signals.get('fvg_bearish', torch.tensor([0]))[-1].item() * params.fvg_weight
                    )
                    
                    # Determine action
                    final_action = 'hold'
                    signal_strength = 0.0
                    
                    if action == 1 and long_strength > params.entry_threshold:
                        final_action = 'buy'
                        signal_strength = long_strength
                    elif action == 2 and short_strength > params.entry_threshold:
                        final_action = 'sell'
                        signal_strength = short_strength
                    elif long_strength < params.exit_threshold and short_strength < params.exit_threshold:
                        final_action = 'exit'
                    
                    # Create signal
                    signal = {
                        'symbol': symbol,
                        'action': final_action,
                        'signal_strength': signal_strength,
                        'ml_confidence': confidence,
                        'price': df['close'].iloc[-1],
                        'timestamp': df['timestamp'].iloc[-1],
                        'exchange': sample['exchange']
                    }
                    
                    cycle_results['signals'].append(signal)
                    
                    # Execute trade
                    if final_action != 'hold':
                        trade_result = self._execute_real_data_trade(signal, params)
                        cycle_results['trades'].append(trade_result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update portfolio
        cycle_results['portfolio_summary'] = self.leverage_manager.get_portfolio_summary()
        
        return cycle_results
    
    def _execute_real_data_trade(self, signal: Dict, params: TradingParams) -> Dict:
        """Execute trade with real data optimized parameters."""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        signal_strength = signal['signal_strength']
        ml_confidence = signal['ml_confidence']
        exchange = signal['exchange']
        
        # Calculate position size based on ML confidence and signal strength
        base_size = self.leverage_manager.config.starting_balance * params.position_size_pct
        adjusted_size = base_size * signal_strength * ml_confidence
        
        if action == 'buy':
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='long',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🟢 REAL DATA LONG {symbol} ({exchange}) @ ${price:.6f} - "
                          f"Strength: {signal_strength:.2f}, Confidence: {ml_confidence:.2f}")
                return {'action': 'buy', 'success': True, 'position': result['position']}
        
        elif action == 'sell':
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='short',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🔴 REAL DATA SHORT {symbol} ({exchange}) @ ${price:.6f} - "
                          f"Strength: {signal_strength:.2f}, Confidence: {ml_confidence:.2f}")
                return {'action': 'sell', 'success': True, 'position': result['position']}
        
        return {'action': action, 'success': False, 'reason': 'No action taken'}
    
    def save_model_state(self, results: List[Dict], params: TradingParams, performance_score: float):
        """Save model state and parameters when we get good results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        results_dir = "results/ml_optimized_trading"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model weights
        model_path = f"{results_dir}/model_{timestamp}.pth"
        torch.save({
            'model_state_dict': self.signal_predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'params': params,
            'performance_score': performance_score,
            'timestamp': timestamp
        }, model_path)
        
        # Save performance report
        report_path = f"{results_dir}/performance_report_{timestamp}.md"
        self._save_performance_report(results, params, performance_score, report_path)
        
        # Save optimized parameters
        params_path = f"{results_dir}/optimized_params_{timestamp}.yaml"
        self._save_parameters(params, params_path)
        
        logger.info(f"💾 SAVED MODEL STATE!")
        logger.info(f"📁 Model: {model_path}")
        logger.info(f"📁 Report: {report_path}")
        logger.info(f"📁 Params: {params_path}")
        logger.info(f"🎯 Performance Score: {performance_score:.2f}")
        
        return model_path, report_path, params_path
    
    def _save_performance_report(self, results: List[Dict], params: TradingParams, 
                                performance_score: float, file_path: str):
        """Save detailed performance report."""
        final_summary = results[-1]['portfolio_summary']
        performance_metrics = self.leverage_manager.get_performance_metrics()
        
        total_return = ((final_summary['current_balance'] / self.leverage_manager.config.starting_balance) - 1) * 100
        
        with open(file_path, 'w') as f:
            f.write("# 🚀 ML-Optimized Meme Coin Trading Performance Report\n\n")
            f.write(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Performance Score:** {performance_score:.2f}\n\n")
            
            f.write("## 💰 Account Performance\n\n")
            f.write(f"- **Starting Balance:** ${self.leverage_manager.config.starting_balance:.2f}\n")
            f.write(f"- **Final Balance:** ${final_summary['current_balance']:.2f}\n")
            f.write(f"- **Total Return:** {total_return:.2f}%\n")
            f.write(f"- **Daily PnL:** ${final_summary['daily_pnl']:.2f}\n")
            f.write(f"- **Max Drawdown:** {final_summary['current_drawdown'] * 100:.2f}%\n\n")
            
            if performance_metrics:
                f.write("## 📊 Trading Metrics\n\n")
                f.write(f"- **Total Trades:** {performance_metrics['total_trades']}\n")
                f.write(f"- **Winning Trades:** {performance_metrics['winning_trades']}\n")
                f.write(f"- **Losing Trades:** {performance_metrics['losing_trades']}\n")
                f.write(f"- **Win Rate:** {performance_metrics['win_rate'] * 100:.1f}%\n")
                f.write(f"- **Total PnL:** ${performance_metrics['total_pnl']:.2f}\n")
                f.write(f"- **Profit Factor:** {performance_metrics['profit_factor']:.2f}\n\n")
            
            f.write("## 🎯 Optimized Parameters\n\n")
            f.write(f"- **Entry Threshold:** {params.entry_threshold:.3f}\n")
            f.write(f"- **Exit Threshold:** {params.exit_threshold:.3f}\n")
            f.write(f"- **BOS Weight:** {params.bos_weight:.3f}\n")
            f.write(f"- **CHoCH Weight:** {params.choch_weight:.3f}\n")
            f.write(f"- **Order Block Weight:** {params.order_block_weight:.3f}\n")
            f.write(f"- **FVG Weight:** {params.fvg_weight:.3f}\n")
            f.write(f"- **Max Leverage:** {params.max_leverage:.1f}x\n")
            f.write(f"- **Position Size:** {params.position_size_pct:.1%}\n")
            f.write(f"- **Stop Loss:** {params.stop_loss_pct:.1%}\n")
            f.write(f"- **Take Profit:** {params.take_profit_pct:.1%}\n\n")
            
            f.write("## 📊 Data Summary\n\n")
            f.write(f"- **Training Samples:** {len(self.training_data)}\n")
            f.write(f"- **Validation Samples:** {len(self.validation_data)}\n")
            f.write(f"- **Exchanges Used:** {len(set(s['exchange'] for s in self.training_data))}\n")
            f.write(f"- **Symbols Traded:** {len(set(s['symbol'] for s in self.training_data))}\n\n")
    
    def _save_parameters(self, params: TradingParams, file_path: str):
        """Save optimized parameters to YAML file."""
        import yaml
        
        params_dict = {
            'entry_threshold': params.entry_threshold,
            'exit_threshold': params.exit_threshold,
            'bos_weight': params.bos_weight,
            'choch_weight': params.choch_weight,
            'order_block_weight': params.order_block_weight,
            'fvg_weight': params.fvg_weight,
            'max_leverage': params.max_leverage,
            'position_size_pct': params.position_size_pct,
            'stop_loss_pct': params.stop_loss_pct,
            'take_profit_pct': params.take_profit_pct
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(params_dict, f, default_flow_style=False)
    
    def calculate_performance_score(self, results: List[Dict]) -> float:
        """Calculate performance score focusing on returns."""
        final_summary = results[-1]['portfolio_summary']
        
        # Calculate total return
        total_return = (final_summary['current_balance'] / self.leverage_manager.config.starting_balance) - 1
        
        # Calculate return per trade
        performance_metrics = self.leverage_manager.get_performance_metrics()
        if performance_metrics and performance_metrics['total_trades'] > 0:
            return_per_trade = total_return / performance_metrics['total_trades']
        else:
            return_per_trade = 0
        
        # Calculate drawdown penalty
        drawdown_penalty = final_summary['current_drawdown'] * 2
        
        # Performance score = returns - drawdown penalty + return per trade bonus
        performance_score = (total_return * 100) - drawdown_penalty + (return_per_trade * 1000)
        
        return performance_score
    
    def _print_real_data_performance_report(self, results: List[Dict], params: TradingParams):
        """Print real data performance report."""
        logger.info("📊 REAL DATA ML PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        final_summary = results[-1]['portfolio_summary']
        performance_metrics = self.leverage_manager.get_performance_metrics()
        
        # Calculate performance score
        performance_score = self.calculate_performance_score(results)
        
        # Account performance
        logger.info(f"💰 Starting Balance: ${self.leverage_manager.config.starting_balance:.2f}")
        logger.info(f"💰 Final Balance: ${final_summary['current_balance']:.2f}")
        logger.info(f"📈 Total Return: {((final_summary['current_balance'] / self.leverage_manager.config.starting_balance) - 1) * 100:.2f}%")
        logger.info(f"📊 Daily PnL: ${final_summary['daily_pnl']:.2f}")
        logger.info(f"📉 Max Drawdown: {final_summary['current_drawdown'] * 100:.2f}%")
        logger.info(f"🎯 Performance Score: {performance_score:.2f}")
        
        # ML-specific metrics
        if performance_metrics:
            logger.info(f"🎯 Total Trades: {performance_metrics['total_trades']}")
            logger.info(f"✅ Winning Trades: {performance_metrics['winning_trades']}")
            logger.info(f"❌ Losing Trades: {performance_metrics['losing_trades']}")
            logger.info(f"📊 Win Rate: {performance_metrics['win_rate'] * 100:.1f}%")
            logger.info(f"💰 Total PnL: ${performance_metrics['total_pnl']:.2f}")
            logger.info(f"📈 Profit Factor: {performance_metrics['profit_factor']:.2f}")
        
        # Risk metrics
        logger.info(f"🔒 Total Exposure: ${final_summary['total_exposure']:.2f}")
        logger.info(f"⚡ Active Positions: {final_summary['active_positions']}")
        logger.info(f"📊 Trades Today: {final_summary['trades_today']}")
        logger.info(f"🎯 Win Rate Today: {final_summary['win_rate_today'] * 100:.1f}%")
        
        # Real data info
        logger.info(f"📊 Training Samples: {len(self.training_data)}")
        logger.info(f"📊 Validation Samples: {len(self.validation_data)}")
        logger.info(f"🎯 Optimized Entry Threshold: {params.entry_threshold:.3f}")
        logger.info(f"🎯 Optimized Exit Threshold: {params.exit_threshold:.3f}")
        
        logger.info("=" * 60)
        
        # Save state if performance is good
        if performance_score > 5.0:  # Save if we get >5% return
            self.save_model_state(results, params, performance_score)
        elif performance_score > 2.0:  # Save if we get >2% return
            logger.info("🎯 Good performance! Saving model state...")
            self.save_model_state(results, params, performance_score)
        else:
            logger.info("📊 Performance below threshold, not saving state")
        
        return performance_score

def main():
    """Main function to run the real data ML trader."""
    print("🧠 Real Data ML-Optimized Meme Coin HFT System")
    print("=" * 60)
    
    # Create real data ML trader
    trader = RealDataMLTrader()
    
    # Run real data ML trading
    print("🚀 Starting real data ML trading...")
    print("📊 Using real CSV data from multiple exchanges")
    print("🧠 Advanced ML training on actual market conditions")
    print("🎯 Auto-optimized parameters from real data patterns")
    print("=" * 60)
    
    # Run real data ML optimization and trading
    results, optimized_params = trader.run_real_data_trading(cycles=3)
    
    print("✅ Real data ML trading complete!")

if __name__ == "__main__":
    main() 