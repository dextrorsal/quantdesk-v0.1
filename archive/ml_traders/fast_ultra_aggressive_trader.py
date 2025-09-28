#!/usr/bin/env python3
"""
FAST ULTRA-AGGRESSIVE ML TRADER - QUICK TEST VERSION
Optimized for speed with limited symbols and data for quick results
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List
import os
import json
from dataclasses import dataclass
from pathlib import Path

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ml.models.strategy.logistic_regression_torch import LogisticRegression, LogisticConfig
from src.trading.leverage_manager import LeverageManager, LeverageConfig
from src.ml.features.market_structure import MarketStructureDetector, MarketStructureConfig
from src.data.csv_storage import CSVStorage, StorageConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FastUltraConfig:
    """Fast ultra-aggressive configuration for quick testing"""
    # Account settings
    starting_balance: float = 1000.0
    max_leverage: float = 50.0  # High leverage but not extreme
    position_size_pct: float = 0.5  # 50% position size
    
    # Risk management (aggressive)
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.25  # 25% take profit
    max_drawdown: float = 0.30  # 30% max drawdown
    
    # Trading parameters (ultra-aggressive)
    force_trade_prob: float = 0.8  # 80% force trade probability
    position_hold_time: int = 6  # 6 cycles - shorter holds
    min_signal_threshold: float = 0.001  # Very low threshold
    
    # ML parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 20  # Fewer epochs for speed
    
    # Market structure
    bos_threshold: float = 0.01  # 1% BOS threshold
    choch_threshold: float = 0.02  # 2% CHoCH threshold
    
    # Fees
    maker_fee: float = 0.0001
    taker_fee: float = 0.0002

class FastNeuralNetwork(nn.Module):
    """Fast neural network for quick training"""
    
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

class FastUltraAggressiveTrader:
    """Fast ultra-aggressive ML trader for quick testing"""
    
    def __init__(self, config: FastUltraConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        leverage_config = LeverageConfig(
            starting_balance=config.starting_balance,
            max_leverage=config.max_leverage,
            max_drawdown=config.max_drawdown,
            base_position_size=config.position_size_pct,
            daily_loss_limit=config.max_drawdown
        )
        self.leverage_manager = LeverageManager(leverage_config)
        
        market_config = MarketStructureConfig(
            min_structure_strength=0.6,
            volume_threshold=1.5
        )
        self.market_structure = MarketStructureDetector(market_config)
        
        storage_config = StorageConfig(data_path=Path("data/historical/processed"))
        self.csv_storage = CSVStorage(storage_config)
        
        # ML models
        logistic_config = LogisticConfig(
            learning_rate=config.learning_rate,
            device=str(self.device)
        )
        self.logistic_model = LogisticRegression(logistic_config)
        
        self.neural_network = FastNeuralNetwork(
            input_size=22,  # Match the actual number of features
            hidden_size=128
        ).to(self.device)
        
        # Position tracking
        self.positions = {}
        
        logger.info("Fast Ultra-Aggressive ML Trader initialized for QUICK TESTING!")

    def prepare_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Prepare features for ML models - simplified version"""
        features = []
        
        # Basic price features
        features.extend([
            df['close'].pct_change().fillna(0),
            df['volume'].pct_change().fillna(0),
            (df['high'] - df['low']) / df['close'],
            (df['close'] - df['open']) / df['open']
        ])
        
        # Simple moving averages
        for period in [5, 10, 20]:
            features.append(df['close'].rolling(period).mean().pct_change().fillna(0))
            features.append(df['volume'].rolling(period).mean().pct_change().fillna(0))
        
        # RSI-like features
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        features.append(rsi.fillna(50) / 100)
        
        # Volatility features
        for period in [5, 10, 20]:
            volatility = df['close'].rolling(period).std() / df['close'].rolling(period).mean()
            features.append(volatility.fillna(0))
        
        # Market structure features (simplified)
        try:
            bos_signals = self.market_structure.detect_bos(df)
            choch_signals = self.market_structure.detect_choch(df)
            order_blocks = self.market_structure.detect_order_blocks(df)
            fvgs = self.market_structure.detect_fair_value_gaps(df)
            
            # Convert tensors to numpy arrays
            bos_bullish = bos_signals['bos_bullish'].cpu().numpy()
            bos_bearish = bos_signals['bos_bearish'].cpu().numpy()
            choch_bullish = choch_signals['choch_bullish'].cpu().numpy()
            choch_bearish = choch_signals['choch_bearish'].cpu().numpy()
            ob_bullish = order_blocks['order_block_bullish'].cpu().numpy()
            ob_bearish = order_blocks['order_block_bearish'].cpu().numpy()
            fvg_bullish = fvgs['fvg_bullish'].cpu().numpy()
            fvg_bearish = fvgs['fvg_bearish'].cpu().numpy()
            
            features.extend([
                pd.Series(bos_bullish, index=df.index),
                pd.Series(bos_bearish, index=df.index),
                pd.Series(choch_bullish, index=df.index),
                pd.Series(choch_bearish, index=df.index),
                pd.Series(ob_bullish, index=df.index),
                pd.Series(ob_bearish, index=df.index),
                pd.Series(fvg_bullish, index=df.index),
                pd.Series(fvg_bearish, index=df.index)
            ])
        except Exception as e:
            logger.warning(f"Market structure error: {e}")
            # Add zeros for market structure features
            for _ in range(8):
                features.append(pd.Series(0, index=df.index))
        
        # Combine all features
        feature_df = pd.concat(features, axis=1)
        feature_df = feature_df.fillna(0)
        
        return torch.tensor(feature_df.values, dtype=torch.float32, device=self.device)

    def train_models(self, df: pd.DataFrame) -> Dict:
        """Train ML models - simplified version"""
        if len(df) < 100:
            return {'error': 'Insufficient data'}
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Create simple labels (1 if price goes up next period, 0 otherwise)
        future_returns = df['close'].pct_change().shift(-1).fillna(0)
        y = (future_returns > 0).astype(int)
        y = torch.tensor(y.values, dtype=torch.float32, device=self.device)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_train) < 50:
            return {'error': 'Insufficient training data'}
        
        # Train neural network
        optimizer = optim.Adam(self.neural_network.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()
        
        self.neural_network.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            outputs = self.neural_network(X_train).squeeze()
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate
        self.neural_network.eval()
        with torch.no_grad():
            nn_pred = (self.neural_network(X_test).squeeze() > 0.5).float()
            nn_accuracy = (nn_pred == y_test).float().mean().item()
        
        logger.info(f"Neural Network Accuracy: {nn_accuracy:.4f}")
        
        return {
            'nn_accuracy': nn_accuracy,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def generate_signals(self, df: pd.DataFrame) -> Dict:
        """Generate trading signals - simplified version"""
        if len(df) < 50:
            return {
                'buy_signals': np.zeros(len(df)),
                'sell_signals': np.zeros(len(df)),
                'confidence': np.zeros(len(df))
            }
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Get neural network predictions
        self.neural_network.eval()
        with torch.no_grad():
            nn_pred = self.neural_network(X).squeeze().cpu().numpy()
        
        # Generate signals based on predictions
        buy_signals = (nn_pred > 0.6).astype(int)  # High confidence buy
        sell_signals = (nn_pred < 0.4).astype(int)  # Low confidence sell
        
        # Force some trades for testing
        if np.random.random() < self.config.force_trade_prob:
            if np.random.random() < 0.5:
                buy_signals[-1] = 1
                sell_signals[-1] = 0
            else:
                buy_signals[-1] = 0
                sell_signals[-1] = 1
        
        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'confidence': nn_pred
        }

    def execute_trades(self, symbol: str, current_price: float, signals: Dict) -> List[Dict]:
        """Execute trades - simplified version"""
        trades = []
        
        # Check for buy signals
        if signals['buy_signals'][-1] and symbol not in self.positions:
            trade_result = self.leverage_manager.open_position(
                symbol=symbol,
                side='long',
                price=current_price,
                signal_strength=signals['confidence'][-1]
            )
            
            if trade_result['success']:
                self.positions[symbol] = {
                    'side': 'long',
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'hold_cycles': 0
                }
                trades.append(trade_result)
                logger.info(f"🟢 FAST LONG: {symbol} @ ${current_price:.6f}")
        
        # Check for sell signals
        elif signals['sell_signals'][-1] and symbol not in self.positions:
            trade_result = self.leverage_manager.open_position(
                symbol=symbol,
                side='short',
                price=current_price,
                signal_strength=signals['confidence'][-1]
            )
            
            if trade_result['success']:
                self.positions[symbol] = {
                    'side': 'short',
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'hold_cycles': 0
                }
                trades.append(trade_result)
                logger.info(f"🔴 FAST SHORT: {symbol} @ ${current_price:.6f}")
        
        # Manage existing positions
        for symbol, position in list(self.positions.items()):
            position['hold_cycles'] += 1
            
            # Simple exit conditions
            if position['hold_cycles'] >= self.config.position_hold_time:
                trade_result = self.leverage_manager.close_position(symbol, current_price)
                if trade_result['success']:
                    trades.append(trade_result)
                    del self.positions[symbol]
                    logger.info(f"⏰ TIME EXIT: {symbol} @ ${current_price:.6f}")
        
        return trades

    async def run_backtest(self, symbols: List[str], days: int = 3) -> Dict:
        """Run fast backtest with limited data"""
        logger.info(f"🚀 STARTING FAST BACKTEST: {len(symbols)} symbols, {days} days")
        
        # Load limited data
        all_data = {}
        # Use July 16th data since that's what we have
        end_date = datetime(2025, 7, 16, 23, 59, 59)
        start_date = datetime(2025, 7, 16, 0, 0, 0)
        
        for symbol in symbols[:3]:  # Only process first 3 symbols
            try:
                # Try to load data from kraken 1m (we know it has data)
                data = await self.csv_storage.load_candles(
                    exchange='kraken',
                    pair=symbol,
                    interval='1m',
                    start_time=start_date,
                    end_time=end_date
                )
                
                if data is not None and len(data) > 50:
                    all_data[symbol] = data
                    logger.info(f"Loaded {symbol}: {len(data)} candles")
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
        
        if not all_data:
            return {'error': 'No data loaded'}
        
        # Train models on first symbol
        first_symbol = list(all_data.keys())[0]
        df = all_data[first_symbol]
        training_result = self.train_models(df)
        logger.info(f"Training result: {training_result}")
        
        # Run backtest
        total_trades = []
        total_pnl = 0.0
        
        for symbol, df in all_data.items():
            logger.info(f"Processing {symbol} with {len(df)} candles")
            
            # Generate signals
            signals = self.generate_signals(df)
            
            # Execute trades
            for i in range(len(df)):
                current_price = df.iloc[i]['close']
                current_signals = {
                    'buy_signals': signals['buy_signals'][:i+1],
                    'sell_signals': signals['sell_signals'][:i+1],
                    'confidence': signals['confidence'][:i+1]
                }
                
                trades = self.execute_trades(symbol, current_price, current_signals)
                total_trades.extend(trades)
        
        # Get final results
        portfolio_summary = self.leverage_manager.get_portfolio_summary()
        performance_metrics = self.leverage_manager.get_performance_metrics()
        
        # Calculate total return manually
        total_return = (portfolio_summary['current_balance'] - self.config.starting_balance) / self.config.starting_balance
        
        results = {
            'final_balance': portfolio_summary['current_balance'],
            'total_return': total_return,
            'total_trades': len(total_trades),
            'daily_pnl': performance_metrics.get('daily_pnl', 0.0),
            'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
            'win_rate': performance_metrics.get('win_rate', 0.0),
            'training_result': training_result
        }
        
        logger.info(f"🎯 FAST BACKTEST COMPLETE: ${results['final_balance']:.2f} | "
                   f"Return: {results['total_return']:.2%} | "
                   f"Trades: {results['total_trades']}")
        
        return results

async def main():
    """Main function for fast testing"""
    config = FastUltraConfig()
    
    # Use only 3 symbols for speed
    symbols = ['FARTCOIN', 'PEPE', 'WIF']
    
    trader = FastUltraAggressiveTrader(config)
    results = await trader.run_backtest(symbols, days=3)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/fast_ultra_aggressive_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    logger.info(f"Final Results: {results}")

if __name__ == "__main__":
    asyncio.run(main()) 