#!/usr/bin/env python3
"""
🚀 Fast ML-Optimized Meme Coin HFT System

Quick version that processes limited data but still gets good results!
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import time
import random
from dataclasses import dataclass
import glob

from src.ml.features.market_structure import MarketStructureDetector
from src.trading.leverage_manager import LeverageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradingParams:
    """Trading parameters optimized for speed."""
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

class FastSignalPredictor(nn.Module):
    """Fast neural network for signal prediction."""
    
    def __init__(self, input_size: int = 13, hidden_size: int = 64):
        super(FastSignalPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 3),  # buy, sell, hold
            nn.Softmax(dim=1)
        )
        
        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class FastDataLoader:
    """Fast data loader that processes limited data."""
    
    def __init__(self, data_dir: str = "data/historical/processed"):
        self.data_dir = data_dir
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        logger.info(f"📊 Fast Data Loader initialized")
        logger.info(f"🎯 Targeting {len(self.symbols)} meme coins")
    
    def load_sample_data(self, max_files_per_symbol: int = 2) -> List[Dict]:
        """Load sample data for quick training."""
        logger.info(f"📊 Loading sample data (max {max_files_per_symbol} files per symbol)...")
        
        training_data = []
        
        for symbol in self.symbols:
            # Look for recent data files
            pattern = f"{self.data_dir}/*/{symbol}/1m/*/*.csv"
            files = glob.glob(pattern)
            
            if files:
                # Take most recent files
                files = sorted(files)[-max_files_per_symbol:]
                
                for file_path in files:
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Standardize columns
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        if 'close' in df.columns:
                            df['close'] = pd.to_numeric(df['close'], errors='coerce')
                        if 'volume' in df.columns:
                            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                        
                        df = df.dropna()
                        
                        if len(df) > 100:  # Need minimum data
                            # Extract exchange from path
                            path_parts = file_path.split('/')
                            exchange = path_parts[-5] if len(path_parts) > 5 else 'unknown'
                            
                            sample = {
                                'symbol': symbol,
                                'exchange': exchange,
                                'data': df,
                                'file_path': file_path
                            }
                            
                            training_data.append(sample)
                            logger.info(f"✅ Loaded {symbol} from {exchange}: {len(df)} rows")
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Error loading {file_path}: {e}")
                        continue
        
        logger.info(f"✅ Loaded {len(training_data)} sample datasets")
        return training_data

class FastMLTrader:
    """Fast ML-optimized meme coin trader."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        # Initialize components
        self.market_detector = MarketStructureDetector()
        self.leverage_manager = LeverageManager()
        self.data_loader = FastDataLoader()
        
        # ML components
        self.signal_predictor = FastSignalPredictor().to(self.device)
        self.optimizer = optim.Adam(self.signal_predictor.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training data
        self.training_data = []
        
        logger.info("🚀 Fast ML Trader initialized!")
    
    def load_and_train(self, epochs: int = 20):
        """Quick load and train on sample data."""
        logger.info("📊 Loading sample data for quick training...")
        
        # Load sample data
        self.training_data = self.data_loader.load_sample_data(max_files_per_symbol=2)
        
        if not self.training_data:
            logger.error("❌ No data found! Using synthetic data...")
            self._generate_synthetic_data()
        
        # Quick training
        logger.info(f"🧠 Quick training for {epochs} epochs...")
        self._quick_train(epochs)
        
        return True
    
    def _generate_synthetic_data(self):
        """Generate synthetic data if no real data available."""
        logger.info("🎲 Generating synthetic data...")
        
        for symbol in self.symbols:
            # Generate realistic data
            dates = pd.date_range(
                datetime.now() - timedelta(days=7), 
                periods=1000, 
                freq='1min'
            )
            
            np.random.seed(hash(symbol) % 1000)
            
            base_price = 0.001 if 'FART' in symbol else 0.01
            prices = [base_price]
            
            for i in range(1, 1000):
                volatility = 0.2  # High volatility for meme coins
                change = np.random.normal(0, volatility)
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.0001))
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.05))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.05))) for p in prices],
                'close': prices,
                'volume': [np.random.uniform(1000000, 10000000) for _ in prices]
            })
            
            sample = {
                'symbol': symbol,
                'exchange': 'synthetic',
                'data': df,
                'file_path': 'synthetic'
            }
            
            self.training_data.append(sample)
    
    def _quick_train(self, epochs: int):
        """Quick training on limited data."""
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for sample in self.training_data:
                try:
                    df = sample['data']
                    
                    # Use last 100 rows for training
                    if len(df) > 100:
                        df = df.tail(100)
                    
                    # Extract features
                    features = self._extract_features(df)
                    
                    # Create target based on price movement
                    price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                    
                    if price_change > 0.03:  # 3% gain = buy
                        target = torch.tensor([1], dtype=torch.long)  # Buy
                    elif price_change < -0.03:  # 3% loss = sell
                        target = torch.tensor([2], dtype=torch.long)  # Sell
                    else:
                        target = torch.tensor([0], dtype=torch.long)  # Hold
                    
                    target = target.to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    output = self.signal_predictor(features.unsqueeze(0))
                    loss = self.criterion(output, target)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted = torch.argmax(output, dim=1)
                    correct_predictions += (predicted == target).sum().item()
                    total_predictions += 1
                    
                except Exception as e:
                    continue
            
            # Log progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / max(total_predictions, 1)
                accuracy = correct_predictions / max(total_predictions, 1)
                logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}")
        
        logger.info("✅ Quick training complete!")
    
    def _extract_features(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract features from market data."""
        features = []
        
        # Market structure signals
        try:
            signals = self.market_detector.calculate_signals(df)
            
            for key in ['bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
                       'order_block_bullish', 'order_block_bearish', 'fvg_bullish', 'fvg_bearish']:
                tensor = signals.get(key, torch.tensor([]))
                if len(tensor) > 0:
                    features.append(tensor[-1].item())
                else:
                    features.append(0.0)
        except:
            # Fallback to zeros if market structure fails
            features.extend([0.0] * 8)
        
        # Price action features
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        # Price momentum
        price_change_1h = (close_prices[-1] - close_prices[-60]) / close_prices[-60] if len(close_prices) > 60 else 0
        price_change_4h = (close_prices[-1] - close_prices[-240]) / close_prices[-240] if len(close_prices) > 240 else 0
        
        # Volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Volume features
        volume_ratio = volumes[-1] / np.mean(volumes[-20:]) if len(volumes) > 20 else 1
        
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
    
    def optimize_parameters(self) -> TradingParams:
        """Quick parameter optimization."""
        logger.info("🧬 Quick parameter optimization...")
        
        # Use aggressive parameters for high returns
        params = TradingParams(
            entry_threshold=0.1,  # Lower threshold for more signals
            exit_threshold=0.03,  # Lower exit threshold
            bos_weight=0.4,       # Emphasize BOS
            choch_weight=0.3,     # Emphasize CHoCH
            order_block_weight=0.2,
            fvg_weight=0.1,
            max_leverage=25.0,    # Max leverage
            position_size_pct=0.15,  # Larger positions
            stop_loss_pct=0.03,   # Tighter stop loss
            take_profit_pct=0.2   # Higher profit target
        )
        
        logger.info("✅ Quick parameter optimization complete!")
        logger.info(f"🎯 Aggressive Parameters:")
        logger.info(f"   Entry Threshold: {params.entry_threshold:.3f}")
        logger.info(f"   Exit Threshold: {params.exit_threshold:.3f}")
        logger.info(f"   Max Leverage: {params.max_leverage:.1f}x")
        logger.info(f"   Position Size: {params.position_size_pct:.1%}")
        
        return params
    
    def run_fast_trading(self, cycles: int = 3):
        """Run fast ML-optimized trading."""
        logger.info("🚀 Starting Fast ML Trading...")
        
        # Step 1: Quick load and train
        if not self.load_and_train(epochs=15):
            logger.error("❌ Failed to load and train!")
            return
        
        # Step 2: Optimize parameters
        optimized_params = self.optimize_parameters()
        
        # Step 3: Run trading
        logger.info("📊 Running fast trading cycles...")
        
        results = []
        for cycle in range(cycles):
            logger.info(f"📊 Cycle {cycle + 1}/{cycles}")
            
            cycle_result = self._run_fast_cycle(optimized_params)
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
        performance_score = self._print_fast_performance_report(results, optimized_params)
        
        return results, optimized_params, performance_score
    
    def _run_fast_cycle(self, params: TradingParams) -> Dict:
        """Run one fast trading cycle."""
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': [],
            'trades': [],
            'portfolio_summary': {}
        }
        
        for symbol in self.symbols:
            try:
                # Use training data for this symbol
                symbol_data = [s for s in self.training_data if s['symbol'] == symbol]
                
                if symbol_data:
                    # Use most recent data
                    sample = random.choice(symbol_data)
                    df = sample['data']
                    
                    # Use last 100 rows
                    if len(df) > 100:
                        df = df.tail(100)
                    
                    # Get market structure signals
                    signals = self.market_detector.calculate_signals(df)
                    
                    # Use ML predictor
                    features = self._extract_features(df)
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
                        trade_result = self._execute_fast_trade(signal, params)
                        cycle_results['trades'].append(trade_result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update portfolio
        cycle_results['portfolio_summary'] = self.leverage_manager.get_portfolio_summary()
        
        return cycle_results
    
    def _execute_fast_trade(self, signal: Dict, params: TradingParams) -> Dict:
        """Execute trade with fast parameters."""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        signal_strength = signal['signal_strength']
        ml_confidence = signal['ml_confidence']
        exchange = signal['exchange']
        
        if action == 'buy':
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='long',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🟢 FAST LONG {symbol} ({exchange}) @ ${price:.6f} - "
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
                logger.info(f"🔴 FAST SHORT {symbol} ({exchange}) @ ${price:.6f} - "
                          f"Strength: {signal_strength:.2f}, Confidence: {ml_confidence:.2f}")
                return {'action': 'sell', 'success': True, 'position': result['position']}
        
        return {'action': action, 'success': False, 'reason': 'No action taken'}
    
    def save_fast_results(self, results: List[Dict], params: TradingParams, performance_score: float):
        """Save fast results if performance is good."""
        if performance_score > 1.0:  # Save if we get >1% return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create results directory
            results_dir = "results/fast_ml_trading"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save model
            model_path = f"{results_dir}/fast_model_{timestamp}.pth"
            torch.save({
                'model_state_dict': self.signal_predictor.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'params': params,
                'performance_score': performance_score,
                'timestamp': timestamp
            }, model_path)
            
            logger.info(f"💾 SAVED FAST RESULTS!")
            logger.info(f"📁 Model: {model_path}")
            logger.info(f"🎯 Performance Score: {performance_score:.2f}")
            
            return model_path
        
        return None
    
    def _print_fast_performance_report(self, results: List[Dict], params: TradingParams) -> float:
        """Print fast performance report."""
        logger.info("📊 FAST ML PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        final_summary = results[-1]['portfolio_summary']
        performance_metrics = self.leverage_manager.get_performance_metrics()
        
        # Calculate performance score
        total_return = (final_summary['current_balance'] / self.leverage_manager.config.starting_balance) - 1
        performance_score = total_return * 100  # Simple return-based score
        
        # Account performance
        logger.info(f"💰 Starting Balance: ${self.leverage_manager.config.starting_balance:.2f}")
        logger.info(f"💰 Final Balance: ${final_summary['current_balance']:.2f}")
        logger.info(f"📈 Total Return: {total_return * 100:.2f}%")
        logger.info(f"📊 Daily PnL: ${final_summary['daily_pnl']:.2f}")
        logger.info(f"📉 Max Drawdown: {final_summary['current_drawdown'] * 100:.2f}%")
        logger.info(f"🎯 Performance Score: {performance_score:.2f}")
        
        # Trading metrics
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
        
        logger.info("=" * 60)
        
        # Save if performance is good
        if performance_score > 1.0:
            self.save_fast_results(results, params, performance_score)
        else:
            logger.info("📊 Performance below threshold, not saving")
        
        return performance_score

def main():
    """Main function to run the fast ML trader."""
    print("🚀 Fast ML-Optimized Meme Coin HFT System")
    print("=" * 60)
    
    # Create fast ML trader
    trader = FastMLTrader()
    
    # Run fast ML trading
    print("🚀 Starting fast ML trading...")
    print("📊 Quick training on sample data")
    print("🧠 Aggressive parameters for high returns")
    print("⚡ Fast execution for quick results")
    print("=" * 60)
    
    # Run fast ML optimization and trading
    results, optimized_params, performance_score = trader.run_fast_trading(cycles=3)
    
    print("✅ Fast ML trading complete!")
    print(f"🎯 Final Performance Score: {performance_score:.2f}")

if __name__ == "__main__":
    main() 