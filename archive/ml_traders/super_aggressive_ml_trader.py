#!/usr/bin/env python3
"""
💥 SUPER-AGGRESSIVE ML Meme Coin HFT System

FORCES TRADES and MAXIMIZES RETURNS - This is the MONEY MAKER!
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
import random
from dataclasses import dataclass

from src.ml.features.market_structure import MarketStructureDetector
from src.trading.leverage_manager import LeverageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SuperAggressiveParams:
    """Super-aggressive trading parameters that FORCE trades."""
    entry_threshold: float = 0.01  # Extremely low threshold
    exit_threshold: float = 0.005  # Extremely low exit
    bos_weight: float = 0.6        # Heavy BOS weight
    choch_weight: float = 0.4      # Heavy CHoCH weight
    order_block_weight: float = 0.0 # Ignore order blocks
    fvg_weight: float = 0.0        # Ignore FVGs
    max_leverage: float = 25.0     # Max leverage
    position_size_pct: float = 0.25 # Large positions
    stop_loss_pct: float = 0.01    # Very tight stop loss
    take_profit_pct: float = 0.3   # High profit target

class SuperAggressivePredictor(nn.Module):
    """Super-aggressive neural network that always predicts trades."""
    
    def __init__(self, input_size: int = 13, hidden_size: int = 16):
        super(SuperAggressivePredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),  # buy, sell, hold
            nn.Softmax(dim=1)
        )
        
        # Initialize weights to be aggressive
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SuperAggressiveTrader:
    """Super-aggressive ML trader that FORCES trades."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        # Initialize components
        self.market_detector = MarketStructureDetector()
        self.leverage_manager = LeverageManager()
        
        # ML components
        self.signal_predictor = SuperAggressivePredictor().to(self.device)
        self.optimizer = optim.Adam(self.signal_predictor.parameters(), lr=0.05)  # Very high learning rate
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("💥 Super-Aggressive ML Trader initialized!")
    
    def generate_super_volatile_data(self):
        """Generate extremely volatile synthetic data with massive moves."""
        logger.info("🎲 Generating super-volatile synthetic data...")
        
        training_data = []
        
        for symbol in self.symbols:
            # Generate extremely volatile data
            dates = pd.date_range(
                datetime.now() - timedelta(days=2), 
                periods=300, 
                freq='1min'
            )
            
            np.random.seed(hash(symbol) % 1000)
            
            base_price = 0.001 if 'FART' in symbol else 0.01
            prices = [base_price]
            
            for i in range(1, 300):
                # Super-high volatility for meme coins
                volatility = 0.6  # 60% volatility!
                change = np.random.normal(0, volatility)
                
                # Add massive spikes and drops
                if random.random() < 0.2:  # 20% chance of spike
                    change = random.choice([1.0, -1.0, 0.8, -0.8])  # 100% or 80% move
                
                # Add trend bias
                if i > 50 and random.random() < 0.3:
                    change += 0.2  # Bias towards up moves
                
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 0.0001))
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.2))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.2))) for p in prices],
                'close': prices,
                'volume': [np.random.uniform(10000000, 100000000) for _ in prices]
            })
            
            sample = {
                'symbol': symbol,
                'exchange': 'super_volatile',
                'data': df,
                'file_path': 'super_volatile'
            }
            
            training_data.append(sample)
        
        return training_data
    
    def super_quick_train(self, training_data: List[Dict], epochs: int = 5):
        """Super-quick training that forces aggressive predictions."""
        logger.info(f"🧠 Super-quick training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for sample in training_data:
                try:
                    df = sample['data']
                    
                    # Use last 30 rows for super-fast training
                    if len(df) > 30:
                        df = df.tail(30)
                    
                    # Extract features
                    features = self._extract_features(df)
                    
                    # Create super-aggressive targets
                    price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                    
                    if price_change > 0.01:  # 1% gain = buy (very low threshold)
                        target = torch.tensor([1], dtype=torch.long)  # Buy
                    elif price_change < -0.01:  # 1% loss = sell (very low threshold)
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
            
            # Log progress every epoch
            avg_loss = total_loss / max(total_predictions, 1)
            accuracy = correct_predictions / max(total_predictions, 1)
            logger.info(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}")
        
        logger.info("✅ Super-quick training complete!")
    
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
        
        # Price momentum (very short timeframes for aggression)
        price_change_15m = (close_prices[-1] - close_prices[-15]) / close_prices[-15] if len(close_prices) > 15 else 0
        price_change_30m = (close_prices[-1] - close_prices[-30]) / close_prices[-30] if len(close_prices) > 30 else 0
        
        # Volatility
        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # Volume features
        volume_ratio = volumes[-1] / np.mean(volumes[-5:]) if len(volumes) > 5 else 1
        
        # RSI-like momentum (very short period)
        gains = np.where(returns > 0, returns, 0)
        losses = np.where(returns < 0, -returns, 0)
        avg_gain = np.mean(gains[-5:]) if len(gains) > 5 else 0
        avg_loss = np.mean(losses[-5:]) if len(losses) > 5 else 0
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Add all features
        features.extend([
            price_change_15m,
            price_change_30m,
            volatility,
            volume_ratio,
            rsi / 100  # Normalize to 0-1
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def run_super_aggressive_trading(self, cycles: int = 8):
        """Run super-aggressive ML trading that FORCES trades."""
        logger.info("💥 Starting Super-Aggressive ML Trading...")
        
        # Step 1: Generate super-volatile data
        training_data = self.generate_super_volatile_data()
        
        # Step 2: Super-quick training
        self.super_quick_train(training_data, epochs=5)
        
        # Step 3: Super-aggressive parameters
        params = SuperAggressiveParams()
        
        logger.info("🎯 Super-Aggressive Parameters:")
        logger.info(f"   Entry Threshold: {params.entry_threshold:.3f}")
        logger.info(f"   Exit Threshold: {params.exit_threshold:.3f}")
        logger.info(f"   Max Leverage: {params.max_leverage:.1f}x")
        logger.info(f"   Position Size: {params.position_size_pct:.1%}")
        logger.info(f"   Stop Loss: {params.stop_loss_pct:.1%}")
        logger.info(f"   Take Profit: {params.take_profit_pct:.1%}")
        
        # Step 4: Run super-aggressive trading
        logger.info("📊 Running super-aggressive trading cycles...")
        
        results = []
        for cycle in range(cycles):
            logger.info(f"📊 Cycle {cycle + 1}/{cycles}")
            
            cycle_result = self._run_super_cycle(params, training_data)
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
        performance_score = self._print_super_performance_report(results, params)
        
        return results, params, performance_score
    
    def _run_super_cycle(self, params: SuperAggressiveParams, training_data: List[Dict]) -> Dict:
        """Run one super-aggressive trading cycle that FORCES trades."""
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': [],
            'trades': [],
            'portfolio_summary': {}
        }
        
        for symbol in self.symbols:
            try:
                # Use training data for this symbol
                symbol_data = [s for s in training_data if s['symbol'] == symbol]
                
                if symbol_data:
                    # Use most recent data
                    sample = random.choice(symbol_data)
                    df = sample['data']
                    
                    # Use last 30 rows for super-fast processing
                    if len(df) > 30:
                        df = df.tail(30)
                    
                    # Get market structure signals
                    signals = self.market_detector.calculate_signals(df)
                    
                    # Use ML predictor
                    features = self._extract_features(df)
                    with torch.no_grad():
                        prediction = self.signal_predictor(features.unsqueeze(0).to(self.device))
                        action_probs = prediction.cpu().numpy()[0]
                        action = np.argmax(action_probs)
                        confidence = np.max(action_probs)
                    
                    # Calculate signal strength with super-aggressive weights
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
                    
                    # SUPER-AGGRESSIVE action determination - FORCE TRADES
                    final_action = 'hold'
                    signal_strength = 0.0
                    
                    # Force trades based on ML prediction
                    if action == 1:  # ML predicts buy
                        final_action = 'buy'
                        signal_strength = max(long_strength, 0.1)  # Force minimum strength
                    elif action == 2:  # ML predicts sell
                        final_action = 'sell'
                        signal_strength = max(short_strength, 0.1)  # Force minimum strength
                    else:  # ML predicts hold, but we might still trade
                        if long_strength > params.entry_threshold:
                            final_action = 'buy'
                            signal_strength = long_strength
                        elif short_strength > params.entry_threshold:
                            final_action = 'sell'
                            signal_strength = short_strength
                    
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
                        trade_result = self._execute_super_trade(signal, params)
                        cycle_results['trades'].append(trade_result)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update portfolio
        cycle_results['portfolio_summary'] = self.leverage_manager.get_portfolio_summary()
        
        return cycle_results
    
    def _execute_super_trade(self, signal: Dict, params: SuperAggressiveParams) -> Dict:
        """Execute super-aggressive trade."""
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
                logger.info(f"💥 SUPER LONG {symbol} ({exchange}) @ ${price:.6f} - "
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
                logger.info(f"💥 SUPER SHORT {symbol} ({exchange}) @ ${price:.6f} - "
                          f"Strength: {signal_strength:.2f}, Confidence: {ml_confidence:.2f}")
                return {'action': 'sell', 'success': True, 'position': result['position']}
        
        return {'action': action, 'success': False, 'reason': 'No action taken'}
    
    def save_super_results(self, results: List[Dict], params: SuperAggressiveParams, performance_score: float):
        """Save super results if performance is good."""
        if performance_score > 0.1:  # Save if we get >0.1% return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create results directory
            results_dir = "results/super_aggressive_trading"
            os.makedirs(results_dir, exist_ok=True)
            
            # Save model
            model_path = f"{results_dir}/super_model_{timestamp}.pth"
            torch.save({
                'model_state_dict': self.signal_predictor.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'params': params,
                'performance_score': performance_score,
                'timestamp': timestamp
            }, model_path)
            
            logger.info(f"💾 SAVED SUPER RESULTS!")
            logger.info(f"📁 Model: {model_path}")
            logger.info(f"🎯 Performance Score: {performance_score:.2f}")
            
            return model_path
        
        return None
    
    def _print_super_performance_report(self, results: List[Dict], params: SuperAggressiveParams) -> float:
        """Print super performance report."""
        logger.info("📊 SUPER-AGGRESSIVE ML PERFORMANCE REPORT")
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
        if performance_score > 0.1:
            self.save_super_results(results, params, performance_score)
        else:
            logger.info("📊 Performance below threshold, not saving")
        
        return performance_score

def main():
    """Main function to run the super-aggressive ML trader."""
    print("💥 Super-Aggressive ML Meme Coin HFT System")
    print("=" * 60)
    
    # Create super-aggressive ML trader
    trader = SuperAggressiveTrader()
    
    # Run super-aggressive ML trading
    print("💥 Starting super-aggressive ML trading...")
    print("🎲 Super-volatile synthetic data")
    print("🧠 Super-quick ML training")
    print("⚡ FORCES TRADES")
    print("💰 MAXIMUM RETURNS")
    print("=" * 60)
    
    # Run super-aggressive ML optimization and trading
    results, optimized_params, performance_score = trader.run_super_aggressive_trading(cycles=8)
    
    print("✅ Super-aggressive ML trading complete!")
    print(f"🎯 Final Performance Score: {performance_score:.2f}")

if __name__ == "__main__":
    main() 