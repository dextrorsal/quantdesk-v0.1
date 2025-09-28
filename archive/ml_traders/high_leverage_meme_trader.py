#!/usr/bin/env python3
"""
🚀 High-Leverage Meme Coin HFT System

Aggressive high-frequency trading system for meme coins with 25x leverage.
Combines BOS/CHoCH detection with cross margin risk management.

Features:
- Market structure analysis (BOS, CHoCH, Order Blocks, FVGs)
- 25x leverage with dynamic position sizing
- Cross margin risk management
- Real-time 1-minute trading
- AMD ROCm GPU acceleration
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import time

from src.ml.features.market_structure import MarketStructureDetector, MarketStructureConfig
from src.trading.leverage_manager import LeverageManager, LeverageConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HighLeverageMemeTrader:
    """High-leverage meme coin trading system."""
    
    def __init__(self):
        # Initialize components
        self.market_detector = MarketStructureDetector()
        self.leverage_manager = LeverageManager()
        
        # Trading pairs
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        # Performance tracking
        self.trade_history = []
        self.daily_stats = {}
        
        # Check GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"High-Leverage Meme Trader using device: {self.device}")
        
        logger.info("🚀 High-Leverage Meme Coin HFT System Initialized!")
    
    def generate_sample_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Generate sample 1-minute data for testing."""
        dates = pd.date_range(
            datetime.now() - timedelta(minutes=periods), 
            periods=periods, 
            freq='1min'
        )
        
        # Generate realistic price movements
        np.random.seed(hash(symbol) % 1000)  # Different seed per symbol
        
        base_price = 0.001 if 'FART' in symbol else 0.01
        prices = [base_price]
        
        for i in range(1, periods):
            # Volatile price movements
            volatility = 0.05  # 5% volatility
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.0001))  # Minimum price
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(1000000, 10000000) for _ in prices]
        })
        
        return df
    
    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure for trading signals."""
        logger.info(f"Analyzing market structure for {len(df)} bars...")
        
        # Get market structure signals
        signals = self.market_detector.calculate_signals(df)
        
        # Get latest signals
        latest_signals = {}
        for key, tensor in signals.items():
            if len(tensor) > 0:
                latest_signals[key] = tensor[-1].item()
        
        return latest_signals
    
    def generate_trading_signals(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate trading signals based on market structure."""
        # Analyze market structure
        signals = self.analyze_market_structure(df)
        
        # Determine trading action
        long_strength = signals.get('long_signal', 0)
        short_strength = signals.get('short_signal', 0)
        
        # Signal thresholds
        entry_threshold = 0.6
        exit_threshold = 0.3
        
        action = 'hold'
        side = None
        signal_strength = 0.0
        
        # Long signal
        if long_strength > entry_threshold:
            action = 'buy'
            side = 'long'
            signal_strength = long_strength
        # Short signal
        elif short_strength > entry_threshold:
            action = 'sell'
            side = 'short'
            signal_strength = short_strength
        # Exit signals
        elif long_strength < exit_threshold and short_strength < exit_threshold:
            action = 'exit'
        
        return {
            'symbol': symbol,
            'action': action,
            'side': side,
            'signal_strength': signal_strength,
            'long_strength': long_strength,
            'short_strength': short_strength,
            'price': df['close'].iloc[-1],
            'timestamp': df['timestamp'].iloc[-1]
        }
    
    def execute_trade(self, signal: Dict) -> Dict:
        """Execute a trade based on signal."""
        symbol = signal['symbol']
        action = signal['action']
        price = signal['price']
        signal_strength = signal['signal_strength']
        
        if action == 'buy':
            # Open long position
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='long',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🟢 LONG {symbol} @ ${price:.6f} - Strength: {signal_strength:.2f}")
                return {'action': 'buy', 'success': True, 'position': result['position']}
            else:
                logger.warning(f"❌ Failed to open LONG {symbol}: {result['reason']}")
                return {'action': 'buy', 'success': False, 'reason': result['reason']}
        
        elif action == 'sell':
            # Open short position
            result = self.leverage_manager.open_position(
                symbol=symbol,
                side='short',
                price=price,
                signal_strength=signal_strength
            )
            
            if result['success']:
                logger.info(f"🔴 SHORT {symbol} @ ${price:.6f} - Strength: {signal_strength:.2f}")
                return {'action': 'sell', 'success': True, 'position': result['position']}
            else:
                logger.warning(f"❌ Failed to open SHORT {symbol}: {result['reason']}")
                return {'action': 'sell', 'success': False, 'reason': result['reason']}
        
        elif action == 'exit':
            # Close existing position
            if symbol in self.leverage_manager.positions:
                result = self.leverage_manager.close_position(symbol, price)
                if result['success']:
                    logger.info(f"🔄 EXIT {symbol} @ ${price:.6f} - PnL: ${result['pnl']:.2f}")
                    return {'action': 'exit', 'success': True, 'pnl': result['pnl']}
                else:
                    logger.warning(f"❌ Failed to exit {symbol}: {result['reason']}")
                    return {'action': 'exit', 'success': False, 'reason': result['reason']}
        
        return {'action': action, 'success': False, 'reason': 'No action taken'}
    
    def run_trading_cycle(self) -> Dict:
        """Run one complete trading cycle for all symbols."""
        logger.info("🔄 Starting trading cycle...")
        
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': [],
            'trades': [],
            'portfolio_summary': {}
        }
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                # Generate sample data (replace with real data feed)
                df = self.generate_sample_data(symbol, periods=50)
                
                # Generate trading signals
                signal = self.generate_trading_signals(symbol, df)
                cycle_results['signals'].append(signal)
                
                # Execute trade if signal exists
                if signal['action'] != 'hold':
                    trade_result = self.execute_trade(signal)
                    cycle_results['trades'].append(trade_result)
                
                # Small delay between symbols
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Update portfolio
        current_prices = {
            symbol: self.generate_sample_data(symbol, 1)['close'].iloc[-1] 
            for symbol in self.symbols
        }
        self.leverage_manager.update_positions(current_prices)
        
        # Get portfolio summary
        cycle_results['portfolio_summary'] = self.leverage_manager.get_portfolio_summary()
        
        logger.info(f"✅ Trading cycle complete - {len(cycle_results['trades'])} trades executed")
        
        return cycle_results
    
    def run_paper_trading(self, cycles: int = 10, delay_seconds: int = 60):
        """Run paper trading simulation."""
        logger.info(f"🧪 Starting paper trading simulation - {cycles} cycles")
        
        results = []
        
        for cycle in range(cycles):
            logger.info(f"📊 Cycle {cycle + 1}/{cycles}")
            
            # Run trading cycle
            cycle_result = self.run_trading_cycle()
            results.append(cycle_result)
            
            # Print summary
            summary = cycle_result['portfolio_summary']
            logger.info(
                f"💰 Balance: ${summary['current_balance']:.2f} | "
                f"Equity: ${summary['total_equity']:.2f} | "
                f"PnL: ${summary['daily_pnl']:.2f} | "
                f"Positions: {summary['active_positions']}"
            )
            
            # Wait for next cycle
            if cycle < cycles - 1:
                logger.info(f"⏳ Waiting {delay_seconds} seconds for next cycle...")
                time.sleep(delay_seconds)
        
        # Final performance report
        self.print_performance_report(results)
        
        return results
    
    def print_performance_report(self, results: List[Dict]):
        """Print comprehensive performance report."""
        logger.info("📊 PERFORMANCE REPORT")
        logger.info("=" * 50)
        
        # Get final metrics
        final_summary = results[-1]['portfolio_summary']
        performance_metrics = self.leverage_manager.get_performance_metrics()
        
        # Account performance
        logger.info(f"💰 Starting Balance: ${self.leverage_manager.config.starting_balance:.2f}")
        logger.info(f"💰 Final Balance: ${final_summary['current_balance']:.2f}")
        logger.info(f"📈 Total Return: {((final_summary['current_balance'] / self.leverage_manager.config.starting_balance) - 1) * 100:.2f}%")
        logger.info(f"📊 Daily PnL: ${final_summary['daily_pnl']:.2f}")
        logger.info(f"📉 Max Drawdown: {final_summary['current_drawdown'] * 100:.2f}%")
        
        # Trading performance
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
        
        logger.info("=" * 50)

def main():
    """Main function to run the high-leverage meme coin trader."""
    print("🚀 High-Leverage Meme Coin HFT System")
    print("=" * 50)
    
    # Create trader
    trader = HighLeverageMemeTrader()
    
    # Run paper trading simulation
    print("🧪 Starting paper trading simulation...")
    print("📊 Trading pairs: FARTCOIN, POPCAT, WIF, PONKE, SPX, GIGA")
    print("⚡ Leverage: Up to 25x")
    print("💰 Starting balance: $1,000")
    print("=" * 50)
    
    # Run simulation
    results = trader.run_paper_trading(cycles=5, delay_seconds=10)
    
    print("✅ Paper trading simulation complete!")

if __name__ == "__main__":
    main() 