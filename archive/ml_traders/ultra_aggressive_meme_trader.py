#!/usr/bin/env python3
"""
🚀 Ultra-Aggressive Meme Coin HFT System

Maximum aggression for high-frequency meme coin trading.
Extremely low thresholds for maximum signal generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pandas as pd
import numpy as np
from typing import Dict, List
import logging
from datetime import datetime, timedelta
import time

from src.ml.features.market_structure import MarketStructureDetector, MarketStructureConfig
from src.trading.leverage_manager import LeverageManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraAggressiveMemeTrader:
    """Ultra-aggressive meme coin trading system."""
    
    def __init__(self):
        # Ultra-aggressive config
        config = MarketStructureConfig(
            min_structure_strength=0.1,  # Very low threshold
            volume_threshold=1.0         # Any volume spike
        )
        self.market_detector = MarketStructureDetector(config)
        self.leverage_manager = LeverageManager()
        
        # Trading pairs
        self.symbols = ['FARTCOIN', 'POPCAT', 'WIF', 'PONKE', 'SPX', 'GIGA']
        
        # Check GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Ultra-Aggressive Meme Trader using device: {self.device}")
        
        logger.info("🚀 Ultra-Aggressive Meme Coin HFT System Initialized!")
    
    def generate_ultra_volatile_data(self, symbol: str, periods: int = 100) -> pd.DataFrame:
        """Generate extremely volatile data for maximum signals."""
        dates = pd.date_range(
            datetime.now() - timedelta(minutes=periods), 
            periods=periods, 
            freq='1min'
        )
        
        # Generate extremely volatile price movements
        np.random.seed(hash(symbol) % 1000)
        
        base_price = 0.001 if 'FART' in symbol else 0.01
        prices = [base_price]
        
        for i in range(1, periods):
            # Ultra-high volatility
            volatility = 0.20  # 20% volatility
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.0001))
        
        # Create OHLCV data with extreme movements
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.08))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.08))) for p in prices],
            'close': prices,
            'volume': [np.random.uniform(10000000, 25000000) for _ in prices]
        })
        
        return df
    
    def generate_trading_signals(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate ultra-aggressive trading signals."""
        # Get market structure signals
        signals = self.market_detector.calculate_signals(df)
        
        # Get latest signals
        latest_signals = {}
        for key, tensor in signals.items():
            if len(tensor) > 0:
                latest_signals[key] = tensor[-1].item()
        
        # Get individual component strengths
        bos_bullish = latest_signals.get('bos_bullish', 0)
        bos_bearish = latest_signals.get('bos_bearish', 0)
        choch_bullish = latest_signals.get('choch_bullish', 0)
        choch_bearish = latest_signals.get('choch_bearish', 0)
        order_block_bullish = latest_signals.get('order_block_bullish', 0)
        order_block_bearish = latest_signals.get('order_block_bearish', 0)
        fvg_bullish = latest_signals.get('fvg_bullish', 0)
        fvg_bearish = latest_signals.get('fvg_bearish', 0)
        
        # Ultra-aggressive signal calculation
        long_strength = (
            bos_bullish * 0.4 +      # BOS is most important
            choch_bullish * 0.3 +    # CHoCH confirms trend
            order_block_bullish * 0.2 +  # Order blocks provide support
            fvg_bullish * 0.1        # FVGs are bonus
        )
        
        short_strength = (
            bos_bearish * 0.4 +      # BOS is most important
            choch_bearish * 0.3 +    # CHoCH confirms trend
            order_block_bearish * 0.2 +  # Order blocks provide resistance
            fvg_bearish * 0.1        # FVGs are bonus
        )
        
        # Ultra-low thresholds for maximum aggression
        entry_threshold = 0.05   # Extremely low
        exit_threshold = 0.02    # Extremely low
        
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
        
        # Log signal details
        logger.info(f"📊 {symbol} - Long: {long_strength:.3f}, Short: {short_strength:.3f}, Action: {action}")
        
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
        logger.info("🔄 Starting ultra-aggressive trading cycle...")
        
        cycle_results = {
            'timestamp': datetime.now(),
            'signals': [],
            'trades': [],
            'portfolio_summary': {}
        }
        
        # Process each symbol
        for symbol in self.symbols:
            try:
                # Generate ultra-volatile data
                df = self.generate_ultra_volatile_data(symbol, periods=50)
                
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
            symbol: self.generate_ultra_volatile_data(symbol, 1)['close'].iloc[-1] 
            for symbol in self.symbols
        }
        self.leverage_manager.update_positions(current_prices)
        
        # Get portfolio summary
        cycle_results['portfolio_summary'] = self.leverage_manager.get_portfolio_summary()
        
        logger.info(f"✅ Trading cycle complete - {len(cycle_results['trades'])} trades executed")
        
        return cycle_results
    
    def run_paper_trading(self, cycles: int = 3, delay_seconds: int = 5):
        """Run ultra-aggressive paper trading simulation."""
        logger.info(f"🧪 Starting ultra-aggressive paper trading - {cycles} cycles")
        
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
        logger.info("📊 ULTRA-AGGRESSIVE PERFORMANCE REPORT")
        logger.info("=" * 60)
        
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
        
        logger.info("=" * 60)

def main():
    """Main function to run the ultra-aggressive meme coin trader."""
    print("🚀 Ultra-Aggressive Meme Coin HFT System")
    print("=" * 60)
    
    # Create trader
    trader = UltraAggressiveMemeTrader()
    
    # Run paper trading simulation
    print("🧪 Starting ultra-aggressive paper trading...")
    print("📊 Trading pairs: FARTCOIN, POPCAT, WIF, PONKE, SPX, GIGA")
    print("⚡ Leverage: Up to 25x")
    print("💰 Starting balance: $1,000")
    print("🎯 Ultra-low thresholds: Entry=0.05, Exit=0.02")
    print("🔥 Ultra-high volatility: 20%")
    print("=" * 60)
    
    # Run simulation
    results = trader.run_paper_trading(cycles=3, delay_seconds=5)
    
    print("✅ Ultra-aggressive paper trading complete!")

if __name__ == "__main__":
    main() 