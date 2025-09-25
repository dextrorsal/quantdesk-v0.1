#!/usr/bin/env python3
"""
Live Lag Trading CLI

Command-line interface for running the live lag-based trading model.
This integrates with the Quantify CLI system and provides real-time trading capabilities.

Usage:
    python -m src.cli.trading.lag_live_trader run --config path/to/config.json
    python -m src.cli.trading.lag_live_trader status
    python -m src.cli.trading.lag_live_trader test
"""

import asyncio
import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ml.models.strategy.lag_trading_model import LagTradingModel
from src.data.csv_storage import CSVStorage, StorageConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/lag_live_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LagLiveTraderCLI:
    """
    CLI interface for the live lag trading model.
    """
    
    def __init__(self):
        """Initialize the CLI interface."""
        self.model = None
        self.trader = None
        self.storage = CSVStorage(StorageConfig(data_path="data/historical/processed"))
    
    async def run_trader(self, config_path: Optional[str] = None, 
                        cycle_interval: int = 60, max_cycles: Optional[int] = None):
        """
        Run the live trading model.
        
        Args:
            config_path: Path to model configuration
            cycle_interval: Seconds between trading cycles
            max_cycles: Maximum number of cycles to run (None for infinite)
        """
        logger.info("🚀 Starting Live Lag Trading Model")
        
        try:
            # Initialize model
            self.model = LagTradingModel(config_path)
            logger.info("✅ Model initialized successfully")
            
            # Initialize trader (placeholder for now)
            # self.trader = LagModelTrader(self.model, exchange_client)
            
            # Run trading cycles
            cycle_count = 0
            while max_cycles is None or cycle_count < max_cycles:
                cycle_count += 1
                logger.info(f"🔄 Starting trading cycle {cycle_count}")
                
                try:
                    # Get live data
                    live_data = await self._get_live_data()
                    
                    if not live_data:
                        logger.warning("No live data available, skipping cycle")
                        await asyncio.sleep(cycle_interval)
                        continue
                    
                    # Generate signals
                    signals = await self.model.generate_signals(live_data)
                    
                    if signals:
                        logger.info(f"📊 Generated {len(signals)} signals")
                        for signal in signals:
                            logger.info(f"  {signal['pair']}: {signal['action']} "
                                      f"{signal['follower']} (conf: {signal['confidence']:.2f})")
                    else:
                        logger.info("No signals generated in this cycle")
                    
                    # Simulate portfolio for testing
                    portfolio = {
                        'total_value': 10000,
                        'positions': {},
                        'cash': 10000
                    }
                    
                    # Execute signals
                    trades = self.model.execute_signals(signals, portfolio)
                    
                    if trades:
                        logger.info(f"💼 Executing {len(trades)} trades")
                        for trade in trades:
                            logger.info(f"  {trade['action']} {trade['quantity']} "
                                      f"{trade['symbol']} @ {trade['price']}")
                    else:
                        logger.info("No trades to execute")
                    
                    # Update model
                    await self.model.update_model(live_data)
                    
                    # Save model state periodically
                    if cycle_count % 10 == 0:
                        self.model.save_model_state("results/lag_strategy/trading_model/model_state.json")
                        logger.info("💾 Model state saved")
                    
                    logger.info(f"✅ Cycle {cycle_count} complete")
                    
                except Exception as e:
                    logger.error(f"❌ Error in cycle {cycle_count}: {e}")
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval)
            
            logger.info("🏁 Trading completed")
            
        except KeyboardInterrupt:
            logger.info("⏹️ Trading stopped by user")
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            raise
    
    async def _get_live_data(self) -> dict:
        """
        Get live market data for all configured pairs.
        
        Returns:
            Dict with symbol -> DataFrame mapping
        """
        if not self.model:
            return {}
        
        live_data = {}
        end_date = datetime.now()
        start_date = end_date.replace(minute=end_date.minute - 60)  # Last hour
        
        for pair_name, pair_config in self.model.model_config['pairs'].items():
            leader = pair_config['leader']
            follower = pair_config['follower']
            
            # Get data for both assets
            for symbol in [leader, follower]:
                if symbol not in live_data:
                    try:
                        # Try to get recent data from storage
                        data = None
                        for exchange in ['bitget', 'binance', 'kraken']:
                            for timeframe in ['1m', '5m', '5min']:
                                try:
                                    data = await self.storage.load_candles(
                                        exchange, f"{symbol}USDT", timeframe, 
                                        start_date, end_date
                                    )
                                    if not data.empty:
                                        break
                                except:
                                    continue
                            if data is not None and not data.empty:
                                break
                        
                        if data is not None and not data.empty:
                            live_data[symbol] = data
                        else:
                            logger.warning(f"No data found for {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error getting data for {symbol}: {e}")
        
        return live_data
    
    def show_status(self):
        """Show current model status."""
        if not self.model:
            logger.error("Model not initialized")
            return
        
        status = self.model.get_model_status()
        
        print("\n📊 Lag Trading Model Status")
        print("=" * 40)
        print(f"Model Version: {status['model_version']}")
        print(f"Active Pairs: {status['active_pairs']}")
        print(f"Total Signals Generated: {status['total_signals_generated']}")
        print(f"Recent Signals (1h): {status['recent_signals']}")
        
        print("\n🎯 Risk Management:")
        risk = status['risk_management']
        print(f"  Max Positions: {risk['max_positions']}")
        print(f"  Risk per Trade: {risk['risk_per_trade']*100}%")
        print(f"  Stop Loss: {risk['stop_loss_pct']*100}%")
        print(f"  Take Profit: {risk['take_profit_pct']*100}%")
        
        print("\n📡 Signal Generation:")
        sig_gen = status['signal_generation']
        print(f"  Min Correlation: {sig_gen['min_correlation']}")
        print(f"  Max Lag Minutes: {sig_gen['max_lag_minutes']}")
        print(f"  Volume Confirmation: {sig_gen['volume_confirmation']}")
        print(f"  Confidence Threshold: {sig_gen['signal_confidence_threshold']}")
        
        # Show configured pairs
        print("\n🔗 Configured Pairs:")
        for pair_name, pair_config in self.model.model_config['pairs'].items():
            print(f"  {pair_name}: threshold={pair_config['threshold']}%, "
                  f"priority={pair_config['priority']}")
    
    async def test_model(self):
        """Test the model with sample data."""
        logger.info("🧪 Testing Lag Trading Model")
        
        try:
            # Initialize model
            self.model = LagTradingModel()
            
            # Get some real data for testing
            end_date = datetime.now()
            start_date = end_date.replace(hour=end_date.hour - 2)  # Last 2 hours
            
            test_data = {}
            
            # Try to get BTC and PEPE data
            for symbol in ['BTC', 'PEPE']:
                try:
                    for exchange in ['bitget', 'binance', 'kraken']:
                        for timeframe in ['1m', '5m', '5min']:
                            try:
                                data = await self.storage.load_candles(
                                    exchange, f"{symbol}USDT", timeframe, 
                                    start_date, end_date
                                )
                                if not data.empty:
                                    test_data[symbol] = data
                                    break
                            except:
                                continue
                        if symbol in test_data:
                            break
                except Exception as e:
                    logger.error(f"Error getting test data for {symbol}: {e}")
            
            if not test_data:
                logger.warning("No test data available, using sample data")
                # Use sample data if no real data available
                import pandas as pd
                import numpy as np
                
                dates = pd.date_range(start='2024-01-01', end='2024-01-01 02:00', freq='1min')
                test_data = {
                    'BTC': pd.DataFrame({
                        'open': np.random.normal(45000, 100, len(dates)),
                        'high': np.random.normal(45100, 100, len(dates)),
                        'low': np.random.normal(44900, 100, len(dates)),
                        'close': np.random.normal(45000, 100, len(dates)),
                        'volume': np.random.normal(1000, 100, len(dates))
                    }, index=dates),
                    'PEPE': pd.DataFrame({
                        'open': np.random.normal(0.0001, 0.00001, len(dates)),
                        'high': np.random.normal(0.00011, 0.00001, len(dates)),
                        'low': np.random.normal(0.00009, 0.00001, len(dates)),
                        'close': np.random.normal(0.0001, 0.00001, len(dates)),
                        'volume': np.random.normal(1000000, 100000, len(dates))
                    }, index=dates)
                }
            
            # Generate signals
            signals = await self.model.generate_signals(test_data)
            
            print(f"\n✅ Test completed successfully!")
            print(f"📊 Generated {len(signals)} signals")
            
            for signal in signals:
                print(f"  {signal['pair']}: {signal['action']} {signal['follower']} "
                      f"(confidence: {signal['confidence']:.2f})")
            
            # Show model status
            status = self.model.get_model_status()
            print(f"\n📈 Model Status:")
            print(f"  Active Pairs: {status['active_pairs']}")
            print(f"  Total Signals: {status['total_signals_generated']}")
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
            raise


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Live Lag Trading Model CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run the live trading model')
    run_parser.add_argument('--config', type=str, help='Path to model configuration file')
    run_parser.add_argument('--interval', type=int, default=60, 
                           help='Trading cycle interval in seconds (default: 60)')
    run_parser.add_argument('--max-cycles', type=int, 
                           help='Maximum number of trading cycles (default: infinite)')
    
    # Status command
    subparsers.add_parser('status', help='Show model status')
    
    # Test command
    subparsers.add_parser('test', help='Test the model with sample data')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create CLI instance
    cli = LagLiveTraderCLI()
    
    # Execute command
    if args.command == 'run':
        asyncio.run(cli.run_trader(
            config_path=args.config,
            cycle_interval=args.interval,
            max_cycles=args.max_cycles
        ))
    elif args.command == 'status':
        cli.show_status()
    elif args.command == 'test':
        asyncio.run(cli.test_model())


if __name__ == "__main__":
    main() 