#!/usr/bin/env python3
"""
Lag-Based Strategy Integration Script

This script is the main integration point for the QuantDesk lag-based trading strategy.

Purpose:
    - Loads historical OHLCV data from your CSV storage system
    - Runs lag-based analysis, backtesting, and signal generation
    - Integrates with the QuantDesk framework and CLI
    - Supports both all-followers and grouped-followers modes
    - Saves results and logs for further research

How to Use:
    $ python scripts/lag_strategy_integration.py --action analyze --data-path data/historical/processed
    $ python scripts/lag_strategy_integration.py --action backtest --config configs/lag_strategy_config.yaml
    $ python scripts/lag_strategy_integration.py --action signals --live

What is the Lag Strategy?
    The lag-based strategy identifies price moves in major assets (BTC, ETH, SOL) that are followed, with a delay, by meme/DeFi coins. It quantifies lag, correlation, and response rate, helping you discover which coins reliably follow leaders and with what delay.

See also: docs/lag-based.md for a full strategy guide.
"""

import asyncio
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

# Import QuantDesk modules
from src.data.csv_storage import CSVStorage, StorageConfig
from src.ml.models.strategy.lag_based_strategy import LagBasedStrategy, StrategyConfig
from src.ml.models.strategy.lag_analysis_tools import LagAnalysisTools
from src.utils.log_setup import setup_logging

logger = logging.getLogger(__name__)


class LagStrategyIntegration:
    """
    Main integration class for the lag-based trading strategy.

    - Loads and manages configuration
    - Loads historical data for leaders and followers
    - Runs analysis, backtesting, and signal generation
    - Handles both all-followers and grouped-followers modes
    - Used by CLI and scripts for all lag strategy operations
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integration with config and storage.
        Args:
            config_path: Path to YAML config file
        """
        self.config = self._load_config(config_path)
        self.storage = CSVStorage(StorageConfig(data_path=self.config['data_path']))
        self.strategy = None
        self.analyzer = LagAnalysisTools()
        
        # Setup logging
        setup_logging()
        
        logger.info("Initialized LagStrategyIntegration")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from YAML file or use defaults.
        Args:
            config_path: Path to config file
        Returns:
            Configuration dictionary
        """
        default_config = {
            'data_path': 'data/historical/processed',
            'leader_assets': ['BTC', 'ETH', 'SOL'],
            'follower_assets': ['WIF', 'FARTCOIN', 'POPCAT', 'BOME', 'PEPE'],
            'threshold': 1.5,
            'max_lag_minutes': 60,
            'min_correlation': 0.3,
            'risk_per_trade': 0.02,
            'max_positions': 3,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'timeframe': '5min',
            'analysis_days': 30,
            'output_dir': 'results/lag_strategy'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    async def load_historical_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Load historical OHLCV data for all leaders and followers.
        Args:
            start_date: Start datetime
            end_date: End datetime
        Returns:
            Dict of symbol -> DataFrame
        """
        all_data = {}
        # Collect all unique assets needed
        assets_to_load = set(self.config['leader_assets'])
        if self.config.get('grouped_mode', False) and self.config.get('grouped_followers'):
            for followers in self.config['grouped_followers'].values():
                assets_to_load.update(followers)
        else:
            assets_to_load.update(self.config['follower_assets'])
        
        # Load leader assets
        for asset in self.config['leader_assets']:
            try:
                # Try different exchanges
                for exchange in ['bitget', 'binance', 'kraken']:
                    try:
                        data = await self.storage.load_candles(
                            exchange, f"{asset}USDT", self.config['timeframe'],
                            start_date, end_date
                        )
                        if not data.empty:
                            all_data[asset] = data
                            logger.info(f"Loaded {asset} data from {exchange}: {len(data)} records")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to load {asset} from {exchange}: {e}")
                        continue
                
                if asset not in all_data:
                    logger.warning(f"Could not load data for {asset}")
                    
            except Exception as e:
                logger.error(f"Error loading {asset}: {e}")
        
        # Load follower assets
        for asset in assets_to_load: # Use assets_to_load here
            if asset in self.config['follower_assets']: # Check if it's a follower asset
                try:
                    # Try different exchanges
                    for exchange in ['bitget', 'binance', 'kraken']:
                        try:
                            data = await self.storage.load_candles(
                                exchange, f"{asset}USDT", self.config['timeframe'],
                                start_date, end_date
                            )
                            if not data.empty:
                                all_data[asset] = data
                                logger.info(f"Loaded {asset} data from {exchange}: {len(data)} records")
                                break
                        except Exception as e:
                            logger.debug(f"Failed to load {asset} from {exchange}: {e}")
                            continue
                    
                    if asset not in all_data:
                        logger.warning(f"Could not load data for {asset}")
                        
                except Exception as e:
                    logger.error(f"Error loading {asset}: {e}")
        
        logger.info(f"Loaded data for {len(all_data)} assets")
        return all_data
    
    async def run_analysis(self, output_dir: Optional[str] = None):
        """
        Run comprehensive lag-based analysis and save results.
        Args:
            output_dir: Directory to save analysis results
        Returns:
            Analysis results or None if insufficient data
        """
        if output_dir is None:
            output_dir = self.config['output_dir']
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['analysis_days'])
        
        logger.info(f"Running analysis from {start_date} to {end_date}")
        
        # Load data
        all_data = await self.load_historical_data(start_date, end_date)
        
        if len(all_data) < 2:
            logger.error("Insufficient data for analysis")
            return
        
        # Determine leader/follower mapping based on mode
        if self.config.get('grouped_mode', False) and self.config.get('grouped_followers'):
            leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
            grouped_followers = self.config['grouped_followers']
            # Build mapping: {leader: {follower: df, ...}, ...}
            follower_data = {leader: {f: all_data[f] for f in followers if f in all_data}
                             for leader, followers in grouped_followers.items()}
            # Pass both leader_data and follower_data to analyzer/strategy
        else:
            leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
            follower_data = {k: v for k, v in all_data.items() if k in self.config['follower_assets']}
        
        # Run analyses
        logger.info("Running move distribution analysis...")
        move_results = self.analyzer.analyze_move_distributions(leader_data, follower_data)
        
        logger.info("Running correlation analysis...")
        corr_results = self.analyzer.analyze_correlations(leader_data, follower_data)
        
        logger.info("Running threshold optimization...")
        opt_results = self.analyzer.optimize_thresholds_comprehensive(leader_data, follower_data)
        
        # Generate report
        logger.info("Generating analysis report...")
        report = self.analyzer.generate_analysis_report(f"{output_dir}/analysis_report.txt")
        
        # Save results
        self.analyzer.save_results(f"{output_dir}/analysis_results.json")
        
        # Create plots
        self.analyzer.plot_analysis_results(f"{output_dir}/analysis_plots.png")
        
        logger.info(f"Analysis complete. Results saved to {output_dir}")
        
        # Print summary
        print("\n" + "="*60)
        print("LAG-BASED STRATEGY ANALYSIS SUMMARY")
        print("="*60)
        print(report)
        
        return self.analyzer
    
    async def run_backtest(self, start_date: Optional[datetime] = None, 
                          end_date: Optional[datetime] = None,
                          initial_capital: float = 10000) -> Dict:
        """
        Run backtest of the lag-based strategy.
        
        Args:
            start_date: Start datetime
            end_date: End datetime
            initial_capital: Starting capital for backtest
        Returns:
            Backtest results dict
        """
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=self.config['analysis_days'])
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Load data
        all_data = await self.load_historical_data(start_date, end_date)
        
        if len(all_data) < 2:
            logger.error("Insufficient data for backtest")
            return {}
        
        # Determine leader/follower mapping based on mode
        if self.config.get('grouped_mode', False) and self.config.get('grouped_followers'):
            leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
            grouped_followers = self.config['grouped_followers']
            follower_data = {leader: {f: all_data[f] for f in followers if f in all_data}
                             for leader, followers in grouped_followers.items()}
        else:
            leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
            follower_data = {k: v for k, v in all_data.items() if k in self.config['follower_assets']}
        
        # Initialize strategy
        strategy_config = StrategyConfig(
            leader_assets=self.config['leader_assets'],
            follower_assets=self.config['follower_assets'],
            threshold=self.config['threshold'],
            max_lag_minutes=self.config['max_lag_minutes'],
            min_correlation=self.config['min_correlation'],
            risk_per_trade=self.config['risk_per_trade'],
            max_positions=self.config['max_positions'],
            stop_loss_pct=self.config['stop_loss_pct'],
            take_profit_pct=self.config['take_profit_pct']
        )
        
        self.strategy = LagBasedStrategy(strategy_config)
        
        # Run backtest
        results = self.strategy.backtest(all_data, initial_capital)
        
        # Print results
        print("\n" + "="*60)
        print("LAG-BASED STRATEGY BACKTEST RESULTS")
        print("="*60)
        print(f"Total trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.1%}")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print(f"Final capital: ${results['final_capital']:,.2f}")
        print(f"Return: {results['return_pct']:.2f}%")
        print(f"Signals generated: {results['signals_generated']}")
        
        # Save results
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_df = pd.DataFrame([results])
        results_df.to_csv(output_dir / "backtest_results.csv", index=False)
        
        logger.info(f"Backtest complete. Results saved to {output_dir}")
        
        return results
    
    async def generate_signals(self, live: bool = False) -> List:
        """
        Generate trading signals using the lag-based strategy.
        
        Args:
            live: If True, use live data
        Returns:
            List of signals
        """
        if live:
            # For live signals, we'd need real-time data feeds
            # This is a placeholder for live signal generation
            logger.info("Live signal generation not yet implemented")
            return []
        else:
            # Use recent historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # Last 7 days
            
            logger.info(f"Generating signals from {start_date} to {end_date}")
            
            # Load data
            all_data = await self.load_historical_data(start_date, end_date)
            
            if len(all_data) < 2:
                logger.error("Insufficient data for signal generation")
                return []
            
            # Determine leader/follower mapping based on mode
            if self.config.get('grouped_mode', False) and self.config.get('grouped_followers'):
                leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
                grouped_followers = self.config['grouped_followers']
                follower_data = {leader: {f: all_data[f] for f in followers if f in all_data}
                                 for leader, followers in grouped_followers.items()}
            else:
                leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
                follower_data = {k: v for k, v in all_data.items() if k in self.config['follower_assets']}
            
            # Initialize strategy if not already done
            if self.strategy is None:
                strategy_config = StrategyConfig(
                    leader_assets=self.config['leader_assets'],
                    follower_assets=self.config['follower_assets'],
                    threshold=self.config['threshold'],
                    max_lag_minutes=self.config['max_lag_minutes'],
                    min_correlation=self.config['min_correlation']
                )
                self.strategy = LagBasedStrategy(strategy_config)
            
            # Generate signals
            signals = self.strategy.generate_signals(leader_data, follower_data)
            
            # Print signals
            print(f"\nGenerated {len(signals)} signals:")
            for signal in signals[-5:]:  # Show last 5 signals
                print(f"  {signal.timestamp}: {signal.leader_asset}->{signal.follower_asset} "
                      f"({signal.direction.value}) - Confidence: {signal.confidence:.2f}")
            
            return signals
    
    async def optimize_strategy(self) -> Dict:
        """
        Optimize lag-based strategy parameters.
        Returns:
            Dict of best parameters
        """
        logger.info("Running strategy optimization...")
        
        # Load data for optimization
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config['analysis_days'])
        
        all_data = await self.load_historical_data(start_date, end_date)
        
        if len(all_data) < 2:
            logger.error("Insufficient data for optimization")
            return {}
        
        # Determine leader/follower mapping based on mode
        if self.config.get('grouped_mode', False) and self.config.get('grouped_followers'):
            leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
            grouped_followers = self.config['grouped_followers']
            follower_data = {leader: {f: all_data[f] for f in followers if f in all_data}
                             for leader, followers in grouped_followers.items()}
        else:
            leader_data = {k: v for k, v in all_data.items() if k in self.config['leader_assets']}
            follower_data = {k: v for k, v in all_data.items() if k in self.config['follower_assets']}
        
        # Run optimization
        optimization_results = self.analyzer.optimize_thresholds_comprehensive(
            leader_data, follower_data
        )
        
        # Find best parameters
        best_params = self._find_best_parameters(optimization_results)
        
        print("\n" + "="*60)
        print("STRATEGY OPTIMIZATION RESULTS")
        print("="*60)
        print("Recommended parameters:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        return best_params
    
    def _find_best_parameters(self, optimization_results: Dict) -> Dict:
        """
        Find the best parameters from optimization results.
        Args:
            optimization_results: Dict of parameter sets and scores
        Returns:
            Dict of best parameters
        """
        best_params = {
            'threshold': self.config['threshold'],
            'max_lag_minutes': self.config['max_lag_minutes'],
            'min_correlation': self.config['min_correlation']
        }
        
        # Find best threshold based on average hit rate and signal frequency
        if optimization_results:
            threshold_scores = {}
            
            for pair_name, pair_results in optimization_results.items():
                for threshold_str, data in pair_results.items():
                    threshold = float(threshold_str)
                    
                    if threshold not in threshold_scores:
                        threshold_scores[threshold] = []
                    
                    # Score based on hit rate and signal frequency
                    if data['hit_rate'] > 0:
                        score = data['hit_rate'] * min(data['signal_frequency_per_day'], 5)
                        threshold_scores[threshold].append(score)
            
            # Find threshold with highest average score
            if threshold_scores:
                avg_scores = {t: np.mean(scores) for t, scores in threshold_scores.items()}
                best_threshold = max(avg_scores, key=avg_scores.get)
                best_params['threshold'] = best_threshold
        
        return best_params


async def main():
    """
    Main entry point for running the integration script directly.
    Usage:
        $ python scripts/lag_strategy_integration.py --action analyze --data-path data/historical/processed
    """
    parser = argparse.ArgumentParser(description='Lag-Based Strategy Integration')
    parser.add_argument('--action', choices=['analyze', 'backtest', 'signals', 'optimize'],
                       required=True, help='Action to perform')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-path', type=str, help='Path to data directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for results')
    parser.add_argument('--live', action='store_true', help='Use live data for signals')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital for backtest')
    
    args = parser.parse_args()
    
    # Initialize integration
    integration = LagStrategyIntegration(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        integration.config['data_path'] = args.data_path
    if args.output_dir:
        integration.config['output_dir'] = args.output_dir
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Execute action
    if args.action == 'analyze':
        await integration.run_analysis(args.output_dir)
    
    elif args.action == 'backtest':
        await integration.run_backtest(start_date, end_date, args.capital)
    
    elif args.action == 'signals':
        await integration.generate_signals(args.live)
    
    elif args.action == 'optimize':
        await integration.optimize_strategy()


if __name__ == "__main__":
    asyncio.run(main()) 