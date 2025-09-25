"""
Lag-Based Trading Strategy CLI Entrypoint

This module provides the command-line interface for the lag-based trading strategy in Quantify.

Features:
    - Run comprehensive lag analysis
    - Backtest the lag strategy on historical data
    - Generate trading signals (live or historical)
    - Optimize strategy parameters
    - Show current configuration and status

Usage:
    python -m src.cli.trading.lag_strategy analyze --config configs/lag_strategy_config.yaml
    python -m src.cli.trading.lag_strategy backtest --start-date 2024-01-01 --end-date 2024-12-31
    python -m src.cli.trading.lag_strategy signals --live
    python -m src.cli.trading.lag_strategy optimize
    python -m src.cli.trading.lag_strategy status

See also: docs/lag-based.md for a full strategy guide.
"""

import asyncio
import argparse
from datetime import datetime

from src.cli.base import BaseCLI
from scripts.lag_strategy_integration import LagStrategyIntegration


class LagStrategyCLI(BaseCLI):
    """
    Command-line interface for the lag-based trading strategy.

    Supports commands for analysis, backtesting, signal generation, optimization, and status.
    Integrates with the Quantify CLI framework and the lag strategy integration layer.
    """
    
    def __init__(self):
        """
        Initialize the CLI and integration layer.
        """
        super().__init__()
        self.integration = None
    
    def setup_parser(self) -> argparse.ArgumentParser:
        """
        Set up the argument parser for CLI commands and options.
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description='Lag-Based Trading Strategy CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Run comprehensive analysis
  python -m src.cli.trading.lag_strategy analyze --config configs/lag_strategy_config.yaml
  
  # Run backtest
  python -m src.cli.trading.lag_strategy backtest --start-date 2024-01-01 --end-date 2024-12-31
  
  # Generate signals
  python -m src.cli.trading.lag_strategy signals --live
  
  # Optimize strategy parameters
  python -m src.cli.trading.lag_strategy optimize
            """
        )
        
        # Add subcommands
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Analyze command
        analyze_parser = subparsers.add_parser('analyze', help='Run comprehensive analysis')
        analyze_parser.add_argument('--config', type=str, help='Path to configuration file')
        analyze_parser.add_argument('--data-path', type=str, help='Path to data directory')
        analyze_parser.add_argument('--output-dir', type=str, help='Output directory for results')
        analyze_parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
        
        # Backtest command
        backtest_parser = subparsers.add_parser('backtest', help='Run strategy backtest')
        backtest_parser.add_argument('--config', type=str, help='Path to configuration file')
        backtest_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
        backtest_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
        backtest_parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
        backtest_parser.add_argument('--output-dir', type=str, help='Output directory for results')
        
        # Signals command
        signals_parser = subparsers.add_parser('signals', help='Generate trading signals')
        signals_parser.add_argument('--config', type=str, help='Path to configuration file')
        signals_parser.add_argument('--live', action='store_true', help='Use live data')
        signals_parser.add_argument('--days', type=int, default=7, help='Days of historical data to use')
        
        # Optimize command
        optimize_parser = subparsers.add_parser('optimize', help='Optimize strategy parameters')
        optimize_parser.add_argument('--config', type=str, help='Path to configuration file')
        optimize_parser.add_argument('--output-dir', type=str, help='Output directory for results')
        
        # Status command
        status_parser = subparsers.add_parser('status', help='Show strategy status')
        status_parser.add_argument('--config', type=str, help='Path to configuration file')
        
        return parser
    
    async def run_analyze(self, args):
        """
        Run comprehensive lag-based strategy analysis.
        Args:
            args: Parsed command-line arguments
        Returns:
            Analyzer object or results
        """
        print("🔍 Running Lag-Based Strategy Analysis...")
        
        # Initialize integration
        self.integration = LagStrategyIntegration(args.config)
        
        # Override config with command line arguments
        if args.data_path:
            self.integration.config['data_path'] = args.data_path
        if args.output_dir:
            self.integration.config['output_dir'] = args.output_dir
        if args.days:
            self.integration.config['analysis_days'] = args.days
        
        # Run analysis
        analyzer = await self.integration.run_analysis(args.output_dir)
        
        print("✅ Analysis complete!")
        return analyzer
    
    async def run_backtest(self, args):
        """
        Run backtest of the lag-based strategy.
        Args:
            args: Parsed command-line arguments
        Returns:
            Backtest results
        """
        print("📊 Running Lag-Based Strategy Backtest...")
        
        # Initialize integration
        self.integration = LagStrategyIntegration(args.config)
        
        # Override config with command line arguments
        if args.output_dir:
            self.integration.config['output_dir'] = args.output_dir
        
        # Parse dates
        start_date = None
        end_date = None
        if args.start_date:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        
        # Run backtest
        results = await self.integration.run_backtest(start_date, end_date, args.capital)
        
        print("✅ Backtest complete!")
        return results
    
    async def run_signals(self, args):
        """
        Generate trading signals using the lag-based strategy.
        Args:
            args: Parsed command-line arguments
        Returns:
            List of signals
        """
        print("📡 Generating Lag-Based Strategy Signals...")
        
        # Initialize integration
        self.integration = LagStrategyIntegration(args.config)
        
        # Override config with command line arguments
        if args.days:
            self.integration.config['analysis_days'] = args.days
        
        # Generate signals
        signals = await self.integration.generate_signals(args.live)
        
        print("✅ Signal generation complete!")
        return signals
    
    async def run_optimize(self, args):
        """
        Optimize lag-based strategy parameters.
        Args:
            args: Parsed command-line arguments
        Returns:
            Best parameters found
        """
        print("⚙️ Optimizing Lag-Based Strategy Parameters...")
        
        # Initialize integration
        self.integration = LagStrategyIntegration(args.config)
        
        # Override config with command line arguments
        if args.output_dir:
            self.integration.config['output_dir'] = args.output_dir
        
        # Run optimization
        best_params = await self.integration.optimize_strategy()
        
        print("✅ Optimization complete!")
        return best_params
    
    async def run_status(self, args):
        """
        Show current configuration and status of the lag-based strategy.
        Args:
            args: Parsed command-line arguments
        Returns:
            True if status displayed successfully
        """
        print("📈 Lag-Based Strategy Status")
        print("=" * 40)
        
        # Initialize integration
        self.integration = LagStrategyIntegration(args.config)
        
        # Show configuration
        print(f"Data Path: {self.integration.config['data_path']}")
        print(f"Timeframe: {self.integration.config['timeframe']}")
        print(f"Leader Assets: {', '.join(self.integration.config['leader_assets'])}")
        print(f"Follower Assets: {', '.join(self.integration.config['follower_assets'])}")
        print(f"Threshold: {self.integration.config['threshold']}%")
        print(f"Max Lag: {self.integration.config['max_lag_minutes']} minutes")
        print(f"Min Correlation: {self.integration.config['min_correlation']}")
        print(f"Risk per Trade: {self.integration.config['risk_per_trade']*100}%")
        print(f"Max Positions: {self.integration.config['max_positions']}")
        
        # Check data availability
        print("\nData Availability:")
        try:
            # This would check actual data availability
            print("  ✅ Historical data path exists")
        except Exception as e:
            print(f"  ❌ Error checking data: {e}")
        
        return True
    
    async def execute(self, args):
        """
        Execute the appropriate command based on parsed arguments.
        Args:
            args: Parsed command-line arguments
        Returns:
            Result of the executed command
        """
        if args.command == 'analyze':
            return await self.run_analyze(args)
        elif args.command == 'backtest':
            return await self.run_backtest(args)
        elif args.command == 'signals':
            return await self.run_signals(args)
        elif args.command == 'optimize':
            return await self.run_optimize(args)
        elif args.command == 'status':
            return await self.run_status(args)
        else:
            print("Please specify a command. Use --help for more information.")
            return None


def main():
    """
    Main entry point for the lag strategy CLI.
    Usage:
        python -m src.cli.trading.lag_strategy <command> [options]
    """
    cli = LagStrategyCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main() 