#!/usr/bin/env python3
"""
Example Paper Trading Script

This script demonstrates how to use the PaperTradingFramework to:
1. Run backtests for different strategies
2. Optimize strategy parameters
3. Compare multiple strategies
4. Generate performance reports

Usage:
    python scripts/example_paper_trading.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.paper_trading_framework import (
    PaperTradingFramework, 
    BacktestConfig, 
    StrategyType
)


async def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    # Create realistic price data
    dates = pd.date_range(start='2024-01-01', end='2024-03-01', freq='1h')
    
    # BTC data with trend and volatility
    np.random.seed(42)
    btc_returns = np.random.normal(0.0001, 0.02, len(dates))  # 2% hourly volatility
    btc_prices = 50000 * np.exp(np.cumsum(btc_returns))
    
    btc_data = pd.DataFrame({
        'timestamp': dates,
        'open': btc_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': btc_prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
        'low': btc_prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
        'close': btc_prices,
        'volume': np.random.uniform(100, 1000, len(dates))
    })
    
    # ETH data (correlated with BTC)
    eth_returns = btc_returns * 0.7 + np.random.normal(0, 0.015, len(dates))
    eth_prices = 3000 * np.exp(np.cumsum(eth_returns))
    
    eth_data = pd.DataFrame({
        'timestamp': dates,
        'open': eth_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': eth_prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
        'low': eth_prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
        'close': eth_prices,
        'volume': np.random.uniform(50, 500, len(dates))
    })
    
    # SOL data (more volatile)
    sol_returns = btc_returns * 0.5 + np.random.normal(0, 0.03, len(dates))
    sol_prices = 100 * np.exp(np.cumsum(sol_returns))
    
    sol_data = pd.DataFrame({
        'timestamp': dates,
        'open': sol_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': sol_prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
        'low': sol_prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
        'close': sol_prices,
        'volume': np.random.uniform(200, 2000, len(dates))
    })
    
    # WIF data (follower with lag)
    wif_returns = btc_returns * 0.3 + np.random.normal(0, 0.04, len(dates))
    wif_prices = 2.0 * np.exp(np.cumsum(wif_returns))
    
    wif_data = pd.DataFrame({
        'timestamp': dates,
        'open': wif_prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': wif_prices * (1 + abs(np.random.normal(0, 0.002, len(dates)))),
        'low': wif_prices * (1 - abs(np.random.normal(0, 0.002, len(dates)))),
        'close': wif_prices,
        'volume': np.random.uniform(1000, 10000, len(dates))
    })
    
    print(f"✅ Created sample data: {len(dates)} hours of data")
    return {
        'BTC': btc_data,
        'ETH': eth_data,
        'SOL': sol_data,
        'WIF': wif_data
    }


async def example_1_single_strategy_backtest():
    """Example 1: Run backtest for a single strategy"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Strategy Backtest")
    print("="*60)
    
    # Create sample data
    data = await create_sample_data()
    
    # Initialize framework
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        max_positions=3
    )
    
    framework = PaperTradingFramework(config)
    
    # Run backtest for Lorentzian strategy
    print("🧠 Testing Lorentzian Classifier...")
    results = await framework.backtest_strategy(
        strategy_name=StrategyType.LORENTZIAN,
        data=data['BTC']  # Single asset
    )
    
    # Print results
    metrics = results['metrics']
    print(f"\n📈 Results:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Win Rate: {metrics.win_rate:.1%}")
    print(f"   Total Return: {metrics.total_return:.2f}%")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2f}%")
    
    # Plot results
    framework.plot_results(results)
    
    return results


async def example_2_multi_asset_strategy():
    """Example 2: Run lag-based strategy with multiple assets"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Asset Lag-Based Strategy")
    print("="*60)
    
    # Create sample data
    data = await create_sample_data()
    
    # Initialize framework
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        max_positions=3
    )
    
    framework = PaperTradingFramework(config)
    
    # Run backtest for lag-based strategy
    print("⏰ Testing Lag-Based Strategy...")
    results = await framework.backtest_strategy(
        strategy_name=StrategyType.LAG_BASED,
        data=data  # Multiple assets
    )
    
    # Print results
    metrics = results['metrics']
    print(f"\n📈 Results:")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Win Rate: {metrics.win_rate:.1%}")
    print(f"   Total Return: {metrics.total_return:.2f}%")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2f}%")
    
    return results


async def example_3_parameter_optimization():
    """Example 3: Optimize strategy parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Parameter Optimization")
    print("="*60)
    
    # Create sample data
    data = await create_sample_data()
    
    # Initialize framework
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        max_positions=3
    )
    
    framework = PaperTradingFramework(config)
    
    # Define parameter ranges for Lorentzian strategy
    param_ranges = {
        'lookback_bars': [30, 50, 70],
        'k_neighbors': [10, 20, 30],
        'prediction_bars': [2, 4, 6]
    }
    
    print("🔧 Optimizing Lorentzian parameters...")
    results = await framework.optimize_strategy_parameters(
        strategy_name=StrategyType.LORENTZIAN,
        data=data['BTC'],
        param_ranges=param_ranges,
        metric='total_return'
    )
    
    # Print results
    print(f"\n🏆 Best Parameters: {results['best_params']}")
    print(f"📊 Best Return: {results['best_metric']:.2f}%")
    print(f"🔍 Combinations Tested: {len(results['all_results'])}")
    
    # Show top 3 results
    print(f"\n🥇 Top 3 Results:")
    for i, result in enumerate(results['all_results'][:3]):
        print(f"   {i+1}. {result['params']} -> {result['metric']:.2f}%")
    
    return results


async def example_4_strategy_comparison():
    """Example 4: Compare multiple strategies"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Strategy Comparison")
    print("="*60)
    
    # Create sample data
    data = await create_sample_data()
    
    # Initialize framework
    config = BacktestConfig(
        initial_capital=10000,
        commission=0.001,
        slippage=0.0005,
        max_positions=3
    )
    
    framework = PaperTradingFramework(config)
    
    # Test multiple strategies
    strategies = [
        StrategyType.LORENTZIAN,
        StrategyType.LOGISTIC_REGRESSION,
        StrategyType.CHANDELIER_EXIT
    ]
    
    comparison_results = {}
    
    for strategy in strategies:
        print(f"🧪 Testing {strategy.value}...")
        
        try:
            results = await framework.backtest_strategy(
                strategy_name=strategy,
                data=data['BTC']
            )
            comparison_results[strategy.value] = results
            
        except Exception as e:
            print(f"❌ Error testing {strategy.value}: {e}")
            comparison_results[strategy.value] = None
    
    # Print comparison
    print(f"\n📊 Strategy Comparison:")
    print(f"{'Strategy':<20} {'Return %':<10} {'Win Rate %':<12} {'Sharpe':<8} {'Trades':<8}")
    print("-" * 60)
    
    for strategy_name, results in comparison_results.items():
        if results is not None:
            metrics = results['metrics']
            print(f"{strategy_name:<20} {metrics.total_return:<10.2f} "
                  f"{metrics.win_rate*100:<12.1f} {metrics.sharpe_ratio:<8.2f} "
                  f"{metrics.total_trades:<8}")
        else:
            print(f"{strategy_name:<20} {'N/A':<10} {'N/A':<12} {'N/A':<8} {'N/A':<8}")
    
    return comparison_results


async def example_5_custom_parameters():
    """Example 5: Test with custom parameters"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Parameters")
    print("="*60)
    
    # Create sample data
    data = await create_sample_data()
    
    # Initialize framework with custom config
    config = BacktestConfig(
        initial_capital=50000,  # Higher capital
        commission=0.0005,      # Lower commission
        slippage=0.0002,        # Lower slippage
        max_positions=5,        # More positions
        risk_per_trade=0.01,    # Lower risk per trade
        stop_loss_pct=0.03,     # Tighter stop loss
        take_profit_pct=0.06    # Lower take profit
    )
    
    framework = PaperTradingFramework(config)
    
    # Test with custom parameters
    custom_params = {
        'lookback_bars': 40,
        'k_neighbors': 15,
        'prediction_bars': 3
    }
    
    print("⚙️ Testing with custom parameters...")
    results = await framework.backtest_strategy(
        strategy_name=StrategyType.LORENTZIAN,
        data=data['BTC'],
        params=custom_params
    )
    
    # Print results
    metrics = results['metrics']
    print(f"\n📈 Custom Parameters Results:")
    print(f"   Parameters: {custom_params}")
    print(f"   Total Trades: {metrics.total_trades}")
    print(f"   Win Rate: {metrics.win_rate:.1%}")
    print(f"   Total Return: {metrics.total_return:.2f}%")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.2f}%")
    
    return results


async def main():
    """Run all examples"""
    print("🚀 Paper Trading Framework Examples")
    print("="*60)
    
    try:
        # Run all examples
        await example_1_single_strategy_backtest()
        await example_2_multi_asset_strategy()
        await example_3_parameter_optimization()
        await example_4_strategy_comparison()
        await example_5_custom_parameters()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60)
        print("\n📁 Results saved to: results/paper_trading/")
        print("📊 Check the generated plots and CSV files for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main()) 