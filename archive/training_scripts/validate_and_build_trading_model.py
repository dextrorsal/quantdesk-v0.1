#!/usr/bin/env python3
"""
Lag Strategy Validation and Trading Model Builder

This script validates the lag-based strategy analysis, optimizes parameters,
runs comprehensive backtests, and builds a trading model for integration.

Purpose:
    - Validate analysis results and identify viable pairs
    - Optimize thresholds for different market conditions
    - Run backtests to measure performance
    - Build a production-ready trading model
    - Generate trading signals for live deployment

Usage:
    python scripts/validate_and_build_trading_model.py

Output:
    - Validation report in results/lag_strategy/validation/
    - Optimized parameters in results/lag_strategy/optimization/
    - Backtest results in results/lag_strategy/backtests/
    - Trading model in results/lag_strategy/trading_model/
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from data.csv_storage import CSVStorage, StorageConfig
from ml.models.strategy.lag_based_strategy import LagBasedStrategy, StrategyConfig
from ml.models.strategy.lag_analysis_tools import LagAnalysisTools


class LagStrategyValidator:
    """
    Validates lag strategy analysis and builds trading models.
    """
    
    def __init__(self):
        """Initialize the validator with storage and tools."""
        self.storage = CSVStorage(StorageConfig(data_path="data/historical/processed"))
        self.analyzer = LagAnalysisTools()
        self.results_dir = Path("results/lag_strategy")
        
        # Create validation directories
        self.validation_dir = self.results_dir / "validation"
        self.optimization_dir = self.results_dir / "optimization"
        self.backtest_dir = self.results_dir / "backtests"
        self.model_dir = self.results_dir / "trading_model"
        
        for dir_path in [self.validation_dir, self.optimization_dir, 
                        self.backtest_dir, self.model_dir]:
            dir_path.mkdir(exist_ok=True)
    
    async def validate_analysis_results(self) -> Dict:
        """
        Validate the analysis results and identify viable pairs.
        Returns:
            Dict with validation results
        """
        logger.info("🔍 Validating analysis results...")
        
        # Load summary
        summary_file = self.results_dir / "summary" / "all_pairs_summary.json"
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        validation_results = {
            'validation_date': datetime.now().isoformat(),
            'total_pairs_analyzed': summary['total_followers'],
            'successful_analyses': 0,
            'viable_pairs': [],
            'issues_found': [],
            'recommendations': []
        }
        
        # Check each follower's analysis
        for follower, follower_data in summary['summary_by_follower'].items():
            if not follower_data.get('analysis_completed', False):
                validation_results['issues_found'].append(
                    f"No data available for {follower}"
                )
                continue
            
            # Load detailed analysis
            analysis_file = self.results_dir / follower / "analysis_results.json"
            if not analysis_file.exists():
                validation_results['issues_found'].append(
                    f"Missing analysis file for {follower}"
                )
                continue
            
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
            
            validation_results['successful_analyses'] += 1
            
            # Validate each leader-follower pair
            for leader, leader_analysis in analysis['leader_analysis'].items():
                pair_name = f"{leader}-{follower}"
                
                # Check for viable signals
                significant_moves = leader_analysis['significant_moves']
                total_moves = leader_analysis['total_leader_moves']
                correlation = abs(leader_analysis['correlation'])
                
                # Calculate signal rate
                signal_rate = significant_moves / total_moves if total_moves > 0 else 0
                
                # Determine if pair is viable
                is_viable = (
                    significant_moves >= 10 and  # At least 10 significant moves
                    signal_rate >= 0.001 and     # At least 0.1% signal rate
                    correlation >= 0.01          # Some correlation
                )
                
                if is_viable:
                    validation_results['viable_pairs'].append({
                        'pair': pair_name,
                        'leader': leader,
                        'follower': follower,
                        'significant_moves': significant_moves,
                        'signal_rate': signal_rate,
                        'correlation': correlation,
                        'avg_leader_move': leader_analysis['avg_leader_move'],
                        'avg_follower_move': leader_analysis['avg_follower_move']
                    })
                else:
                    validation_results['issues_found'].append(
                        f"Low viability for {pair_name}: "
                        f"{significant_moves} moves, {signal_rate:.4f} rate, "
                        f"{correlation:.4f} correlation"
                    )
        
        # Generate recommendations
        if len(validation_results['viable_pairs']) == 0:
            validation_results['recommendations'].append(
                "No viable pairs found. Consider lowering the move threshold "
                "from 1.5% to 0.5% or 1.0% for 1-minute data."
            )
        else:
            validation_results['recommendations'].append(
                f"Found {len(validation_results['viable_pairs'])} viable pairs. "
                "Proceed with optimization and backtesting."
            )
        
        # Save validation results
        validation_file = self.validation_dir / "validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"✅ Validation complete: {len(validation_results['viable_pairs'])} viable pairs found")
        return validation_results
    
    async def optimize_thresholds(self, viable_pairs: List[Dict]) -> Dict:
        """
        Optimize thresholds for viable pairs.
        Args:
            viable_pairs: List of viable leader-follower pairs
        Returns:
            Dict with optimized parameters
        """
        logger.info("⚙️ Optimizing thresholds...")
        
        optimization_results = {
            'optimization_date': datetime.now().isoformat(),
            'pairs_optimized': len(viable_pairs),
            'best_parameters': {},
            'pair_results': {}
        }
        
        # Test different thresholds
        thresholds = [0.5, 1.0, 1.5, 2.0, 3.0]
        
        for pair_data in viable_pairs:
            leader = pair_data['leader']
            follower = pair_data['follower']
            pair_name = pair_data['pair']
            
            logger.info(f"Optimizing {pair_name}...")
            
            # Load data for this pair
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            try:
                # Load leader data
                leader_data = None
                for exchange in ['bitget', 'binance', 'kraken']:
                    for timeframe in ['1m', '5m', '5min']:
                        try:
                            leader_data = await self.storage.load_candles(
                                exchange, f"{leader}USDT", timeframe, start_date, end_date
                            )
                            if not leader_data.empty:
                                break
                        except:
                            continue
                    if leader_data is not None and not leader_data.empty:
                        break
                
                # Load follower data
                follower_data = None
                for exchange in ['bitget', 'binance', 'kraken']:
                    for timeframe in ['1m', '5m', '5min']:
                        try:
                            follower_data = await self.storage.load_candles(
                                exchange, f"{follower}USDT", timeframe, start_date, end_date
                            )
                            if not follower_data.empty:
                                break
                        except:
                            continue
                    if follower_data is not None and not follower_data.empty:
                        break
                
                if leader_data is None or follower_data is None:
                    logger.warning(f"Could not load data for {pair_name}")
                    continue
                
                # Test each threshold
                threshold_results = {}
                for threshold in thresholds:
                    # Count significant moves
                    leader_moves = (leader_data['close'] - leader_data['open']) / leader_data['open'] * 100
                    significant_moves = leader_moves[abs(leader_moves) > threshold]
                    
                    # Calculate metrics
                    signal_count = len(significant_moves)
                    signal_rate = signal_count / len(leader_moves) if len(leader_moves) > 0 else 0
                    
                    # Calculate correlation
                    follower_moves = (follower_data['close'] - follower_data['open']) / follower_data['open'] * 100
                    correlation = leader_moves.corr(follower_moves) if len(leader_moves) > 0 else 0
                    
                    threshold_results[threshold] = {
                        'signal_count': signal_count,
                        'signal_rate': signal_rate,
                        'correlation': correlation,
                        'score': signal_count * abs(correlation)  # Simple scoring
                    }
                
                # Find best threshold
                best_threshold = max(threshold_results.keys(), 
                                   key=lambda t: threshold_results[t]['score'])
                
                optimization_results['pair_results'][pair_name] = {
                    'threshold_results': threshold_results,
                    'best_threshold': best_threshold,
                    'best_score': threshold_results[best_threshold]['score']
                }
                
                optimization_results['best_parameters'][pair_name] = {
                    'threshold': best_threshold,
                    'signal_count': threshold_results[best_threshold]['signal_count'],
                    'signal_rate': threshold_results[best_threshold]['signal_rate'],
                    'correlation': threshold_results[best_threshold]['correlation']
                }
                
            except Exception as e:
                logger.error(f"Error optimizing {pair_name}: {e}")
                continue
        
        # Save optimization results
        optimization_file = self.optimization_dir / "optimization_results.json"
        with open(optimization_file, 'w') as f:
            json.dump(optimization_results, f, indent=2, default=str)
        
        logger.info(f"✅ Optimization complete for {len(optimization_results['best_parameters'])} pairs")
        return optimization_results
    
    async def run_backtests(self, optimized_pairs: Dict) -> Dict:
        """
        Run backtests for optimized pairs.
        Args:
            optimized_pairs: Dict with optimized parameters
        Returns:
            Dict with backtest results
        """
        logger.info("📊 Running backtests...")
        
        backtest_results = {
            'backtest_date': datetime.now().isoformat(),
            'pairs_backtested': len(optimized_pairs),
            'pair_results': {},
            'overall_performance': {}
        }
        
        for pair_name, params in optimized_pairs.items():
            leader, follower = pair_name.split('-')
            threshold = params['threshold']
            
            logger.info(f"Backtesting {pair_name} with threshold {threshold}%...")
            
            try:
                # Load data for backtest
                end_date = datetime.now()
                start_date = end_date - timedelta(days=60)  # 60 days for backtest
                
                # Load data
                leader_data = None
                follower_data = None
                
                for exchange in ['bitget', 'binance', 'kraken']:
                    for timeframe in ['1m', '5m', '5min']:
                        try:
                            if leader_data is None:
                                leader_data = await self.storage.load_candles(
                                    exchange, f"{leader}USDT", timeframe, start_date, end_date
                                )
                            if follower_data is None:
                                follower_data = await self.storage.load_candles(
                                    exchange, f"{follower}USDT", timeframe, start_date, end_date
                                )
                        except:
                            continue
                
                if leader_data is None or follower_data is None:
                    logger.warning(f"Could not load data for {pair_name} backtest")
                    continue
                
                # Create strategy config
                config = StrategyConfig(
                    leader_assets=[leader],
                    follower_assets=[follower],
                    threshold=threshold,
                    max_lag_minutes=60,
                    min_correlation=0.01,
                    risk_per_trade=0.02,
                    max_positions=1,
                    stop_loss_pct=0.05,
                    take_profit_pct=0.10
                )
                
                # Initialize strategy
                strategy = LagBasedStrategy(config)
                
                # Run backtest
                historical_data = {leader: leader_data, follower: follower_data}
                backtest_result = strategy.backtest(historical_data, initial_capital=10000)
                
                backtest_results['pair_results'][pair_name] = {
                    'threshold': threshold,
                    'total_return': backtest_result.get('total_return', 0),
                    'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                    'max_drawdown': backtest_result.get('max_drawdown', 0),
                    'win_rate': backtest_result.get('win_rate', 0),
                    'total_trades': backtest_result.get('total_trades', 0),
                    'profit_factor': backtest_result.get('profit_factor', 0)
                }
                
            except Exception as e:
                logger.error(f"Error backtesting {pair_name}: {e}")
                continue
        
        # Calculate overall performance
        if backtest_results['pair_results']:
            returns = [r['total_return'] for r in backtest_results['pair_results'].values()]
            backtest_results['overall_performance'] = {
                'avg_return': np.mean(returns),
                'best_return': max(returns),
                'worst_return': min(returns),
                'profitable_pairs': sum(1 for r in returns if r > 0),
                'total_pairs': len(returns)
            }
        
        # Save backtest results
        backtest_file = self.backtest_dir / "backtest_results.json"
        with open(backtest_file, 'w') as f:
            json.dump(backtest_results, f, indent=2, default=str)
        
        logger.info(f"✅ Backtest complete: {len(backtest_results['pair_results'])} pairs tested")
        return backtest_results
    
    def build_trading_model(self, backtest_results: Dict) -> Dict:
        """
        Build a production-ready trading model from backtest results.
        Args:
            backtest_results: Results from backtests
        Returns:
            Dict with trading model configuration
        """
        logger.info("🏗️ Building trading model...")
        
        # Filter profitable pairs
        profitable_pairs = {}
        for pair_name, results in backtest_results['pair_results'].items():
            if results['total_return'] > 0 and results['total_trades'] >= 5:
                profitable_pairs[pair_name] = results
        
        # Sort by performance
        sorted_pairs = sorted(profitable_pairs.items(), 
                            key=lambda x: x[1]['total_return'], reverse=True)
        
        # Build model configuration
        trading_model = {
            'model_date': datetime.now().isoformat(),
            'model_version': '1.0',
            'description': 'Lag-based trading model for meme/DeFi coins',
            'profitable_pairs': len(profitable_pairs),
            'pairs': {},
            'risk_management': {
                'max_positions': 3,
                'risk_per_trade': 0.02,
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'max_drawdown_limit': 0.20
            },
            'signal_generation': {
                'min_correlation': 0.01,
                'max_lag_minutes': 60,
                'volume_confirmation': True,
                'min_volume_ratio': 1.2
            }
        }
        
        # Add top performing pairs
        for i, (pair_name, results) in enumerate(sorted_pairs[:10]):  # Top 10
            leader, follower = pair_name.split('-')
            
            # Get optimized parameters
            optimization_file = self.optimization_dir / "optimization_results.json"
            with open(optimization_file, 'r') as f:
                optimization = json.load(f)
            
            optimized_params = optimization['best_parameters'].get(pair_name, {})
            
            trading_model['pairs'][pair_name] = {
                'leader': leader,
                'follower': follower,
                'threshold': optimized_params.get('threshold', 1.5),
                'performance': {
                    'total_return': results['total_return'],
                    'sharpe_ratio': results['sharpe_ratio'],
                    'win_rate': results['win_rate'],
                    'total_trades': results['total_trades']
                },
                'priority': i + 1  # 1 = highest priority
            }
        
        # Save trading model
        model_file = self.model_dir / "trading_model.json"
        with open(model_file, 'w') as f:
            json.dump(trading_model, f, indent=2, default=str)
        
        # Create model summary
        summary = {
            'model_summary': {
                'total_pairs': len(trading_model['pairs']),
                'avg_return': np.mean([p['performance']['total_return'] 
                                     for p in trading_model['pairs'].values()]),
                'best_pair': sorted_pairs[0][0] if sorted_pairs else None,
                'best_return': sorted_pairs[0][1]['total_return'] if sorted_pairs else 0
            }
        }
        
        summary_file = self.model_dir / "model_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"✅ Trading model built with {len(trading_model['pairs'])} profitable pairs")
        return trading_model


async def main():
    """
    Main function to validate and build the trading model.
    """
    print("🚀 Lag Strategy Validation and Trading Model Builder")
    print("=" * 60)
    
    validator = LagStrategyValidator()
    
    # Step 1: Validate analysis results
    print("\n1️⃣ Validating analysis results...")
    validation_results = await validator.validate_analysis_results()
    
    if len(validation_results['viable_pairs']) == 0:
        print("❌ No viable pairs found. Check validation results for issues.")
        return
    
    print(f"✅ Found {len(validation_results['viable_pairs'])} viable pairs")
    
    # Step 2: Optimize thresholds
    print("\n2️⃣ Optimizing thresholds...")
    optimization_results = await validator.optimize_thresholds(
        validation_results['viable_pairs']
    )
    
    if len(optimization_results['best_parameters']) == 0:
        print("❌ No pairs could be optimized. Check data availability.")
        return
    
    print(f"✅ Optimized {len(optimization_results['best_parameters'])} pairs")
    
    # Step 3: Run backtests
    print("\n3️⃣ Running backtests...")
    backtest_results = await validator.run_backtests(
        optimization_results['best_parameters']
    )
    
    if len(backtest_results['pair_results']) == 0:
        print("❌ No backtests completed successfully.")
        return
    
    print(f"✅ Completed backtests for {len(backtest_results['pair_results'])} pairs")
    
    # Step 4: Build trading model
    print("\n4️⃣ Building trading model...")
    trading_model = validator.build_trading_model(backtest_results)
    
    print(f"✅ Trading model built with {len(trading_model['pairs'])} profitable pairs")
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   Viable pairs: {len(validation_results['viable_pairs'])}")
    print(f"   Optimized pairs: {len(optimization_results['best_parameters'])}")
    print(f"   Backtested pairs: {len(backtest_results['pair_results'])}")
    print(f"   Profitable pairs: {len(trading_model['pairs'])}")
    
    if trading_model['pairs']:
        best_pair = list(trading_model['pairs'].keys())[0]
        best_return = trading_model['pairs'][best_pair]['performance']['total_return']
        print(f"   Best performing pair: {best_pair} ({best_return:.2f}% return)")
    
    print(f"\n📁 Results saved in: results/lag_strategy/")
    print(f"   Validation: {validator.validation_dir}")
    print(f"   Optimization: {validator.optimization_dir}")
    print(f"   Backtests: {validator.backtest_dir}")
    print(f"   Trading Model: {validator.model_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 