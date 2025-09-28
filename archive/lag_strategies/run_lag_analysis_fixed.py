#!/usr/bin/env python3
"""
Lag-Based Strategy Analysis Runner (Fixed Version)

This script is part of the QuantDesk lag-based trading strategy research suite.

Purpose:
    - Runs the lag-based analysis for meme/DeFi coins and major leaders (BTC, ETH, SOL)
    - Loads historical OHLCV data from your CSV storage system
    - Analyzes lag relationships between leaders and followers
    - Saves results in a per-symbol, per-leader organized folder structure
    - Produces summary files for easy review

How to Use:
    $ python scripts/run_lag_analysis_fixed.py

    - Results are saved in results/lag_strategy/{SYMBOL}/
    - Each follower (e.g., PEPE) gets its own folder with JSON results
    - See results/lag_strategy/summary/all_pairs_summary.json for a summary

What is the Lag Strategy?
    The lag-based strategy looks for price moves in major assets (BTC, ETH, SOL) that are followed, with a delay, by meme/DeFi coins. It quantifies the lag, correlation, and response rate, helping you identify which coins reliably follow leaders and with what delay.

Typical Workflow:
    1. Fetch and clean your data (already done)
    2. Run this script to analyze lag relationships
    3. Review results in the results/lag_strategy/ folders
    4. Use findings to inform backtesting or live trading

See also: docs/lag-based.md for a full strategy guide.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the lag strategy components
from data.csv_storage import CSVStorage, StorageConfig
from ml.models.strategy.lag_analysis_tools import LagAnalysisTools


class LagAnalysisRunner:
    """
    Runs lag strategy analysis for all configured followers and leaders.

    - Loads OHLCV data for each follower and leader from CSV storage
    - Computes lag/correlation stats for each leader-follower pair
    - Saves results in organized folders for easy review
    - Produces a summary JSON for all pairs
    """
    
    def __init__(self):
        """
        Initialize the analysis runner, set up results directories, and tools.
        """
        self.results_dir = Path("results/lag_strategy")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary directory
        self.summary_dir = self.results_dir / "summary"
        self.summary_dir.mkdir(exist_ok=True)
        
        # Initialize storage and tools
        self.storage = CSVStorage(StorageConfig(data_path="data/historical/processed"))
        self.analyzer = LagAnalysisTools()
        
        logger.info(f"Initialized Lag Analysis Runner")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def create_follower_results_dir(self, follower: str) -> Path:
        """
        Create results directory for a specific follower symbol.
        Args:
            follower: The follower asset symbol (e.g., 'PEPE')
        Returns:
            Path to the follower's results directory
        """
        follower_dir = self.results_dir / follower
        follower_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (follower_dir / "leader_analysis").mkdir(exist_ok=True)
        (follower_dir / "charts").mkdir(exist_ok=True)
        
        return follower_dir
    
    async def run_analysis_for_follower(self, follower: str, leaders: list = None):
        """
        Run lag analysis for a single follower against all leaders.
        Loads data, computes stats, and saves results.
        Args:
            follower: The follower asset symbol
            leaders: List of leader asset symbols (default: BTC, ETH, SOL)
        Returns:
            Analysis results dict, or None if data missing
        """
        if leaders is None:
            leaders = ["BTC", "ETH", "SOL"]
        
        logger.info(f"🔍 Running analysis for {follower} against leaders: {', '.join(leaders)}")
        
        # Create results directory for this follower
        follower_dir = self.create_follower_results_dir(follower)
        
        # Calculate date range (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        try:
            # Load data for this follower
            follower_data = None
            for exchange in ['bitget', 'binance', 'kraken']:
                # Try different timeframe formats
                for timeframe in ['1m', '5m', '5min']:
                    try:
                        follower_data = await self.storage.load_candles(
                            exchange, f"{follower}USDT", timeframe, start_date, end_date
                        )
                        if not follower_data.empty:
                            logger.info(f"Loaded {follower} data from {exchange} ({timeframe}): {len(follower_data)} records")
                            break
                    except Exception as e:
                        logger.debug(f"Failed to load {follower} from {exchange} ({timeframe}): {e}")
                        continue
                
                if follower_data is not None and not follower_data.empty:
                    break
            
            if follower_data is None or follower_data.empty:
                logger.warning(f"Could not load data for {follower}")
                return None
            
            # Load leader data
            leader_data = {}
            for leader in leaders:
                for exchange in ['bitget', 'binance', 'kraken']:
                    # Try different timeframe formats
                    for timeframe in ['1m', '5m', '5min']:
                        try:
                            data = await self.storage.load_candles(
                                exchange, f"{leader}USDT", timeframe, start_date, end_date
                            )
                            if not data.empty:
                                leader_data[leader] = data
                                logger.info(f"Loaded {leader} data from {exchange} ({timeframe}): {len(data)} records")
                                break
                        except Exception as e:
                            logger.debug(f"Failed to load {leader} from {exchange} ({timeframe}): {e}")
                            continue
                    
                    if leader in leader_data:
                        break
            
            if not leader_data:
                logger.warning(f"Could not load any leader data for {follower}")
                return None
            
            # Run analysis
            analysis_results = {
                'follower': follower,
                'leaders': list(leader_data.keys()),
                'analysis_date': datetime.now().isoformat(),
                'data_period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': 30
                },
                'data_summary': {
                    'follower_records': len(follower_data),
                    'leader_records': {leader: len(data) for leader, data in leader_data.items()}
                },
                'leader_analysis': {}
            }
            
            # Analyze each leader-follower pair
            for leader, leader_df in leader_data.items():
                logger.info(f"Analyzing {leader} -> {follower}")
                
                # Calculate basic statistics
                leader_moves = (leader_df['close'] - leader_df['open']) / leader_df['open'] * 100
                follower_moves = (follower_data['close'] - follower_data['open']) / follower_data['open'] * 100
                
                # Find significant moves (above 1.5% threshold)
                significant_moves = leader_moves[abs(leader_moves) > 1.5]
                
                leader_analysis = {
                    'total_leader_moves': len(leader_moves),
                    'significant_moves': len(significant_moves),
                    'move_threshold': 1.5,
                    'avg_leader_move': leader_moves.mean(),
                    'avg_follower_move': follower_moves.mean(),
                    'leader_volatility': leader_moves.std(),
                    'follower_volatility': follower_moves.std(),
                    'correlation': leader_moves.corr(follower_moves) if len(leader_moves) > 0 else 0
                }
                
                # Save individual leader analysis
                leader_file = follower_dir / "leader_analysis" / f"{leader}_analysis.json"
                with open(leader_file, 'w') as f:
                    json.dump(leader_analysis, f, indent=2, default=str)
                
                analysis_results['leader_analysis'][leader] = leader_analysis
            
            # Save overall analysis results
            analysis_file = follower_dir / "analysis_results.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"✅ Analysis complete for {follower}")
            logger.info(f"📄 Results saved to: {analysis_file}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {follower}: {e}")
            return None
    
    async def run_comprehensive_analysis(self, followers: list = None):
        """
        Run lag analysis for all followers in the configured list.
        Args:
            followers: List of follower asset symbols (default: preset list)
        Returns:
            Dict of all results
        """
        if followers is None:
            followers = [
                "PEPE", "WIF", "FARTCOIN", "SHIB", "FLOKI", "BOME", "PENGU", "PNUT", 
                "TRUMP", "POPCAT", "HYPE", "BRETT", "GIGA", "MEW", "GOAT", "PONKE"
            ]
        
        logger.info(f"🚀 Starting comprehensive analysis for {len(followers)} followers")
        logger.info(f"Followers: {', '.join(followers)}")
        
        all_results = {}
        
        for follower in followers:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {follower}")
            logger.info(f"{'='*60}")
            
            # Run analysis
            analysis_result = await self.run_analysis_for_follower(follower)
            
            # Store results
            all_results[follower] = {
                'analysis': analysis_result,
                'timestamp': datetime.now().isoformat()
            }
        
        # Create summary
        await self.create_summary(all_results)
        
        logger.info(f"\n🎉 Comprehensive analysis complete!")
        logger.info(f"📁 All results saved to: {self.results_dir}")
        
        return all_results
    
    async def create_summary(self, all_results: dict):
        """
        Create a summary JSON file for all analyzed pairs.
        Args:
            all_results: Dict of all follower analysis results
        Returns:
            Summary dict
        """
        logger.info("📋 Creating summary...")
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_followers': len(all_results),
            'followers_analyzed': list(all_results.keys()),
            'summary_by_follower': {},
            'overall_stats': {}
        }
        
        # Process each follower's results
        for follower, results in all_results.items():
            follower_summary = {
                'follower': follower,
                'analysis_completed': results['analysis'] is not None,
                'timestamp': results['timestamp']
            }
            
            # Add analysis summary if available
            if results['analysis']:
                follower_summary['analysis_summary'] = {
                    'leaders_analyzed': len(results['analysis'].get('leader_analysis', {})),
                    'data_records': results['analysis'].get('data_summary', {}).get('follower_records', 0)
                }
            
            summary['summary_by_follower'][follower] = follower_summary
        
        # Save summary
        summary_file = self.summary_dir / "all_pairs_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        
        # Print summary
        logger.info(f"\n📊 Analysis Summary:")
        logger.info(f"Total followers analyzed: {len(all_results)}")
        successful_analyses = sum(1 for r in all_results.values() if r['analysis'] is not None)
        logger.info(f"Successful analyses: {successful_analyses}")
        
        return summary


async def main():
    """
    Main entry point for running the lag-based analysis.
    Usage:
        $ python scripts/run_lag_analysis_fixed.py
    """
    print("🚀 Lag Strategy Analysis Runner (Fixed)")
    print("=" * 50)
    
    # Initialize runner
    runner = LagAnalysisRunner()
    
    # Define followers to analyze
    followers = [
        "PEPE", "WIF", "FARTCOIN", "SHIB", "FLOKI", "BOME", "PENGU", "PNUT", 
        "TRUMP", "POPCAT", "HYPE", "BRETT", "GIGA", "MEW", "GOAT", "PONKE"
    ]
    
    print(f"📈 Will analyze {len(followers)} followers")
    print(f"📁 Results will be saved to: {runner.results_dir}")
    print(f"⏰ Analysis started at: {datetime.now()}")
    print("-" * 50)
    
    # Run comprehensive analysis
    results = await runner.run_comprehensive_analysis(followers)
    
    print("\n🎉 Analysis complete!")
    print(f"📁 Check results in: {runner.results_dir}")
    print(f"📋 Summary available in: {runner.summary_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 