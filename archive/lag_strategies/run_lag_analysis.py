#!/usr/bin/env python3
"""
Run Lag Strategy Analysis with Organized Results

This script runs the lag-based strategy analysis and saves results in an organized structure:
results/lag_strategy/
├── {FOLLOWER}/
│   ├── analysis_results.json
│   ├── backtest_results.json
│   ├── leader_analysis/
│   │   ├── BTC_analysis.json
│   │   ├── ETH_analysis.json
│   │   └── SOL_analysis.json
│   └── charts/
└── summary/
    ├── all_pairs_summary.json
    └── best_performers.json
"""

import asyncio
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the lag strategy integration
from scripts.lag_strategy_integration import LagStrategyIntegration


class LagAnalysisRunner:
    """Runs lag strategy analysis with organized results."""
    
    def __init__(self, config_path="configs/lag_strategy_config.yaml"):
        """Initialize the analysis runner."""
        self.config_path = config_path
        self.integration = LagStrategyIntegration(config_path)
        self.results_dir = Path("results/lag_strategy")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary directory
        self.summary_dir = self.results_dir / "summary"
        self.summary_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized Lag Analysis Runner")
        logger.info(f"Results will be saved to: {self.results_dir}")
    
    def create_follower_results_dir(self, follower: str) -> Path:
        """Create results directory for a specific follower."""
        follower_dir = self.results_dir / follower
        follower_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (follower_dir / "leader_analysis").mkdir(exist_ok=True)
        (follower_dir / "charts").mkdir(exist_ok=True)
        
        return follower_dir
    
    async def run_analysis_for_follower(self, follower: str, leaders: list = None):
        """Run analysis for a specific follower against all leaders."""
        if leaders is None:
            leaders = ["BTC", "ETH", "SOL"]
        
        logger.info(f"🔍 Running analysis for {follower} against leaders: {', '.join(leaders)}")
        
        # Create results directory for this follower
        follower_dir = self.create_follower_results_dir(follower)
        
        # Update config for this specific analysis
        self.integration.config['follower_assets'] = [follower]
        self.integration.config['leader_assets'] = leaders
        
        # Run analysis
        try:
            analysis_results = await self.integration.run_analysis(str(follower_dir))
            
            # Save analysis results
            analysis_file = follower_dir / "analysis_results.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            
            logger.info(f"✅ Analysis complete for {follower}")
            logger.info(f"📄 Results saved to: {analysis_file}")
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Error analyzing {follower}: {e}")
            return None
    
    async def run_backtest_for_follower(self, follower: str, start_date=None, end_date=None, capital=10000):
        """Run backtest for a specific follower."""
        logger.info(f"📊 Running backtest for {follower}")
        
        # Create results directory for this follower
        follower_dir = self.create_follower_results_dir(follower)
        
        # Update config for this specific backtest
        self.integration.config['follower_assets'] = [follower]
        
        # Set default dates if not provided
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        try:
            backtest_results = await self.integration.run_backtest(start_date, end_date, capital)
            
            # Save backtest results
            backtest_file = follower_dir / "backtest_results.json"
            with open(backtest_file, 'w') as f:
                json.dump(backtest_results, f, indent=2, default=str)
            
            logger.info(f"✅ Backtest complete for {follower}")
            logger.info(f"📄 Results saved to: {backtest_file}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Error backtesting {follower}: {e}")
            return None
    
    async def run_comprehensive_analysis(self, followers: list = None, run_backtest=True):
        """Run comprehensive analysis for all followers."""
        if followers is None:
            # Get followers from config
            followers = self.integration.config.get('follower_assets', [])
        
        logger.info(f"🚀 Starting comprehensive analysis for {len(followers)} followers")
        logger.info(f"Followers: {', '.join(followers)}")
        
        all_results = {}
        
        for follower in followers:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {follower}")
            logger.info(f"{'='*60}")
            
            # Run analysis
            analysis_result = await self.run_analysis_for_follower(follower)
            
            # Run backtest if requested
            backtest_result = None
            if run_backtest:
                backtest_result = await self.run_backtest_for_follower(follower)
            
            # Store results
            all_results[follower] = {
                'analysis': analysis_result,
                'backtest': backtest_result,
                'timestamp': datetime.now().isoformat()
            }
        
        # Create summary
        await self.create_summary(all_results)
        
        logger.info(f"\n🎉 Comprehensive analysis complete!")
        logger.info(f"📁 All results saved to: {self.results_dir}")
        
        return all_results
    
    async def create_summary(self, all_results: dict):
        """Create summary of all results."""
        logger.info("📋 Creating summary...")
        
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_followers': len(all_results),
            'followers_analyzed': list(all_results.keys()),
            'summary_by_follower': {},
            'best_performers': [],
            'overall_stats': {}
        }
        
        # Process each follower's results
        for follower, results in all_results.items():
            follower_summary = {
                'follower': follower,
                'analysis_completed': results['analysis'] is not None,
                'backtest_completed': results['backtest'] is not None,
                'timestamp': results['timestamp']
            }
            
            # Add analysis summary if available
            if results['analysis']:
                follower_summary['analysis_summary'] = {
                    'total_signals': results['analysis'].get('total_signals', 0),
                    'successful_signals': results['analysis'].get('successful_signals', 0),
                    'hit_rate': results['analysis'].get('hit_rate', 0),
                    'avg_lag_time': results['analysis'].get('avg_lag_time', 0)
                }
            
            # Add backtest summary if available
            if results['backtest']:
                follower_summary['backtest_summary'] = {
                    'total_return': results['backtest'].get('total_return', 0),
                    'sharpe_ratio': results['backtest'].get('sharpe_ratio', 0),
                    'max_drawdown': results['backtest'].get('max_drawdown', 0),
                    'win_rate': results['backtest'].get('win_rate', 0),
                    'total_trades': results['backtest'].get('total_trades', 0)
                }
            
            summary['summary_by_follower'][follower] = follower_summary
        
        # Find best performers (based on backtest results)
        performers = []
        for follower, results in all_results.items():
            if results['backtest']:
                performers.append({
                    'follower': follower,
                    'total_return': results['backtest'].get('total_return', 0),
                    'sharpe_ratio': results['backtest'].get('sharpe_ratio', 0),
                    'win_rate': results['backtest'].get('win_rate', 0)
                })
        
        # Sort by total return
        performers.sort(key=lambda x: x['total_return'], reverse=True)
        summary['best_performers'] = performers[:10]  # Top 10
        
        # Save summary
        summary_file = self.summary_dir / "all_pairs_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save best performers separately
        best_performers_file = self.summary_dir / "best_performers.json"
        with open(best_performers_file, 'w') as f:
            json.dump(performers, f, indent=2, default=str)
        
        logger.info(f"📄 Summary saved to: {summary_file}")
        logger.info(f"🏆 Best performers saved to: {best_performers_file}")
        
        # Print top performers
        if performers:
            logger.info("\n🏆 Top 5 Performers:")
            for i, performer in enumerate(performers[:5], 1):
                logger.info(f"  {i}. {performer['follower']}: {performer['total_return']:.2f}% return, {performer['sharpe_ratio']:.2f} Sharpe")
        
        return summary


async def main():
    """Main function to run the lag analysis."""
    print("🚀 Lag Strategy Analysis Runner")
    print("=" * 50)
    
    # Initialize runner
    runner = LagAnalysisRunner()
    
    # Define followers to analyze (you can modify this list)
    followers = [
        "PEPE", "WIF", "FARTCOIN", "SHIB", "FLOKI", "BOME", "PENGU", "PNUT", 
        "TRUMP", "POPCAT", "HYPE", "BRETT", "GIGA", "MEW", "GOAT", "PONKE"
    ]
    
    print(f"📈 Will analyze {len(followers)} followers")
    print(f"📁 Results will be saved to: {runner.results_dir}")
    print(f"⏰ Analysis started at: {datetime.now()}")
    print("-" * 50)
    
    # Run comprehensive analysis
    results = await runner.run_comprehensive_analysis(followers, run_backtest=True)
    
    print("\n🎉 Analysis complete!")
    print(f"📁 Check results in: {runner.results_dir}")
    print(f"📋 Summary available in: {runner.summary_dir}")


if __name__ == "__main__":
    asyncio.run(main()) 