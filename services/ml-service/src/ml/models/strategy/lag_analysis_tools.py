"""
Lag-Based Strategy Analysis Tools

This module provides comprehensive analysis tools for the lag-based trading strategy,
including data analysis, threshold optimization, correlation studies, and performance
validation.

Features:
- Move distribution analysis across timeframes
- Lag time measurement and analysis
- Correlation analysis with rolling windows
- Threshold optimization with hit rate analysis
- Volume confirmation studies
- Performance metrics calculation
- Visualization tools

Usage:
    from src.ml.models.strategy.lag_analysis_tools import LagAnalysisTools
    
    # Initialize analysis tools
    analyzer = LagAnalysisTools()
    
    # Analyze move distributions
    results = analyzer.analyze_move_distributions(leader_data, follower_data)
    
    # Optimize thresholds
    optimization = analyzer.optimize_thresholds(leader_data, follower_data)
    
    # Generate analysis report
    report = analyzer.generate_analysis_report(results, optimization)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class LagAnalysisTools:
    """
    Comprehensive analysis tools for lag-based trading strategy.
    
    Provides methods for analyzing price movements, measuring lag times,
    optimizing thresholds, and validating strategy performance.
    """
    
    def __init__(self):
        """Initialize the analysis tools."""
        self.analysis_results = {}
        self.optimization_results = {}
    
    def analyze_move_distributions(self, leader_data: Dict[str, pd.DataFrame],
                                 follower_data: Dict[str, pd.DataFrame],
                                 timeframes: List[int] = [5, 15, 30, 60]) -> Dict[str, Any]:
        """
        Analyze price movement distributions for all assets across timeframes.
        
        Args:
            leader_data: Dictionary of leader asset dataframes
            follower_data: Dictionary of follower asset dataframes
            timeframes: List of timeframes in minutes to analyze
            
        Returns:
            Dictionary with analysis results for each asset and timeframe
        """
        results = {}
        
        # Analyze leader assets
        for asset, data in leader_data.items():
            logger.info(f"Analyzing move distribution for leader asset: {asset}")
            results[asset] = self._analyze_single_asset_moves(data, asset, timeframes)
        
        # Analyze follower assets
        for asset, data in follower_data.items():
            logger.info(f"Analyzing move distribution for follower asset: {asset}")
            results[asset] = self._analyze_single_asset_moves(data, asset, timeframes)
        
        self.analysis_results = results
        return results
    
    def _analyze_single_asset_moves(self, data: pd.DataFrame, asset: str,
                                  timeframes: List[int]) -> Dict[str, Any]:
        """
        Analyze move distribution for a single asset.
        
        Args:
            data: OHLCV data for the asset
            asset: Asset name
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary with analysis results
        """
        results = {}
        
        for tf in timeframes:
            # Resample to desired timeframe
            ohlc = data.resample(f'{tf}min').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate percentage moves
            moves = (ohlc['close'] - ohlc['open']) / ohlc['open'] * 100
            abs_moves = abs(moves)
            
            # Calculate statistics
            percentiles = [25, 50, 75, 90, 95, 99]
            stats = {}
            
            for p in percentiles:
                pct_value = np.percentile(abs_moves, p)
                stats[f'{p}th_percentile'] = pct_value
            
            # Additional statistics
            stats['mean'] = abs_moves.mean()
            stats['std'] = abs_moves.std()
            stats['skewness'] = abs_moves.skew()
            stats['kurtosis'] = abs_moves.kurtosis()
            stats['total_periods'] = len(moves)
            stats['positive_moves'] = (moves > 0).sum()
            stats['negative_moves'] = (moves < 0).sum()
            
            results[f'{tf}min'] = stats
            
            logger.info(f"{asset} {tf}min - Mean: {stats['mean']:.3f}%, "
                       f"90th percentile: {stats['90th_percentile']:.3f}%, "
                       f"95th percentile: {stats['95th_percentile']:.3f}%")
        
        return results
    
    def measure_lag_times_comprehensive(self, leader_data: Dict[str, pd.DataFrame],
                                      follower_data: Dict[str, pd.DataFrame],
                                      thresholds: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0],
                                      max_lag_minutes: int = 60) -> Dict[str, Any]:
        """
        Comprehensive lag time measurement for all asset pairs.
        
        Args:
            leader_data: Dictionary of leader asset dataframes
            follower_data: Dictionary of follower asset dataframes
            thresholds: List of thresholds to test
            max_lag_minutes: Maximum lag time to consider
            
        Returns:
            Dictionary with lag analysis results
        """
        results = {}
        
        for leader_asset, leader_df in leader_data.items():
            for follower_asset, follower_df in follower_data.items():
                pair_name = f"{leader_asset}-{follower_asset}"
                logger.info(f"Measuring lag times for pair: {pair_name}")
                
                pair_results = {}
                
                for threshold in thresholds:
                    lag_times, valid_lags = self._measure_single_pair_lag(
                        leader_df, follower_df, threshold, max_lag_minutes
                    )
                    
                    if valid_lags:
                        pair_results[threshold] = {
                            'total_signals': len(lag_times),
                            'successful_signals': len(valid_lags),
                            'hit_rate': len(valid_lags) / len(lag_times),
                            'median_lag': np.median(valid_lags),
                            'mean_lag': np.mean(valid_lags),
                            'lag_std': np.std(valid_lags),
                            'min_lag': np.min(valid_lags),
                            'max_lag': np.max(valid_lags),
                            'lag_times': valid_lags
                        }
                    else:
                        pair_results[threshold] = {
                            'total_signals': len(lag_times),
                            'successful_signals': 0,
                            'hit_rate': 0,
                            'median_lag': None,
                            'mean_lag': None,
                            'lag_std': None,
                            'min_lag': None,
                            'max_lag': None,
                            'lag_times': []
                        }
                
                results[pair_name] = pair_results
        
        return results
    
    def _measure_single_pair_lag(self, leader_df: pd.DataFrame, follower_df: pd.DataFrame,
                                threshold: float, max_lag_minutes: int) -> Tuple[List[Optional[int]], List[int]]:
        """
        Measure lag times for a single leader-follower pair.
        
        Args:
            leader_df: Leader asset data
            follower_df: Follower asset data
            threshold: Move threshold
            max_lag_minutes: Maximum lag time
            
        Returns:
            Tuple of (all_lag_times, valid_lag_times)
        """
        # Resample to 5-minute bars
        leader_ohlc = leader_df.resample('5min').agg({
            'open': 'first', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        follower_ohlc = follower_df.resample('5min').agg({
            'open': 'first', 'close': 'last', 'volume': 'sum'  
        }).dropna()
        
        # Calculate moves
        leader_moves = (leader_ohlc['close'] - leader_ohlc['open']) / leader_ohlc['open'] * 100
        follower_moves = (follower_ohlc['close'] - follower_ohlc['open']) / follower_ohlc['open'] * 100
        
        lag_times = []
        
        for timestamp, move_size in leader_moves.items():
            if abs(move_size) > threshold:
                # Look for follower response
                response_found = False
                max_look_ahead = max_lag_minutes // 5
                
                for i in range(1, max_look_ahead + 1):
                    future_time = timestamp + pd.Timedelta(minutes=i*5)
                    if future_time in follower_moves.index:
                        follower_move = follower_moves[future_time]
                        # Check if follower moved in same direction with >50% of leader move
                        if (np.sign(move_size) == np.sign(follower_move) and 
                            abs(follower_move) > abs(move_size) * 0.5):
                            lag_times.append(i * 5)
                            response_found = True
                            break
                
                if not response_found:
                    lag_times.append(None)
        
        valid_lags = [x for x in lag_times if x is not None]
        return lag_times, valid_lags
    
    def analyze_correlations(self, leader_data: Dict[str, pd.DataFrame],
                           follower_data: Dict[str, pd.DataFrame],
                           windows: List[str] = ['1H', '4H', '1D', '7D']) -> Dict[str, Any]:
        """
        Analyze correlations between leader and follower assets.
        
        Args:
            leader_data: Dictionary of leader asset dataframes
            follower_data: Dictionary of follower asset dataframes
            windows: List of rolling correlation windows
            
        Returns:
            Dictionary with correlation analysis results
        """
        results = {}
        
        for leader_asset, leader_df in leader_data.items():
            for follower_asset, follower_df in follower_data.items():
                pair_name = f"{leader_asset}-{follower_asset}"
                logger.info(f"Analyzing correlations for pair: {pair_name}")
                
                pair_results = {}
                
                # Align data
                leader_returns = leader_df['close'].pct_change().dropna()
                follower_returns = follower_df['close'].pct_change().dropna()
                
                aligned_data = pd.concat([leader_returns, follower_returns], axis=1).dropna()
                
                if len(aligned_data) < 2:
                    pair_results['error'] = 'Insufficient data'
                    results[pair_name] = pair_results
                    continue
                
                # Calculate correlations for different windows
                for window in windows:
                    if len(aligned_data) > 20:  # Need enough data for rolling correlation
                        rolling_corr = aligned_data.iloc[:, 0].rolling(window=window).corr(aligned_data.iloc[:, 1])
                        
                        pair_results[window] = {
                            'mean_correlation': rolling_corr.mean(),
                            'std_correlation': rolling_corr.std(),
                            'min_correlation': rolling_corr.min(),
                            'max_correlation': rolling_corr.max(),
                            'positive_correlation_pct': (rolling_corr > 0).mean() * 100,
                            'strong_correlation_pct': (abs(rolling_corr) > 0.5).mean() * 100
                        }
                    else:
                        pair_results[window] = {'error': 'Insufficient data for rolling correlation'}
                
                # Overall correlation
                overall_corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                pair_results['overall'] = {
                    'correlation': overall_corr,
                    'data_points': len(aligned_data)
                }
                
                results[pair_name] = pair_results
        
        return results
    
    def optimize_thresholds_comprehensive(self, leader_data: Dict[str, pd.DataFrame],
                                        follower_data: Dict[str, pd.DataFrame],
                                        thresholds: List[float] = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]) -> Dict[str, Any]:
        """
        Comprehensive threshold optimization for all asset pairs.
        
        Args:
            leader_data: Dictionary of leader asset dataframes
            follower_data: Dictionary of follower asset dataframes
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with optimization results
        """
        results = {}
        
        # Get lag analysis results
        lag_results = self.measure_lag_times_comprehensive(leader_data, follower_data, thresholds)
        
        for pair_name, pair_lag_results in lag_results.items():
            logger.info(f"Optimizing thresholds for pair: {pair_name}")
            
            pair_optimization = {}
            
            for threshold in thresholds:
                if threshold in pair_lag_results:
                    lag_data = pair_lag_results[threshold]
                    
                    # Calculate signal frequency (assuming 30 days of data)
                    signal_frequency = lag_data['total_signals'] / 30  # signals per day
                    
                    pair_optimization[threshold] = {
                        'total_signals': lag_data['total_signals'],
                        'successful_signals': lag_data['successful_signals'],
                        'hit_rate': lag_data['hit_rate'],
                        'signal_frequency_per_day': signal_frequency,
                        'median_lag': lag_data['median_lag'],
                        'mean_lag': lag_data['mean_lag'],
                        'lag_std': lag_data['lag_std']
                    }
            
            results[pair_name] = pair_optimization
        
        self.optimization_results = results
        return results
    
    def generate_analysis_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            save_path: Optional path to save the report
            
        Returns:
            Report text
        """
        if not self.analysis_results and not self.optimization_results:
            return "No analysis results available. Run analysis methods first."
        
        report = []
        report.append("=" * 80)
        report.append("LAG-BASED TRADING STRATEGY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Move Distribution Analysis
        if self.analysis_results:
            report.append("MOVE DISTRIBUTION ANALYSIS")
            report.append("-" * 40)
            
            for asset, asset_results in self.analysis_results.items():
                report.append(f"\n{asset}:")
                for timeframe, stats in asset_results.items():
                    report.append(f"  {timeframe}:")
                    report.append(f"    Mean: {stats['mean']:.3f}%")
                    report.append(f"    90th percentile: {stats['90th_percentile']:.3f}%")
                    report.append(f"    95th percentile: {stats['95th_percentile']:.3f}%")
                    report.append(f"    Total periods: {stats['total_periods']}")
        
        # Threshold Optimization
        if self.optimization_results:
            report.append("\n\nTHRESHOLD OPTIMIZATION")
            report.append("-" * 40)
            
            for pair_name, pair_results in self.optimization_results.items():
                report.append(f"\n{pair_name}:")
                
                # Find best threshold based on hit rate and signal frequency
                best_threshold = None
                best_score = -1
                
                for threshold, data in pair_results.items():
                    if data['hit_rate'] > 0:
                        # Score based on hit rate and signal frequency
                        score = data['hit_rate'] * min(data['signal_frequency_per_day'], 5)  # Cap at 5 signals/day
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                
                if best_threshold:
                    best_data = pair_results[best_threshold]
                    report.append(f"  Recommended threshold: {best_threshold}%")
                    report.append(f"  Hit rate: {best_data['hit_rate']:.1%}")
                    report.append(f"  Signal frequency: {best_data['signal_frequency_per_day']:.2f} per day")
                    report.append(f"  Median lag: {best_data['median_lag']:.0f} minutes")
                
                # Show all thresholds
                report.append("  All thresholds:")
                for threshold, data in pair_results.items():
                    report.append(f"    {threshold}%: {data['hit_rate']:.1%} hit rate, "
                                f"{data['signal_frequency_per_day']:.2f} signals/day")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Analysis report saved to {save_path}")
        
        return report_text
    
    def plot_analysis_results(self, save_path: Optional[str] = None):
        """
        Plot comprehensive analysis results.
        
        Args:
            save_path: Optional path to save plots
        """
        if not self.analysis_results and not self.optimization_results:
            logger.warning("No analysis results to plot")
            return
        
        # Create subplots
        if self.analysis_results and self.optimization_results:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        elif self.analysis_results or self.optimization_results:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            axes = [axes] if len(axes.shape) == 1 else [axes[0], axes[1]]
        
        plot_idx = 0
        
        # Plot 1: Move distributions
        if self.analysis_results:
            ax = axes[plot_idx] if len(axes) > 1 else axes[0]
            
            # Plot 95th percentile moves for each asset
            assets = list(self.analysis_results.keys())
            percentiles_95 = []
            
            for asset in assets:
                if '5min' in self.analysis_results[asset]:
                    percentiles_95.append(self.analysis_results[asset]['5min']['95th_percentile'])
                else:
                    percentiles_95.append(0)
            
            ax.bar(assets, percentiles_95)
            ax.set_title('95th Percentile Moves (5min)')
            ax.set_ylabel('Move Percentage')
            ax.tick_params(axis='x', rotation=45)
            plot_idx += 1
        
        # Plot 2: Hit rates by threshold
        if self.optimization_results:
            ax = axes[plot_idx] if len(axes) > 1 else axes[1]
            
            # Get unique thresholds
            all_thresholds = set()
            for pair_results in self.optimization_results.values():
                all_thresholds.update(pair_results.keys())
            
            thresholds = sorted([float(t) for t in all_thresholds])
            
            # Calculate average hit rate for each threshold
            avg_hit_rates = []
            for threshold in thresholds:
                hit_rates = []
                for pair_results in self.optimization_results.values():
                    if str(threshold) in pair_results:
                        hit_rates.append(pair_results[str(threshold)]['hit_rate'])
                
                if hit_rates:
                    avg_hit_rates.append(np.mean(hit_rates))
                else:
                    avg_hit_rates.append(0)
            
            ax.plot(thresholds, avg_hit_rates, marker='o')
            ax.set_title('Average Hit Rate by Threshold')
            ax.set_xlabel('Threshold (%)')
            ax.set_ylabel('Hit Rate')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Analysis plots saved to {save_path}")
        
        plt.show()
    
    def save_results(self, file_path: str):
        """
        Save analysis results to JSON file.
        
        Args:
            file_path: Path to save the results
        """
        results = {
            'analysis_results': self.analysis_results,
            'optimization_results': self.optimization_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results = convert_numpy(results)
        
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Analysis results saved to {file_path}")
    
    def load_results(self, file_path: str):
        """
        Load analysis results from JSON file.
        
        Args:
            file_path: Path to load the results from
        """
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        self.analysis_results = results.get('analysis_results', {})
        self.optimization_results = results.get('optimization_results', {})
        
        logger.info(f"Analysis results loaded from {file_path}")


# Example usage
def run_comprehensive_analysis(leader_data: Dict[str, pd.DataFrame],
                             follower_data: Dict[str, pd.DataFrame],
                             output_dir: str = "analysis_results"):
    """
    Run comprehensive analysis for lag-based strategy.
    
    Args:
        leader_data: Dictionary of leader asset dataframes
        follower_data: Dictionary of follower asset dataframes
        output_dir: Directory to save results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = LagAnalysisTools()
    
    # Run analyses
    logger.info("Running move distribution analysis...")
    move_results = analyzer.analyze_move_distributions(leader_data, follower_data)
    
    logger.info("Running correlation analysis...")
    corr_results = analyzer.analyze_correlations(leader_data, follower_data)
    
    logger.info("Running threshold optimization...")
    opt_results = analyzer.optimize_thresholds_comprehensive(leader_data, follower_data)
    
    # Generate report
    logger.info("Generating analysis report...")
    report = analyzer.generate_analysis_report(f"{output_dir}/analysis_report.txt")
    
    # Save results
    analyzer.save_results(f"{output_dir}/analysis_results.json")
    
    # Create plots
    analyzer.plot_analysis_results(f"{output_dir}/analysis_plots.png")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    
    return analyzer 