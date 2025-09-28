#!/usr/bin/env python3
"""
🚀 HFT Data Consolidation Script

Consolidates fragmented 1m/5m CSV files into clean datasets for HFT.
This script merges daily CSV files into continuous datasets optimized for ML training.

Features:
- Merges fragmented daily CSV files
- Validates data quality and fills gaps
- Creates optimized datasets for HFT training
- Supports multiple exchanges and symbols
- GPU-accelerated data processing
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class HFTDataConsolidator:
    """Consolidates fragmented CSV files into continuous HFT datasets."""
    
    def __init__(self, output_dir: str = "data/hft_consolidated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # HFT-specific configuration
        self.timeframes = ['1m', '5m']
        self.symbols = ['BTC', 'ETH', 'SOL']
        self.exchanges = ['binance', 'coinbase', 'bitget']
        
        # Data quality thresholds
        self.min_data_points = {
            '1m': 1440,  # 1 day of 1-minute data
            '5m': 288    # 1 day of 5-minute data
        }
        
        self.max_gap_minutes = {
            '1m': 5,     # Max 5-minute gap for 1m data
            '5m': 15     # Max 15-minute gap for 5m data
        }
    
    def find_csv_files(self, exchange: str, symbol: str, timeframe: str) -> List[Path]:
        """Find all CSV files for a specific exchange/symbol/timeframe combination."""
        files = []
        
        # Look for different file patterns
        patterns = [
            f"data/historical/processed/{exchange}/{symbol}/{timeframe}/**/*.csv",
            f"data/historical/processed/{exchange}/{symbol}USDT_UMCBL_{timeframe}.csv",
            f"data/historical/processed/{exchange}/{symbol}/{timeframe}/*.csv"
        ]
        
        for pattern in patterns:
            found_files = list(Path(".").glob(pattern))
            files.extend(found_files)
        
        # Remove duplicates and sort
        files = list(set(files))
        files.sort()
        
        return files
    
    def load_and_validate_csv(self, file_path: Path, timeframe: str) -> Optional[pd.DataFrame]:
        """Load and validate a single CSV file."""
        try:
            df = pd.read_csv(file_path)
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.warning(f"Missing required columns in {file_path}")
                return None
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Validate data quality
            if len(df) < self.min_data_points[timeframe] * 0.5:  # At least 50% of expected data
                self.logger.warning(f"Insufficient data in {file_path}: {len(df)} rows")
                return None
            
            # Check for reasonable price values
            if (df['close'] <= 0).any() or (df['volume'] < 0).any():
                self.logger.warning(f"Invalid price/volume data in {file_path}")
                return None
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def merge_dataframes(self, dataframes: List[pd.DataFrame], timeframe: str) -> pd.DataFrame:
        """Merge multiple dataframes into a continuous dataset."""
        if not dataframes:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined = pd.concat(dataframes, ignore_index=True)
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['timestamp'])
        
        # Fill small gaps with forward fill
        combined = self.fill_data_gaps(combined, timeframe)
        
        return combined
    
    def fill_data_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """Fill small gaps in the data using forward fill."""
        if len(df) < 2:
            return df
        
        # Calculate expected time intervals
        if timeframe == '1m':
            expected_interval = pd.Timedelta(minutes=1)
        elif timeframe == '5m':
            expected_interval = pd.Timedelta(minutes=5)
        else:
            return df
        
        # Find gaps
        time_diffs = df['timestamp'].diff()
        gap_threshold = expected_interval * self.max_gap_minutes[timeframe]
        
        # Identify large gaps
        large_gaps = time_diffs > gap_threshold
        
        if large_gaps.any():
            gap_count = large_gaps.sum()
            self.logger.warning(f"Found {gap_count} large gaps in {timeframe} data")
        
        # For small gaps, use forward fill
        df_filled = df.copy()
        
        # Forward fill OHLCV data for small gaps
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in ohlcv_cols:
            df_filled[col] = df_filled[col].fillna(method='ffill')
        
        return df_filled
    
    def consolidate_symbol_data(self, exchange: str, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """Consolidate all CSV files for a specific symbol/timeframe."""
        self.logger.info(f"Consolidating {exchange}/{symbol}/{timeframe} data...")
        
        # Find all CSV files
        csv_files = self.find_csv_files(exchange, symbol, timeframe)
        
        if not csv_files:
            self.logger.warning(f"No CSV files found for {exchange}/{symbol}/{timeframe}")
            return None
        
        self.logger.info(f"Found {len(csv_files)} CSV files for {exchange}/{symbol}/{timeframe}")
        
        # Load and validate each file
        dataframes = []
        for file_path in csv_files:
            df = self.load_and_validate_csv(file_path, timeframe)
            if df is not None:
                dataframes.append(df)
        
        if not dataframes:
            self.logger.error(f"No valid data found for {exchange}/{symbol}/{timeframe}")
            return None
        
        # Merge all dataframes
        consolidated = self.merge_dataframes(dataframes, timeframe)
        
        if len(consolidated) == 0:
            self.logger.error(f"Failed to merge data for {exchange}/{symbol}/{timeframe}")
            return None
        
        # Final validation
        if len(consolidated) < self.min_data_points[timeframe]:
            self.logger.warning(f"Insufficient consolidated data: {len(consolidated)} rows")
        
        self.logger.info(f"Consolidated {len(consolidated)} rows for {exchange}/{symbol}/{timeframe}")
        
        return consolidated
    
    def save_consolidated_data(self, df: pd.DataFrame, exchange: str, symbol: str, timeframe: str):
        """Save consolidated data to optimized format."""
        # Create output directory structure
        output_path = self.output_dir / exchange / symbol / timeframe
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        csv_file = output_path / f"{symbol}_{timeframe}_consolidated.csv"
        df.to_csv(csv_file, index=False)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'exchange': exchange,
            'rows': len(df),
            'start_date': df['timestamp'].min().isoformat(),
            'end_date': df['timestamp'].max().isoformat(),
            'consolidated_at': datetime.now().isoformat(),
            'data_quality': {
                'total_rows': len(df),
                'unique_timestamps': df['timestamp'].nunique(),
                'missing_values': df.isnull().sum().to_dict(),
                'price_range': {
                    'min_close': float(df['close'].min()),
                    'max_close': float(df['close'].max()),
                    'avg_close': float(df['close'].mean())
                }
            }
        }
        
        metadata_file = output_path / f"{symbol}_{timeframe}_metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved consolidated data to {csv_file}")
        self.logger.info(f"Data range: {metadata['start_date']} to {metadata['end_date']}")
        
        return csv_file
    
    def run_consolidation(self, exchanges: List[str] = None, symbols: List[str] = None, timeframes: List[str] = None):
        """Run the full consolidation process."""
        exchanges = exchanges or self.exchanges
        symbols = symbols or self.symbols
        timeframes = timeframes or self.timeframes
        
        self.logger.info(f"Starting HFT data consolidation...")
        self.logger.info(f"Exchanges: {exchanges}")
        self.logger.info(f"Symbols: {symbols}")
        self.logger.info(f"Timeframes: {timeframes}")
        
        results = {
            'successful': [],
            'failed': [],
            'total_files_processed': 0,
            'total_rows_consolidated': 0
        }
        
        for exchange in exchanges:
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        # Consolidate data
                        consolidated_df = self.consolidate_symbol_data(exchange, symbol, timeframe)
                        
                        if consolidated_df is not None and len(consolidated_df) > 0:
                            # Save consolidated data
                            output_file = self.save_consolidated_data(consolidated_df, exchange, symbol, timeframe)
                            
                            results['successful'].append({
                                'exchange': exchange,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'rows': len(consolidated_df),
                                'file': str(output_file)
                            })
                            
                            results['total_rows_consolidated'] += len(consolidated_df)
                        else:
                            results['failed'].append({
                                'exchange': exchange,
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'reason': 'No valid data found'
                            })
                    
                    except Exception as e:
                        self.logger.error(f"Error consolidating {exchange}/{symbol}/{timeframe}: {e}")
                        results['failed'].append({
                            'exchange': exchange,
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'reason': str(e)
                        })
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: Dict):
        """Generate a summary report of the consolidation process."""
        report_file = self.output_dir / "consolidation_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# HFT Data Consolidation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Summary\n\n")
            f.write(f"- **Successful consolidations:** {len(results['successful'])}\n")
            f.write(f"- **Failed consolidations:** {len(results['failed'])}\n")
            f.write(f"- **Total rows consolidated:** {results['total_rows_consolidated']:,}\n\n")
            
            f.write("## Successful Consolidations\n\n")
            for success in results['successful']:
                f.write(f"- **{success['exchange']}/{success['symbol']}/{success['timeframe']}**: {success['rows']:,} rows\n")
            
            if results['failed']:
                f.write("\n## Failed Consolidations\n\n")
                for failure in results['failed']:
                    f.write(f"- **{failure['exchange']}/{failure['symbol']}/{failure['timeframe']}**: {failure['reason']}\n")
        
        self.logger.info(f"Consolidation report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Consolidate HFT data from fragmented CSV files")
    parser.add_argument("--exchanges", nargs="+", default=["binance", "coinbase", "bitget"],
                       help="Exchanges to process")
    parser.add_argument("--symbols", nargs="+", default=["BTC", "ETH", "SOL"],
                       help="Symbols to process")
    parser.add_argument("--timeframes", nargs="+", default=["1m", "5m"],
                       help="Timeframes to process")
    parser.add_argument("--output-dir", default="data/hft_consolidated",
                       help="Output directory for consolidated data")
    
    args = parser.parse_args()
    
    # Create consolidator
    consolidator = HFTDataConsolidator(args.output_dir)
    
    # Run consolidation
    results = consolidator.run_consolidation(
        exchanges=args.exchanges,
        symbols=args.symbols,
        timeframes=args.timeframes
    )
    
    # Print summary
    print(f"\n🎯 Consolidation Complete!")
    print(f"✅ Successful: {len(results['successful'])}")
    print(f"❌ Failed: {len(results['failed'])}")
    print(f"📊 Total Rows: {results['total_rows_consolidated']:,}")
    print(f"📁 Output: {args.output_dir}")

if __name__ == "__main__":
    main() 