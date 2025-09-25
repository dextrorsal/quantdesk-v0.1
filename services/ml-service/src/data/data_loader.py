"""
Data Loader for Candlestick Data

This module provides a comprehensive data loading system that can load candlestick data
from CSV files with flexible date ranges, multiple exchanges, and pairs.

USE CASES:
- **CSV data loading**: Load data from organized CSV storage system
- **Multiple exchanges/pairs**: Load data from multiple sources simultaneously
- **Flexible date ranges**: Load data for specific time periods
- **Data validation**: Built-in data integrity checks
- **Batch loading**: Load multiple pairs/exchanges at once
- **Data exploration**: Discover available data and date ranges
- **Model training**: Load data for ML model training from CSV files

DIFFERENCES FROM OTHER LOADERS/FETCHERS:
- DataLoader: CSV-based data loading with file system storage
- DatabaseLoader: Simple database loading interface (like pd.read_csv)
- UltimateDataFetcher: Multi-exchange orchestrator with database storage
- ReliableDataFetcher: CCXT-based, fast exchanges only, database storage
- DailyDataFetcher: Automated daily CSV storage with multiple exchanges
- CSVDataFetcher: Manual CSV storage, no database integration

WHEN TO USE:
- When you have data stored in CSV files
- For loading data from the organized CSV storage system
- When you need flexible date ranges and multiple sources
- For data validation and integrity checks
- When you don't need database integration

FEATURES:
- Load data for specific date ranges
- Load data for multiple exchanges and pairs simultaneously
- Support for different timeframes
- Data validation and integrity checks
- Efficient loading with progress tracking
- Support for both single and batch loading

Usage:
    from src.data.data_loader import DataLoader
    
    # Initialize loader
    loader = DataLoader("data/ohlcv")
    
    # Load specific data
    df = loader.load_data(
        exchange="binance",
        pair="BTC/USDT", 
        interval="1h",
        start_date="2025-01-01",
        end_date="2025-01-31"
    )
    
    # Load multiple pairs
    data = loader.load_multiple_pairs(
        exchanges=["binance", "bitget"],
        pairs=["BTC/USDT", "ETH/USDT"],
        interval="1h",
        start_date="2025-01-01",
        end_date="2025-01-31"
    )
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.csv_storage import CSVStorage, StorageConfig


class DataLoader:
    """
    Data loader for candlestick data stored in CSV format.
    
    This class provides methods to load candlestick data from the CSV storage
    system with various filtering and aggregation options.
    """
    
    def __init__(self, data_path: str = "data/ohlcv"):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV data directory
        """
        self.data_path = Path(data_path)
        self.storage = CSVStorage(StorageConfig(data_path=data_path))
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def get_available_data(self) -> Dict[str, Dict]:
        """
        Get information about available data.
        
        Returns:
            Dictionary with available exchanges, pairs, and timeframes
        """
        available = {}
        
        if not self.data_path.exists():
            return available
            
        for exchange_dir in self.data_path.iterdir():
            if not exchange_dir.is_dir():
                continue
                
            exchange = exchange_dir.name
            available[exchange] = {}
            
            for pair_dir in exchange_dir.iterdir():
                if not pair_dir.is_dir():
                    continue
                    
                pair = pair_dir.name
                available[exchange][pair] = {}
                
                for interval_dir in pair_dir.iterdir():
                    if not interval_dir.is_dir():
                        continue
                        
                    interval = interval_dir.name
                    csv_files = list(interval_dir.rglob("*.csv"))
                    
                    if csv_files:
                        # Get date range
                        dates = []
                        for csv_file in csv_files:
                            try:
                                date_str = csv_file.stem  # filename without extension
                                date = datetime.strptime(date_str, "%Y-%m-%d")
                                dates.append(date)
                            except ValueError:
                                continue
                        
                        if dates:
                            min_date = min(dates)
                            max_date = max(dates)
                            available[exchange][pair][interval] = {
                                "start_date": min_date.strftime("%Y-%m-%d"),
                                "end_date": max_date.strftime("%Y-%m-%d"),
                                "file_count": len(csv_files),
                                "total_candles": sum(
                                    len(pd.read_csv(f)) for f in csv_files
                                )
                            }
        
        return available
    
    async def load_data(
        self,
        exchange: str,
        pair: str,
        interval: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load candlestick data for a specific exchange, pair, and interval.
        
        Args:
            exchange: Exchange name (e.g., "binance", "bitget")
            pair: Trading pair (e.g., "BTC/USDT")
            interval: Timeframe (e.g., "1h", "1d")
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime)
            validate: Whether to validate data integrity
            
        Returns:
            DataFrame with candlestick data
        """
        try:
            # Convert dates if needed
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
            # Load data using storage
            df = await self.storage.load_candles(
                exchange=exchange,
                pair=pair,
                interval=interval,
                start_time=start_date,
                end_time=end_date
            )
            
            if df.empty:
                self.logger.warning(
                    f"No data found for {exchange}/{pair}/{interval} "
                    f"from {start_date} to {end_date}"
                )
                return df
            
            # Add metadata columns
            df['exchange'] = exchange
            df['pair'] = pair
            df['interval'] = interval
            
            # Validate data if requested
            if validate:
                self._validate_data(df, exchange, pair, interval)
            
            self.logger.info(
                f"Loaded {len(df)} candles for {exchange}/{pair}/{interval} "
                f"from {df.index.min()} to {df.index.max()}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(
                f"Error loading data for {exchange}/{pair}/{interval}: {e}"
            )
            return pd.DataFrame()
    
    async def load_multiple_pairs(
        self,
        exchanges: List[str],
        pairs: List[str],
        interval: str,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple exchanges and pairs.
        
        Args:
            exchanges: List of exchange names
            pairs: List of trading pairs
            interval: Timeframe
            start_date: Start date
            end_date: End date
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with DataFrames for each exchange/pair combination
        """
        results = {}
        
        # Create combinations
        combinations = []
        for exchange in exchanges:
            for pair in pairs:
                combinations.append((exchange, pair))
        
        # Load data with progress bar
        iterator = tqdm(combinations, desc="Loading data") if show_progress else combinations
        
        for exchange, pair in iterator:
            df = await self.load_data(
                exchange=exchange,
                pair=pair,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                validate=False  # Skip validation for batch loading
            )
            
            if not df.empty:
                key = f"{exchange}_{pair}"
                results[key] = df
        
        self.logger.info(f"Loaded data for {len(results)} exchange/pair combinations")
        return results
    
    async def load_recent_data(
        self,
        exchange: str,
        pair: str,
        interval: str,
        days: int = 30
    ) -> pd.DataFrame:
        """
        Load recent data for the last N days.
        
        Args:
            exchange: Exchange name
            pair: Trading pair
            interval: Timeframe
            days: Number of days to load
            
        Returns:
            DataFrame with recent data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return await self.load_data(
            exchange=exchange,
            pair=pair,
            interval=interval,
            start_date=start_date,
            end_date=end_date
        )
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get a summary of all available data.
        
        Returns:
            DataFrame with summary information
        """
        available = self.get_available_data()
        
        summary_data = []
        for exchange, pairs in available.items():
            for pair, intervals in pairs.items():
                for interval, info in intervals.items():
                    summary_data.append({
                        'exchange': exchange,
                        'pair': pair,
                        'interval': interval,
                        'start_date': info['start_date'],
                        'end_date': info['end_date'],
                        'file_count': info['file_count'],
                        'total_candles': info['total_candles']
                    })
        
        return pd.DataFrame(summary_data)
    
    def _validate_data(
        self, 
        df: pd.DataFrame, 
        exchange: str, 
        pair: str, 
        interval: str
    ) -> None:
        """
        Validate data integrity.
        
        Args:
            df: DataFrame to validate
            exchange: Exchange name
            pair: Trading pair
            interval: Timeframe
        """
        if df.empty:
            return
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(
                f"Missing columns in {exchange}/{pair}/{interval}: {missing_cols}"
            )
        
        # Check for negative values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns and (df[col] < 0).any():
                self.logger.warning(
                    f"Negative values found in {col} for {exchange}/{pair}/{interval}"
                )
        
        # Check for missing values
        for col in df.columns:
            if df[col].isnull().any():
                self.logger.warning(
                    f"Missing values found in {col} for {exchange}/{pair}/{interval}"
                )
        
        # Check OHLC relationship
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).any()
            invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).any()
            
            if invalid_high or invalid_low:
                self.logger.warning(
                    f"Invalid OHLC relationship in {exchange}/{pair}/{interval}"
                )


# Convenience functions
async def load_data(
    exchange: str,
    pair: str,
    interval: str,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    data_path: str = "data/ohlcv"
) -> pd.DataFrame:
    """
    Convenience function to load data quickly.
    
    Args:
        exchange: Exchange name
        pair: Trading pair
        interval: Timeframe
        start_date: Start date
        end_date: End date
        data_path: Path to data directory
        
    Returns:
        DataFrame with candlestick data
    """
    loader = DataLoader(data_path)
    return await loader.load_data(exchange, pair, interval, start_date, end_date)


def get_available_data(data_path: str = "data/ohlcv") -> Dict[str, Dict]:
    """
    Convenience function to get available data.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        Dictionary with available data information
    """
    loader = DataLoader(data_path)
    return loader.get_available_data() 