"""
CSV Storage System for Quantify OHLCV Data

This module provides the CSV-based storage and retrieval system for historical OHLCV data
used throughout the Quantify project. It supports saving, loading, and organizing data
by exchange, symbol, interval, and date, and is used by all data collectors, analysis,
and strategy modules.

Features:
    - Save OHLCV data to per-day CSV files
    - Load data for specific symbols, date ranges, and intervals
    - Organize data in a clean folder structure: {exchange}/{symbol}/{interval}/{YYYY}/{MM}/{YYYY-MM-DD}.csv
    - Used by all Quantify data pipelines and strategies

Usage:
    from src.data.csv_storage import CSVStorage, StorageConfig
    storage = CSVStorage(StorageConfig(data_path='data/historical/processed'))
    storage.save_candles(exchange, symbol, interval, df)
    df = storage.load_candles(exchange, symbol, interval, start_date, end_date)
"""

import logging
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """
    Configuration for CSV data storage.
    
    Attributes:
        data_path: Base path for storing CSV files (default: "data/ohlcv")
        use_compression: Whether to compress CSV files (default: False)
        backup_enabled: Whether to enable backup functionality (default: False)
        backup_path: Path for backups (default: None)
    """
    
    data_path: Path = Path("data/ohlcv")
    use_compression: bool = False
    backup_enabled: bool = False
    backup_path: Optional[Path] = None
    
    def __post_init__(self):
        """Ensure the data path exists."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        if self.backup_enabled and self.backup_path:
            if isinstance(self.backup_path, str):
                self.backup_path = Path(self.backup_path)
            self.backup_path.mkdir(parents=True, exist_ok=True)


class CSVStorage:
    """
    CSV-based storage system for OHLCV data in Quantify.

    Handles saving and loading of per-day CSV files for each symbol, exchange, and interval.
    Used by all data collectors, analysis, and trading modules.
    """
    
    def __init__(self, config: StorageConfig):
        """
        Initialize the CSV storage system.
        Args:
            config: StorageConfig object with data_path
        """
        self.config = config
        self.base_path = config.data_path
        self.use_compression = config.use_compression
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized CSV storage at {self.base_path}")
    
    def _get_file_path(self, exchange: str, pair: str, interval: str, 
                       date: datetime) -> Path:
        """
        Generate the file path for a specific date's OHLCV data.
        Args:
            exchange: Exchange name
            pair: Trading pair
            interval: Candle interval
            date: Date for the file
        Returns:
            Path object for the CSV file
        """
        date_str = date.strftime("%Y-%m-%d")
        year = date_str[:4]
        month = date_str[5:7]
        
        # Extract base symbol from pair (e.g., "BTC" from "BTC/USDT" or "BTCUSDT")
        if '/' in pair:
            # Handle format like "BTC/USDT" -> extract "BTC"
            base_symbol = pair.split('/')[0]
        else:
            # Handle format like "BTCUSDT" -> extract "BTC" (remove quote asset)
            quote_assets = [
                'USDT', 'USD', 'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'XRP',
                'LINK', 'DOT', 'AVAX', 'MATIC', 'UNI', 'AAVE', 'CAKE',
                'ORCA', 'JUP', 'SUI', 'IMX', 'FET', 'BIGTIME', 'BOME',
                'PEPE', 'FLOKI', 'SHIB', 'WIF', 'PENGU', 'PNUT', 'TRUMP',
                'BGB', 'GRASS', 'DRIFT', 'CHILLGUY', 'PI', 'MEW', 'GOAT',
                'PONKE', 'FWOG', 'PEPECOIN', 'DODGE', 'BRETT', 'HYPE',
                'POPCAT', 'GIGA', 'AI16z', 'MOODENG', 'FARTCOIN'
            ]
            base_symbol = pair
            for quote in quote_assets:
                if pair.endswith(quote):
                    base_symbol = pair[:-len(quote)]
                    break
            # If stripping results in empty string, fall back to original pair
            if not base_symbol:
                base_symbol = pair
        
        # Create path: data/ohlcv/{exchange}/{base_symbol}/{interval}/{YYYY}/{MM}/
        # {YYYY-MM-DD}.csv
        file_path = (self.base_path / exchange / base_symbol / interval / 
                     year / month / f"{date_str}.csv")
        
        # Ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        return file_path
    
    async def store_candles(self, exchange: str, pair: str, interval: str, 
                          candles: List[Dict[str, Any]]) -> bool:
        """
        Store candlestick data in CSV format, organized by day.
        
        This method:
        1. Groups candles by date
        2. Creates/updates daily CSV files
        3. Handles deduplication and merging
        4. Maintains data integrity
        
        Args:
            exchange: Exchange name (e.g., "binance", "bitget")
            pair: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Candle interval (e.g., "1m", "5m", "1h", "1d")
            candles: List of candle dictionaries with keys: timestamp, open, high, low, close, volume
            
        Returns:
            bool: True if successful, False otherwise
            
        Example:
            candles = [
                {
                    "timestamp": "2025-01-15T10:00:00",
                    "open": 65000.0,
                    "high": 65100.0,
                    "low": 64900.0,
                    "close": 65050.0,
                    "volume": 12.345
                },
                # ... more candles
            ]
            success = await storage.store_candles("binance", "BTCUSDT", "1h", candles)
        """
        try:
            # Group candles by date
            daily_candles = {}
            
            for candle in candles:
                # Handle different timestamp formats
                if isinstance(candle['timestamp'], str):
                    timestamp = datetime.fromisoformat(candle['timestamp'].replace('Z', '+00:00'))
                elif isinstance(candle['timestamp'], datetime):
                    timestamp = candle['timestamp']
                else:
                    logger.warning(f"Invalid timestamp format: {candle['timestamp']}")
                    continue
                
                date_str = timestamp.strftime("%Y-%m-%d")
                daily_candles.setdefault(date_str, []).append(candle)
            
            # Store each day's data
            for date_str, day_candles in daily_candles.items():
                date = datetime.strptime(date_str, "%Y-%m-%d")
                file_path = self._get_file_path(exchange, pair, interval, date)
                
                # Create DataFrame for new data
                df_new = pd.DataFrame(day_candles)
                
                # Handle timestamp column
                if 'timestamp' in df_new.columns:
                    df_new['timestamp'] = pd.to_datetime(df_new['timestamp'])
                
                # If file exists, merge with existing data
                if file_path.exists():
                    try:
                        df_existing = pd.read_csv(file_path, parse_dates=['timestamp'])
                        
                        # Combine existing and new data
                        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                        
                        # Remove duplicates based on timestamp
                        df_combined.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
                        
                        # Sort by timestamp
                        df_combined.sort_values('timestamp', inplace=True)
                        
                        # Save combined data
                        df_combined.to_csv(file_path, index=False)
                        logger.debug(f"Updated {file_path} with {len(day_candles)} new candles")
                        
                    except Exception as e:
                        logger.warning(f"Error merging with existing file {file_path}: {e}")
                        # If merge fails, overwrite with new data
                        df_new.to_csv(file_path, index=False)
                        logger.info(f"Overwrote {file_path} with {len(day_candles)} candles")
                else:
                    # Create new file
                    df_new.to_csv(file_path, index=False)
                    logger.debug(f"Created {file_path} with {len(day_candles)} candles")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store candles for {exchange}/{pair}/{interval}: {e}")
            return False
    
    async def load_candles(self, exchange: str, pair: str, interval: str,
                          start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load candlestick data for a specified date range.
        
        Args:
            exchange: Exchange name (e.g., "binance", "bitget")
            pair: Trading pair (e.g., "BTCUSDT", "ETHUSDT")
            interval: Candle interval (e.g., "1m", "5m", "1h", "1d")
            start_time: Start datetime (inclusive)
            end_time: End datetime (inclusive)
            
        Returns:
            pd.DataFrame: DataFrame with columns: timestamp, open, high, low, close, volume
                         Empty DataFrame if no data found
            
        Example:
            start = datetime(2025, 1, 1)
            end = datetime(2025, 1, 31)
            df = await storage.load_candles("binance", "BTCUSDT", "1h", start, end)
            print(f"Loaded {len(df)} candles from {start.date()} to {end.date()}")
        """
        data_frames = []
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            date = datetime.combine(current_date, datetime.min.time())
            file_path = self._get_file_path(exchange, pair, interval, date)
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])
                    
                    # Filter data within the specified time range
                    if df.empty or 'timestamp' not in df.columns:
                        continue
                    # Ensure timestamps are a pandas Series of datetime64
                    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    start_time_dt = pd.to_datetime(start_time)
                    end_time_dt = pd.to_datetime(end_time)
                    mask = (df['timestamp'] >= start_time_dt) & (df['timestamp'] <= end_time_dt)
                    df_filtered = df[mask]
                    
                    if not df_filtered.empty:
                        data_frames.append(df_filtered)
                        logger.debug(f"Loaded {len(df_filtered)} candles from {file_path}")
                        
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
            else:
                logger.debug(f"No data file found for {current_date}")
            
            current_date += timedelta(days=1)
        
        if data_frames:
            # Combine all data frames
            combined_df = pd.concat(data_frames, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            combined_df.sort_values('timestamp', inplace=True)
            
            logger.info(f"Loaded {len(combined_df)} total candles for {exchange}/{pair}/{interval}")
            return combined_df
        else:
            logger.warning(f"No data found for {exchange}/{pair}/{interval} from {start_time} to {end_time}")
            return pd.DataFrame()
    
    async def load_candles_by_dates(self, exchange: str, pair: str, interval: str,
                                   dates: List[datetime]) -> pd.DataFrame:
        """
        Load candlestick data for specific dates.
        
        Args:
            exchange: Exchange name
            pair: Trading pair
            interval: Candle interval
            dates: List of specific dates to load
            
        Returns:
            pd.DataFrame: Combined data for all specified dates
        """
        data_frames = []
        
        for date in dates:
            file_path = self._get_file_path(exchange, pair, interval, date)
            
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, parse_dates=['timestamp'])
                    data_frames.append(df)
                    logger.debug(f"Loaded {len(df)} candles from {file_path}")
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
            else:
                logger.debug(f"No data file found for {date.date()}")
        
        if data_frames:
            combined_df = pd.concat(data_frames, ignore_index=True)
            combined_df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
            combined_df.sort_values('timestamp', inplace=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    async def resample_candles(self, df: pd.DataFrame, new_interval: str) -> pd.DataFrame:
        """
        Resample candlestick data to a different interval.
        
        Args:
            df: DataFrame with OHLCV data
            new_interval: Target interval (e.g., "5m", "1h", "1d")
            
        Returns:
            pd.DataFrame: Resampled data
            
        Example:
            # Load 1-minute data and resample to 1-hour
            df_1m = await storage.load_candles("binance", "BTCUSDT", "1m", start, end)
            df_1h = await storage.resample_candles(df_1m, "1h")
        """
        if df.empty:
            return df
        
        # Ensure timestamp is the index
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must have a 'timestamp' column")
        
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy.set_index('timestamp', inplace=True)
        
        # Convert interval to pandas offset
        if new_interval.endswith('D'):
            rule = new_interval
        elif new_interval.endswith('W'):
            rule = new_interval
        elif new_interval.endswith('h'):
            # Handle hour intervals (e.g., "2h" -> "2H")
            hours = new_interval[:-1]
            rule = f"{hours}H"
        else:
            # Assume minutes
            rule = f"{new_interval}min"
        
        # Resample OHLCV data
        resampled = df_copy.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(how='all')
        
        # Reset index to get timestamp column back
        resampled.reset_index(inplace=True)
        
        logger.debug(f"Resampled {len(df)} candles to {new_interval} interval, result: {len(resampled)} candles")
        return resampled
    
    async def verify_data_integrity(self, exchange: str, pair: str, interval: str,
                                   start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """
        Verify data integrity for a given time range.
        
        Args:
            exchange: Exchange name
            pair: Trading pair
            interval: Candle interval
            start_time: Start time
            end_time: End time
            
        Returns:
            Dict with verification results
        """
        df = await self.load_candles(exchange, pair, interval, start_time, end_time)
        
        if df.empty:
            return {
                "total_candles": 0,
                "duplicates": 0,
                "has_nulls": False,
                "expected_candles": 0,
                "data_integrity_ok": False,
                "missing_days": [],
                "message": "No data found"
            }
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['timestamp']).sum()
        
        # Check for null values
        has_nulls = df.isnull().any().any()
        
        # Calculate expected candles (approximate)
        expected_candles = 0
        try:
            if interval.endswith('D'):
                days = (end_time.date() - start_time.date()).days + 1
                expected_candles = days
            elif interval.isdigit():
                minutes = int((end_time - start_time).total_seconds() / 60)
                expected_candles = minutes / int(interval)
        except Exception:
            pass
        
        # Check for missing days
        missing_days = []
        current_date = start_time.date()
        end_date = end_time.date()
        
        while current_date <= end_date:
            date = datetime.combine(current_date, datetime.min.time())
            file_path = self._get_file_path(exchange, pair, interval, date)
            if not file_path.exists():
                missing_days.append(current_date.strftime("%Y-%m-%d"))
            current_date += timedelta(days=1)
        
        return {
            "total_candles": len(df),
            "duplicates": duplicates,
            "has_nulls": has_nulls,
            "expected_candles": expected_candles,
            "data_integrity_ok": duplicates == 0 and not has_nulls,
            "missing_days": missing_days,
            "date_range": {
                "start": df['timestamp'].min().isoformat() if not df.empty else None,
                "end": df['timestamp'].max().isoformat() if not df.empty else None
            }
        }
    
    def list_available_data(self, exchange: str = None, pair: str = None, 
                           interval: str = None) -> List[Dict[str, str]]:
        """
        List available data files in the storage.
        
        Args:
            exchange: Filter by exchange (optional)
            pair: Filter by pair (optional)
            interval: Filter by interval (optional)
            
        Returns:
            List of dictionaries with available data information
        """
        available_data = []
        
        # Build the search path
        search_path = self.base_path
        if exchange:
            search_path = search_path / exchange
        if pair:
            search_path = search_path / pair
        if interval:
            search_path = search_path / interval
        
        if not search_path.exists():
            return available_data
        
        # Find all CSV files
        for csv_file in search_path.rglob("*.csv"):
            try:
                # Extract information from path
                parts = csv_file.relative_to(self.base_path).parts
                if len(parts) >= 4:  # exchange/pair/interval/year/month/file.csv
                    data_info = {
                        "exchange": parts[0],
                        "pair": parts[1],
                        "interval": parts[2],
                        "year": parts[3],
                        "month": parts[4],
                        "date": csv_file.stem,  # YYYY-MM-DD
                        "file_path": str(csv_file),
                        "file_size": csv_file.stat().st_size
                    }
                    
                    # Apply filters
                    if exchange and data_info["exchange"] != exchange:
                        continue
                    if pair and data_info["pair"] != pair:
                        continue
                    if interval and data_info["interval"] != interval:
                        continue
                    
                    available_data.append(data_info)
            except Exception as e:
                logger.warning(f"Error processing file {csv_file}: {e}")
        
        return available_data


# Convenience functions for easy usage
async def store_candles_simple(exchange: str, pair: str, interval: str, 
                              candles: List[Dict[str, Any]], 
                              data_path: str = "data/ohlcv") -> bool:
    """
    Simple function to store candles without creating a storage instance.
    
    Args:
        exchange: Exchange name
        pair: Trading pair
        interval: Candle interval
        candles: List of candle dictionaries
        data_path: Base path for data storage
        
    Returns:
        bool: True if successful
    """
    config = StorageConfig(data_path=data_path)
    storage = CSVStorage(config)
    return await storage.store_candles(exchange, pair, interval, candles)


async def load_candles_simple(exchange: str, pair: str, interval: str,
                             start_time: datetime, end_time: datetime,
                             data_path: str = "data/ohlcv") -> pd.DataFrame:
    """
    Simple function to load candles without creating a storage instance.
    
    Args:
        exchange: Exchange name
        pair: Trading pair
        interval: Candle interval
        start_time: Start datetime
        end_time: End datetime
        data_path: Base path for data storage
        
    Returns:
        pd.DataFrame: Loaded data
    """
    config = StorageConfig(data_path=data_path)
    storage = CSVStorage(config)
    return await storage.load_candles(exchange, pair, interval, start_time, end_time) 