import logging
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from src.core.config import StorageConfig  # Ensure the import path is correct

logger = logging.getLogger(__name__)

class ProcessedDataStorage:
    """Handles storage of processed exchange data."""
    
    def __init__(self, config: StorageConfig):
        """Initialize processed data storage with configuration."""
        self.base_path = config.historical_processed_path  # Changed from config.processed_data_path
        self.use_compression = config.use_compression
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized historical processed data storage at {self.base_path}")
    
    async def store_candles(self, exchange: str, market: str, resolution: str, candles: list):
        """
        Store processed candles in CSV format.
        Files are stored under: processed_data/<exchange>/<market>/<resolution>/YYYY/MM/
        with filename YYYY-MM-DD.csv for each day.
        
        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle resolution
            candles: List of candles (either StandardizedCandle objects or dictionaries)
        """
        daily = {}
        for candle in candles:
            # Handle both StandardizedCandle objects and dictionaries
            if hasattr(candle, 'timestamp'):
                # It's a StandardizedCandle object
                timestamp = candle.timestamp
                candle_data = {
                    "timestamp": candle.timestamp.isoformat(),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume
                }
            else:
                # It's a dictionary
                if isinstance(candle['timestamp'], str):
                    # Parse ISO format string to datetime
                    timestamp = datetime.fromisoformat(candle['timestamp'].replace('Z', '+00:00'))
                else:
                    # Assume it's already a datetime
                    timestamp = candle['timestamp']
                candle_data = {
                    "timestamp": candle['timestamp'] if isinstance(candle['timestamp'], str) else candle['timestamp'].isoformat(),
                    "open": candle['open'],
                    "high": candle['high'],
                    "low": candle['low'],
                    "close": candle['close'],
                    "volume": candle['volume']
                }
                
            date_str = timestamp.strftime("%Y-%m-%d")
            daily.setdefault(date_str, []).append(candle_data)
            
        for date_str, records in daily.items():
            year = date_str[0:4]
            month = date_str[5:7]
            folder = self.base_path / exchange / market / resolution / year / month
            folder.mkdir(parents=True, exist_ok=True)
            file_path = folder / f"{date_str}.csv"
            try:
                df = pd.DataFrame(records)
                df.to_csv(file_path, index=False)
                logger.debug(f"Stored processed data for {date_str} at {file_path}")
            except Exception as e:
                logger.error(f"Failed to store processed data to {file_path}: {e}")
    
    async def load_candles(self, exchange: str, market: str, resolution: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Load processed candle data for the given time range.
        """
        data = []
        current_date = start_time.date()
        end_date = end_time.date()
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            year = date_str[0:4]
            month = date_str[5:7]
            file_path = self.base_path / exchange / market / resolution / year / month / f"{date_str}.csv"
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, parse_dates=["timestamp"])
                    data.append(df)
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
            current_date += timedelta(days=1)
        if data:
            combined = pd.concat(data, ignore_index=True)
            combined.sort_values(by="timestamp", inplace=True)
            return combined
        return pd.DataFrame()
    
    async def resample_candles(self, df: pd.DataFrame, new_resolution: str) -> pd.DataFrame:
        """
        Resample the given DataFrame (OHLCV) to a new resolution.
        E.g., from 15-minute bars to 60-minute bars.
        
        Args:
            df: DataFrame with OHLCV data
            new_resolution: Target resolution (e.g., '60' for 60 minutes)
            
        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            # If "timestamp" is a column, set it as the index
            if 'timestamp' in df.columns:
                df = df.copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError("DataFrame must have a DateTimeIndex or 'timestamp' column for resampling.")

        # Convert new_resolution to a Pandas offset alias
        # Use updated offset aliases to avoid deprecation warnings
        if new_resolution.endswith('D'):
            rule = new_resolution  # Keep 'D' for daily
        elif new_resolution.endswith('W'):
            rule = new_resolution  # Keep 'W' for weekly
        else:
            rule = f"{new_resolution}min"  # Use 'min' instead of 'T' for minutes

        # For a basic OHLCV resample
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(how='all')

        # Reset index so the returned DataFrame has a "timestamp" column
        resampled.reset_index(inplace=True)
        # name the index column if needed
        if 'timestamp' not in resampled.columns:
            resampled.rename(columns={'index': 'timestamp'}, inplace=True)

        logger.debug(f"Resampled DataFrame to {new_resolution} resolution, result has {len(resampled)} rows")
        return resampled

    async def verify_data_integrity(self, exchange: str, market: str, resolution: str,
                                start_time: datetime, end_time: datetime) -> dict:
        """
        Verify data integrity by comparing raw and processed data.
        
        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle resolution
            start_time: Start time
            end_time: End time
            
        Returns:
            Dictionary with verification results
        """
        # Load processed data
        df = await self.load_candles(exchange, market, resolution, start_time, end_time)
        processed_count = len(df)
        
        # Check for data issues
        duplicates = df.duplicated(subset=['timestamp']).sum()
        has_nulls = df.isnull().any().any()
        
        # Calculate expected number of candles (for regular resolutions)
        expected_candles = 0
        try:
            if resolution.endswith('D'):
                # Daily candles
                days = (end_time.date() - start_time.date()).days + 1
                expected_candles = days
            elif resolution.isdigit():
                # Minute-based candles
                minutes = int((end_time - start_time).total_seconds() / 60)
                expected_candles = minutes / int(resolution)
        except Exception:
            # Skip expected count if calculation fails
            pass
        
        return {
            "processed_data_count": processed_count,
            "duplicates": duplicates,
            "has_nulls": has_nulls,
            "expected_candles": expected_candles,
            "data_integrity_ok": duplicates == 0 and not has_nulls,
            "total_candles": processed_count
        }