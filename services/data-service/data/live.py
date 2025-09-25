"""
Live data storage handling for Ultimate Data Fetcher.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import aiofiles
import gzip

from ..core.models import StandardizedCandle
from ..core.exceptions import StorageError
from ..core.config import StorageConfig

logger = logging.getLogger(__name__)

class LiveDataStorage:
    """Handles storage of live exchange data."""
    
    def __init__(self, config: StorageConfig):
        """Initialize live data storage with configuration."""
        self.raw_path = config.live_raw_path  # Changed from config.live_raw_data_path
        self.processed_path = config.live_processed_path  # Changed from config.live_processed_data_path
        self.use_compression = config.use_compression
        
        # Ensure paths exist
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized live data storage at {self.raw_path} (raw) and {self.processed_path} (processed)")
    
    def _get_folder_path(self, base_path: Path, exchange: str, market: str, resolution: str) -> Path:
        """
        Generate folder path for a given exchange, market, and resolution.
        Creates a structure like live/raw/exchange/market/resolution/YYYY/
        """
        current_year = datetime.now().year
        folder_path = base_path / exchange / market / resolution / str(current_year)
        folder_path.mkdir(parents=True, exist_ok=True)
        return folder_path
    
    async def store_raw_candle(self, exchange: str, market: str, resolution: str, candle: StandardizedCandle):
        """Store a single raw candle in the live data storage."""
        try:
            # Prepare the folder path
            storage_folder = self._get_folder_path(self.raw_path, exchange, market, resolution)
            
            # Generate a filename based on current date
            date_str = candle.timestamp.strftime("%Y-%m-%d")
            filename = f"{date_str}.live.raw"
            if self.use_compression:
                filename += ".gz"
            
            file_path = storage_folder / filename
            
            # Format candle data for storage
            candle_data = {
                "timestamp": candle.timestamp.isoformat(),
                "market": market,
                "resolution": resolution,
                "exchange": exchange,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
                "raw_data": candle.raw_data
            }
            
            # Attempt to load existing records, if any
            existing_records = []
            if file_path.exists():
                try:
                    if self.use_compression:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            data = f.read().strip()
                            if data:
                                existing_records = json.loads(data)
                    else:
                        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                            data = await f.read()
                            if data.strip():
                                existing_records = json.loads(data)
                    logger.debug(f"Read {len(existing_records)} existing raw records from {file_path}")
                except Exception as e:
                    logger.warning(f"Error reading existing file {file_path}: {e}")
                    existing_records = []

            # Append the new record
            all_records = existing_records + [candle_data]
            # Sort records by timestamp (ISO8601 strings sort correctly)
            all_records.sort(key=lambda x: x['timestamp'])
            
            # Write back to file
            if self.use_compression:
                with gzip.open(file_path, 'wt', encoding='utf-8') as f:
                    f.write(json.dumps(all_records, ensure_ascii=False))
            else:
                async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(all_records, ensure_ascii=False))
            
            logger.debug(f"Stored live raw candle into {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store live raw candle: {e}")
            return False

    
    async def store_processed_candle(self, exchange: str, market: str, resolution: str, candle: StandardizedCandle):
        """Store a single processed candle in CSV format."""
        try:
            # Prepare the folder path
            storage_folder = self._get_folder_path(self.processed_path, exchange, market, resolution)
            
            # Generate a filename based on current date
            date_str = candle.timestamp.strftime("%Y-%m-%d")
            file_path = storage_folder / f"{date_str}.csv"
            
            # Create a DataFrame for the candle, using the datetime object directly.
            df_new = pd.DataFrame([{
                "timestamp": candle.timestamp,  # Use the datetime object directly
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume
            }])
            
            # If file exists, append to it
            if file_path.exists():
                try:
                    df_existing = pd.read_csv(file_path, parse_dates=["timestamp"])
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    
                    # Remove duplicates based on timestamp
                    df_combined.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                    
                    # Sort by timestamp
                    df_combined.sort_values(by="timestamp", inplace=True)
                    
                    # Write back to file
                    df_combined.to_csv(file_path, index=False)
                except Exception as e:
                    logger.warning(f"Error appending to {file_path}: {e}")
                    # If error in appending, overwrite with new data
                    df_new.to_csv(file_path, index=False)
            else:
                # Create new file
                df_new.to_csv(file_path, index=False)
            
            logger.debug(f"Stored live processed candle into {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store live processed candle: {e}")
            return False
