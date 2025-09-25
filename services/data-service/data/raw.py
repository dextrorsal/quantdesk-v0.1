import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import aiofiles
import aiofiles.os
import gzip

from src.core.models import StandardizedCandle
from src.core.exceptions import StorageError
from src.core.config import StorageConfig

logger = logging.getLogger(__name__)

class RawDataStorage:
    """Handles storage of raw exchange data."""
    
    def __init__(self, config: StorageConfig):
        """Initialize raw data storage with configuration."""
        self.base_path = config.historical_raw_path  # Changed from config.raw_data_path
        self.use_compression = config.use_compression
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized historical raw data storage at {self.base_path}")

    def _get_month_path(self, exchange: str, market: str, resolution: str, date_str: str) -> Path:
        """
        Generate storage folder path for a given month.
        E.g. raw_data/exchange/market/resolution/YYYY/MM/
        """
        year = date_str[0:4]
        month = date_str[5:7]
        storage_path = self.base_path / exchange / market / resolution / year / month
        storage_path.mkdir(parents=True, exist_ok=True)
        return storage_path

    async def store_candles(self, exchange: str, market: str, resolution: str, candles: List[StandardizedCandle]):
        """
        Store all raw candles for each day in a single file.
        Files are grouped into monthly folders.
        """
        # Group candles by day (YYYY-MM-DD)
        daily_groups = {}
        for candle in candles:
            date_str = candle.timestamp.strftime("%Y-%m-%d")
            daily_groups.setdefault(date_str, []).append({
                "timestamp": candle.timestamp.isoformat(),
                "market": market,
                "resolution": resolution,
                "exchange": exchange,
                "raw_data": candle.raw_data
            })
        for date_str, records in daily_groups.items():
            storage_folder = self._get_month_path(exchange, market, resolution, date_str)
            filename = f"{date_str}.raw"
            if self.use_compression:
                filename += ".gz"
            file_path = storage_folder / filename
            existing_records = []
            if file_path.exists():
                try:
                    if self.use_compression:
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            existing_records = json.load(f)
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            existing_records = json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading existing file {file_path}: {e}")
            all_records = existing_records + records
            try:
                if self.use_compression:
                    async with aiofiles.open(file_path, 'wt', encoding='utf-8', newline='') as f:
                        await f.write(json.dumps(all_records, ensure_ascii=False))
                else:
                    async with aiofiles.open(file_path, 'w', encoding='utf-8', newline='') as f:
                        await f.write(json.dumps(all_records, ensure_ascii=False))
                logger.debug(f"Stored {len(records)} raw candles into {file_path}")
            except Exception as e:
                raise StorageError(f"Failed to store raw data to {file_path}: {str(e)}")

    async def store_raw_data(self, exchange: str, market: str, data: str, timestamp: datetime, resolution: str):
        """
        Store a raw data record (CSV text) into a monthly file.
        File path: raw_data/<exchange>/<market>/<resolution>/YYYY/MM/YYYY-MM.raw[.gz]
        """
        record = {
            "timestamp": timestamp.isoformat(),
            "market": market,
            "resolution": resolution,
            "exchange": exchange,
            "raw_data": data
        }
        date_str = timestamp.strftime("%Y-%m-%d")
        storage_folder = self._get_month_path(exchange, market, resolution, date_str)
        filename = f"{timestamp.strftime('%Y-%m')}.raw"
        if self.use_compression:
            filename += ".gz"
        file_path = storage_folder / filename
        existing_records = []
        if file_path.exists():
            try:
                if self.use_compression:
                    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                        existing_records = json.load(f)
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        existing_records = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading existing file {file_path}: {e}")
        all_records = existing_records + [record]
        try:
            if self.use_compression:
                async with aiofiles.open(file_path, 'wt', encoding='utf-8', newline='') as f:
                    await f.write(json.dumps(all_records, ensure_ascii=False))
            else:
                async with aiofiles.open(file_path, 'w', encoding='utf-8', newline='') as f:
                    await f.write(json.dumps(all_records, ensure_ascii=False))
            logger.debug(f"Stored raw data record into {file_path}")
        except Exception as e:
            raise StorageError(f"Failed to store raw data to {file_path}: {str(e)}")

    async def load_candles(self, exchange: str, market: str, resolution: str, start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """
        Load raw candle data for the given time range.
        Files are stored in monthly folders.
        """
        raw_data = []
        try:
            current_date = start_time.date()
            end_date_only = end_time.date()
            while current_date <= end_date_only:
                date_str = current_date.strftime("%Y-%m-%d")
                folder = self.base_path / exchange / market / resolution / current_date.strftime("%Y") / current_date.strftime("%m")
                if folder.exists():
                    file_pattern = "*.raw.gz" if self.use_compression else "*.raw"
                    for file_path in folder.glob(file_pattern):
                        try:
                            if self.use_compression:
                                with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                                    day_records = json.load(f)
                            else:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    day_records = json.load(f)
                            for record in day_records:
                                timestamp = datetime.fromisoformat(record['timestamp'])
                                if start_time <= timestamp <= end_time:
                                    raw_data.append(record)
                        except Exception as e:
                            logger.warning(f"Error reading file {file_path}: {str(e)}")
                current_date += timedelta(days=1)
            raw_data.sort(key=lambda x: x['timestamp'])
            return raw_data
        except Exception as e:
            raise StorageError(f"Failed to load raw data: {str(e)}")

    async def delete_old_data(self, days_to_keep: int):
        """Delete raw data older than specified days."""
        try:
            # Implementation can be added here.
            pass
        except Exception as e:
            raise StorageError(f"Failed to delete old data: {str(e)}")
