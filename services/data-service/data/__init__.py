"""
Storage module initialization.
Provides unified access to raw and processed data storage capabilities.
"""

from typing import Dict, Type
from enum import Enum
from datetime import datetime

from src.storage.raw import RawDataStorage
from src.storage.processed import ProcessedDataStorage
from src.core.config import StorageConfig
import logging
from src.data.providers.supabase_storage import SupabaseDataStorage

logger = logging.getLogger(__name__)

# Export main storage classes
__all__ = ["RawDataStorage", "ProcessedDataStorage", "get_storage", "StorageType"]


# Define storage types
class StorageType(Enum):
    RAW = "raw"
    PROCESSED = "processed"
    TFRECORD = "tfrecord"  # Keep enum but make optional


# Storage registry
STORAGE_HANDLERS: Dict[str, Type] = {
    StorageType.RAW: RawDataStorage,
    StorageType.PROCESSED: ProcessedDataStorage,
}

# Optionally add TFRecord support
try:
    from src.storage.tfrecord_storage import TFRecordStorage

    STORAGE_HANDLERS[StorageType.TFRECORD] = TFRecordStorage
    HAS_TFRECORD = True
except ImportError:
    HAS_TFRECORD = False
    logger.info("TFRecord storage not available - ML features will be disabled")


def get_storage(storage_type: str, config: StorageConfig):
    """
    Factory function to get appropriate storage handler.

    Args:
        storage_type: Type of storage ("raw" or "processed" or "tfrecord")
        config: Storage configuration

    Returns:
        Initialized storage handler

    Raises:
        ValueError: If storage type is not supported
    """
    storage_class = STORAGE_HANDLERS.get(storage_type.lower())
    if not storage_class:
        raise ValueError(f"Unsupported storage type: {storage_type}")

    return storage_class(config)


class DataManager:
    """Manages raw, processed, and Supabase data storage."""

    def __init__(
        self,
        config: StorageConfig,
        supabase_url: str = None,
        supabase_key: str = None,
        storage_backend: str = "supabase",
    ):
        """Initialize data manager with configuration and backend selection."""
        self.config = config
        # If supabase_url/key are not provided, default to csv backend for tests
        if storage_backend.lower() == "supabase" and not (
            supabase_url and supabase_key
        ):
            self.storage_backend = "csv"
        else:
            self.storage_backend = storage_backend.lower()
        self.raw_storage = RawDataStorage(config)
        self.processed_storage = ProcessedDataStorage(config)
        self.supabase_storage = None
        if self.storage_backend == "supabase":
            self.supabase_storage = SupabaseDataStorage(supabase_url, supabase_key)
        self.tfrecord_storage = None
        if HAS_TFRECORD:
            self.tfrecord_storage = TFRecordStorage(config)

    async def store_data(self, data, exchange: str, market: str, resolution: str):
        """Store data using the selected backend."""
        if self.storage_backend == "supabase":
            await self.supabase_storage.store_candles(
                exchange, market, resolution, data
            )
        else:
            # Store raw data
            await self.raw_storage.store_candles(
                exchange=exchange, market=market, resolution=resolution, candles=data
            )
            # Store processed data
            await self.processed_storage.store_candles(
                exchange=exchange, market=market, resolution=resolution, candles=data
            )

    async def load_data(
        self,
        exchange: str,
        market: str,
        resolution: str,
        start_time,
        end_time,
        format_type: str = StorageType.PROCESSED,
    ):
        """
        Load data using the selected backend.
        """
        if self.storage_backend == "supabase":
            return await self.supabase_storage.load_candles(
                exchange, market, resolution, start_time, end_time
            )
        elif format_type == StorageType.RAW:
            return await self.raw_storage.load_candles(
                exchange, market, resolution, start_time, end_time
            )
        else:
            return await self.processed_storage.load_candles(
                exchange, market, resolution, start_time, end_time
            )

    async def verify_data(
        self, exchange: str, market: str, resolution: str, start_time, end_time
    ) -> Dict:
        """Verify data integrity in both storage types."""
        # Get verification results from processed storage
        processed_results = await self.processed_storage.verify_data_integrity(
            exchange, market, resolution, start_time, end_time
        )

        # Load raw data for comparison
        raw_data = await self.raw_storage.load_candles(
            exchange, market, resolution, start_time, end_time
        )

        # Compare data counts
        raw_count = len(raw_data) if raw_data else 0
        processed_count = processed_results.get("total_candles", 0)

        return {
            "raw_data_count": raw_count,
            "processed_data_count": processed_count,
            "data_matches": raw_count == processed_count,
            "processed_verification": processed_results,
        }

    async def backup_all_data(self, backup_path):
        """Create backup of both raw and processed data."""
        try:
            await self.raw_storage.backup_data(backup_path / "raw")
            await self.processed_storage.backup_data(backup_path / "processed")
            return True
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False

    # ML-related methods that require TFRecord
    def _check_tfrecord(self):
        if not self.tfrecord_storage:
            raise NotImplementedError(
                "TFRecord storage not available - install tensorflow to enable ML features"
            )

    async def convert_to_tfrecord(
        self,
        exchange: str,
        market: str,
        resolution: str,
        start_time: datetime,
        end_time: datetime,
    ):
        """Convert processed data to TFRecord format (requires tensorflow)."""
        self._check_tfrecord()
        df = await self.processed_storage.load_candles(
            exchange, market, resolution, start_time, end_time
        )
        if df.empty:
            logger.warning(f"No data found for {exchange}/{market}/{resolution}")
            return False
        return await self.tfrecord_storage.convert_candles_to_tfrecord(
            exchange, market, resolution, df
        )

    def get_pytorch_dataloader(
        self,
        exchange: str,
        market: str,
        resolution: str,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        """Get a PyTorch DataLoader (requires tensorflow)."""
        self._check_tfrecord()
        return self.tfrecord_storage.get_pytorch_dataloader(
            exchange, market, resolution, batch_size, shuffle
        )

    def get_tensorflow_dataset(
        self, exchange: str, market: str, resolution: str, batch_size: int = 32
    ):
        """Get a TensorFlow Dataset (requires tensorflow)."""
        self._check_tfrecord()
        return self.tfrecord_storage.get_tensorflow_dataset(
            exchange, market, resolution, batch_size
        )
