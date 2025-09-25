"""
Database connector for the ML trading bot.

This module provides a singleton database connector that handles all database operations
for the ML trading bot, including connections to the Neon PostgreSQL database.
"""

import os
import logging
import asyncpg  # type: ignore  # Missing stubs
from typing import Optional, List, Dict, Any
import pandas as pd  # type: ignore  # Missing stubs
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()

SCHEMA = "trading_bot"  # Updated to match new professional database structure


class DBConnector:
    """
    Singleton database connector for the ML trading bot.

    This class handles all database operations including:
    - Connection management
    - Data retrieval and storage
    - Signal management
    - Model performance tracking
    """

    _instance = None

    def __new__(cls, connection_string: Optional[str] = None):
        """Create a singleton instance of the connector"""
        if cls._instance is None:
            cls._instance = super(DBConnector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the database connector"""
        if self._initialized:
            return

        # If a connection string is provided, use it
        if connection_string:
            self.connection_string = connection_string
        # Otherwise, try to get it from environment variable
        else:
            self.connection_string = os.environ.get(
                "DATABASE_URL"
            )  # type: ignore

        self._pool = None
        self._initialized = True  # type: ignore[attr-defined]
        logger.info(
            "Database connector initialized"
        )

    async def connect(self) -> None:
        """Establish connection to the database"""
        if self._pool is None:
            try:
                self._pool = await asyncpg.create_pool(self.connection_string)
                logger.info("Connected to database")
            except Exception as e:
                logger.error(f"Failed to connect to database: {e}")
                raise

    async def close(self) -> None:
        """Close the database connection"""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Database connection closed")

    async def execute(self, query: str, *args) -> None:
        """Execute a SQL query"""
        await self.connect()
        async with self._pool.acquire() as conn:
            await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict[str, Any]]:
        """Fetch data from the database"""
        await self.connect()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]

    async def fetch_as_dataframe(self, query: str, *args) -> pd.DataFrame:
        """Fetch data and return as a pandas DataFrame"""
        rows = await self.fetch(query, *args)
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    async def insert_price_data(self, data: Dict[str, Any]) -> None:
        """Insert a single price data point"""
        await self.connect()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {SCHEMA}.price_data (timestamp, symbol, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (timestamp, symbol) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                data["timestamp"],
                data["symbol"],
                data["open"],
                data["high"],
                data["low"],
                data["close"],
                data["volume"],
            )

    async def insert_price_data_batch(self, data_points: List[Dict[str, Any]]) -> None:
        """Insert multiple price data points in a batch"""
        if not data_points:
            return

        await self.connect()
        async with self._pool.acquire() as conn:
            # Prepare the data for batch insert
            values = [
                (
                    d["timestamp"],
                    d["symbol"],
                    d["open"],
                    d["high"],
                    d["low"],
                    d["close"],
                    d["volume"],
                )
                for d in data_points
            ]

            await conn.executemany(
                f"""
                INSERT INTO {SCHEMA}.price_data (timestamp, symbol, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (timestamp, symbol) DO UPDATE
                SET open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                values,
            )

    async def get_recent_price_data(
        self, symbol: str, timeframe: str, days: int = 30
    ) -> pd.DataFrame:
        """Get recent price data for a symbol"""
        await self.connect()

        # Calculate the start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Query the database
        data = await self.fetch_as_dataframe(
            f"""
            SELECT timestamp, symbol, open, high, low, close, volume
            FROM {SCHEMA}.price_data
            WHERE symbol = $1
              AND timestamp >= $2
              AND timestamp <= $3
            ORDER BY timestamp ASC
            """,
            symbol,
            start_date,
            end_date,
        )

        # Convert timestamp to datetime
        if not data.empty and "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            data = data.set_index("timestamp")

        return data

    async def insert_trading_signal(self, signal: Dict[str, Any]) -> None:
        """Insert a trading signal into the database"""
        await self.connect()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {SCHEMA}.signals 
                (timestamp, symbol, signal_type, signal_strength, 
                 rsi_14, wt_value, cci_20, adx_20, rsi_9, 
                 confidence_5m, confidence_15m, weighted_confidence, price)
                VALUES 
                ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (timestamp, symbol, signal_type) DO UPDATE
                SET 
                    signal_strength = $4,
                    rsi_14 = $5,
                    wt_value = $6,
                    cci_20 = $7,
                    adx_20 = $8,
                    rsi_9 = $9,
                    confidence_5m = $10,
                    confidence_15m = $11,
                    weighted_confidence = $12,
                    price = $13
                """,
                signal.get("timestamp"),
                signal.get("symbol"),
                signal.get("signal_type"),
                signal.get("signal_strength"),
                signal.get("rsi_14"),
                signal.get("wt_value"),
                signal.get("cci_20"),
                signal.get("adx_20"),
                signal.get("rsi_9"),
                signal.get("confidence_5m"),
                signal.get("confidence_15m"),
                signal.get("weighted_confidence"),
                signal.get("price"),
            )

    async def get_recent_signals(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """Get recent trading signals for a symbol"""
        await self.connect()

        # Calculate the start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Query the database
        signals = await self.fetch_as_dataframe(
            f"""
            SELECT timestamp, symbol, signal_type, signal_strength, 
                   rsi_14, wt_value, cci_20, adx_20, rsi_9,
                   confidence_5m, confidence_15m, weighted_confidence, price
            FROM {SCHEMA}.signals
            WHERE symbol = $1
              AND timestamp >= $2
              AND timestamp <= $3
            ORDER BY timestamp DESC
            """,
            symbol,
            start_date,
            end_date,
        )

        # Convert timestamp to datetime
        if not signals.empty and "timestamp" in signals.columns:
            signals["timestamp"] = pd.to_datetime(signals["timestamp"])

        return signals

    async def record_model_prediction(self, prediction: Dict[str, Any]) -> None:
        """Record a model prediction"""
        await self.connect()
        async with self._pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {SCHEMA}.model_predictions
                (timestamp, model_id, symbol, timeframe, prediction_value, 
                 actual_outcome, confidence, features_used)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (timestamp, model_id, symbol, timeframe) DO UPDATE
                SET 
                    prediction_value = $5,
                    actual_outcome = $6,
                    confidence = $7,
                    features_used = $8
                """,
                prediction.get("timestamp"),
                prediction.get("model_id"),
                prediction.get("symbol"),
                prediction.get("timeframe"),
                prediction.get("prediction_value"),
                prediction.get("actual_outcome"),
                prediction.get("confidence"),
                prediction.get("features_used"),
            )


# Example usage:
# async def main():
#     db = DBConnector()
#     await db.connect()
#
#     # Get recent price data
#     data = await db.get_recent_price_data("SOLUSDT", "5m", days=7)
#     print(f"Got {len(data)} price records")
#
#     # Get recent signals
#     signals = await db.get_recent_signals("SOLUSDT", days=7)
#     print(f"Got {len(signals)} signals")
#
#     await db.close()
#
# if __name__ == "__main__":
#     asyncio.run(main())
