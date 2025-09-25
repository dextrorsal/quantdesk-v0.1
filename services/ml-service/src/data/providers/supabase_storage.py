import pandas as pd
from datetime import datetime
from typing import List, Union
from shared.PostgresMarketDataAdapter import PostgresMarketDataAdapter


class PostgresDataStorage:
    """
    Storage class for OHLCV data using direct Postgres as the backend.
    Provides store_candles and load_candles methods compatible with DataManager.
    """

    def __init__(self):
        self.adapter = PostgresMarketDataAdapter()

    async def store_candles(
        self,
        exchange: str,
        market: str,
        resolution: str,
        candles: Union[List, pd.DataFrame],
    ):
        """
        Store candles in Postgres. Accepts a list of StandardizedCandle or a DataFrame.
        """
        # Convert to DataFrame if needed
        if isinstance(candles, list) and candles and hasattr(candles[0], "timestamp"):
            df = pd.DataFrame([candle.__dict__ for candle in candles])
        elif isinstance(candles, pd.DataFrame):
            df = candles
        else:
            raise ValueError(
                "Candles must be a list of StandardizedCandle objects or a DataFrame"
            )
        # Ensure timestamp is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.adapter.store_candles(exchange, market, resolution, df)

    async def load_candles(
        self,
        exchange: str,
        market: str,
        resolution: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Load candles from Postgres for the given time range.
        """
        # Not implemented yet in PostgresMarketDataAdapter
        raise NotImplementedError("Loading candles not yet implemented.")
