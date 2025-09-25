"""
Neon Data Collector - Fetches market data and stores it in Neon database
"""

import pandas as pd
from sqlalchemy import create_engine
import ccxt
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
import logging

class NeonDataCollector:
    def __init__(self, connection_string: str, exchange_id: str = 'binance'):
        """
        Initialize the data collector
        
        Args:
            connection_string: Neon database connection string
            exchange_id: CCXT exchange ID (default: 'binance')
        """
        self.engine = create_engine(connection_string)
        self.exchange = getattr(ccxt, exchange_id)()
        self.logger = logging.getLogger(__name__)
        
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1m',
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from exchange
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            since: Start time
            limit: Number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert since to timestamp if provided
            since_ts = int(since.timestamp() * 1000) if since else None
            
            # Fetch data
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=since_ts,
                limit=limit
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
            
    def store_ohlcv(self, df: pd.DataFrame, symbol: str) -> None:
        """
        Store OHLCV data in Neon database
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading pair symbol
        """
        try:
            # Add symbol column
            df['symbol'] = symbol
            
            # Store in database
            df.to_sql(
                'price_data',
                self.engine,
                if_exists='append',
                index=False,
                method='multi'
            )
            
        except Exception as e:
            self.logger.error(f"Error storing data: {str(e)}")
            raise
            
    def collect_historical(
        self,
        symbol: str,
        days: int = 30,
        timeframe: str = '1m'
    ) -> None:
        """
        Collect historical data for a symbol
        
        Args:
            symbol: Trading pair symbol
            days: Number of days to collect
            timeframe: Candle timeframe
        """
        try:
            since = datetime.now() - timedelta(days=days)
            
            while since < datetime.now():
                # Fetch batch of data
                df = self.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=1000
                )
                
                if df.empty:
                    break
                    
                # Store data
                self.store_ohlcv(df, symbol)
                
                # Update since
                since = df['timestamp'].max()
                
                # Rate limit
                time.sleep(self.exchange.rateLimit / 1000)
                
        except Exception as e:
            self.logger.error(f"Error collecting historical data: {str(e)}")
            raise
            
    def collect_realtime(
        self,
        symbols: List[str],
        timeframe: str = '1m'
    ) -> None:
        """
        Collect real-time data for multiple symbols
        
        Args:
            symbols: List of trading pair symbols
            timeframe: Candle timeframe
        """
        try:
            while True:
                for symbol in symbols:
                    # Fetch latest data
                    df = self.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        limit=1
                    )
                    
                    if not df.empty:
                        # Store data
                        self.store_ohlcv(df, symbol)
                    
                    # Rate limit
                    time.sleep(self.exchange.rateLimit / 1000)
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping real-time collection")
        except Exception as e:
            self.logger.error(f"Error collecting real-time data: {str(e)}")
            raise 