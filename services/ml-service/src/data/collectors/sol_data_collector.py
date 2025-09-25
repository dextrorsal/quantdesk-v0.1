from datetime import datetime, timedelta, timezone
import asyncio
import pandas as pd
from binance.spot import Spot
from typing import Optional, List
import logging
import json
import asyncpg
from queue import Queue
import websocket
import threading
import time
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCHEMA = "trading_bot"  # Updated to match new professional database structure


class SOLDataCollector:
    def __init__(self, neon_connection_string: Optional[str] = None):
        self.client = Spot()
        self.symbol = "SOLUSDT"
        self.timeframes = {
            "5m": "5m",
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
        }

        # Initialize database connection
        self.neon_conn_str = neon_connection_string
        self.data_queue: Queue = Queue()
        self.is_running = False
        self._db_pool = None

        # In-memory cache for latest data
        self.cache: dict = {"last_price": None, "last_update": None, "candles": {}}

        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.ws_connected = False
        self.stream_subscriptions = []

    async def init_db(self):
        """Initialize database connection pool"""
        if not self._db_pool and self.neon_conn_str:
            self._db_pool = await asyncpg.create_pool(self.neon_conn_str)

    def on_message(self, ws, message, *args):
        if not isinstance(message, (str, bytes, bytearray)):
            print(f"Skipping non-message object: {message} ({type(message)})")
            return
        print("Received message:", message)
        print("Type of message:", type(message))
        if isinstance(message, bytes):
            try:
                message = message.decode("utf-8")
            except Exception as e:
                print(f"Failed to decode bytes message: {e}")
                return
        try:
            data = json.loads(message)
        except Exception as e:
            print(f"Failed to parse message: {message}")
            print(f"Error: {e}")
            return

        if "e" in data:  # Kline/Candlestick event
            if data["e"] == "kline":
                k = data["k"]
                candle_data = {
                    "symbol": self.symbol,
                    "timestamp": datetime.fromtimestamp(k["t"] / 1000),
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                }

                # Update cache
                self.cache["last_price"] = float(k["c"])
                self.cache["last_update"] = datetime.now()

                # Add to queue for database insertion
                self.data_queue.put(candle_data)

                logger.debug(f"Received candle: {candle_data}")

    def on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close"""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_connected = False

        # Attempt to reconnect if we're supposed to be running
        if self.is_running:
            logger.info("Attempting to reconnect...")
            time.sleep(5)  # Wait before reconnecting
            self._connect_websocket()

    def on_open(self, ws):
        """Handle WebSocket open"""
        logger.info("WebSocket connection established")
        self.ws_connected = True

        # Subscribe to all streams
        if self.stream_subscriptions:
            self._subscribe_to_streams(self.stream_subscriptions)

    def on_ping(self, ws, message):
        """Handle ping from server"""
        logger.debug(f"Received ping: {message}")
        # The websocket-client library will automatically respond with pong

    def on_pong(self, ws, message):
        """Handle pong from server"""
        logger.debug(f"Received pong: {message}")

    def _connect_websocket(self):
        """Connect to Binance WebSocket"""
        # Close existing connection if any
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass

        # Create new WebSocket connection
        websocket.enableTrace(False)
        socket_url = "wss://stream.binance.com:9443/stream"

        self.ws = websocket.WebSocketApp(
            socket_url,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
            on_ping=self.on_ping,
            on_pong=self.on_pong,
        )

        # Start WebSocket in a separate thread
        self.ws_thread = threading.Thread(
            target=self.ws.run_forever,
            kwargs={
                "ping_interval": 180,  # Send ping every 3 minutes
                "ping_timeout": 10,  # Wait 10 seconds for pong response
                "ping_payload": "",  # Empty ping payload as recommended by Binance
            },
        )
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Wait for connection to establish
        timeout = 10  # seconds
        start_time = time.time()
        while not self.ws_connected and time.time() - start_time < timeout:
            time.sleep(0.1)

        if not self.ws_connected:
            logger.error("Failed to establish WebSocket connection")
            return False

        return True

    def _subscribe_to_streams(self, streams: List[str]):
        """Subscribe to multiple streams"""
        if not self.ws_connected:
            logger.error("Cannot subscribe: WebSocket not connected")
            return False

        try:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": int(time.time() * 1000),  # Use timestamp as ID
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to streams: {streams}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to streams: {e}")
            return False

    async def db_worker(self):
        """Background worker to handle database insertions"""
        await self.init_db()

        while self.is_running:
            try:
                if self._db_pool and not self.data_queue.empty():
                    async with self._db_pool.acquire() as conn:
                        batch_data = []
                        while not self.data_queue.empty() and len(batch_data) < 100:
                            data = self.data_queue.get()
                            batch_data.append(
                                (
                                    data["timestamp"],
                                    data["symbol"],
                                    data["open"],
                                    data["high"],
                                    data["low"],
                                    data["close"],
                                    data["volume"],
                                )
                            )

                        if batch_data:
                            await conn.executemany(
                                f"""
                                INSERT INTO {SCHEMA}.price_data (
                                    timestamp, symbol, open, high, low, close, volume)
                                VALUES ($1, $2, $3, $4, $5, $6, $7)
                                ON CONFLICT (timestamp, symbol) DO UPDATE
                                SET open = EXCLUDED.open,
                                    high = EXCLUDED.high,
                                    low = EXCLUDED.low,
                                    close = EXCLUDED.close,
                                    volume = EXCLUDED.volume
                                """,
                                batch_data,
                            )

                await asyncio.sleep(1)  # Prevent CPU overload

            except Exception as e:
                logger.error(f"Database worker error: {e}")
                await asyncio.sleep(5)  # Back off on error

    async def start_websocket(self):
        """Start WebSocket connection for real-time data"""
        self.is_running = True

        try:
            # Prepare stream subscriptions
            self.stream_subscriptions = []
            for tf in self.timeframes.values():
                stream_name = f"{self.symbol.lower()}@kline_{tf}"
                self.stream_subscriptions.append(stream_name)

            # Connect to WebSocket
            if not self._connect_websocket():
                raise Exception("Failed to connect to WebSocket")

            # Start database worker task
            asyncio.create_task(self.db_worker())

            logger.info(f"WebSocket started for {self.symbol}")

            # Keep the async task running
            while self.is_running:
                if not self.ws_connected:
                    logger.warning("WebSocket disconnected, attempting to reconnect...")
                    self._connect_websocket()

                await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_running = False
            raise

    async def stop_websocket(self):
        """Stop WebSocket connection"""
        self.is_running = False

        if self.ws:
            try:
                self.ws.close()
                logger.info("WebSocket stopped")
            except Exception as e:
                logger.error(f"Error stopping WebSocket: {e}")

        if self.ws_thread and self.ws_thread.is_alive():
            # Wait for thread to finish (with timeout)
            self.ws_thread.join(timeout=2)

    def get_cached_price(self) -> Optional[float]:
        """Get latest cached price"""
        return self.cache["last_price"]

    async def fetch_all_timeframes(self, lookback_days=30):
        # Fetches all timeframes for the given lookback period
        # For each timeframe, fetch in batches if needed
        result = {}
        for tf in self.timeframes:
            print(f"Fetching {tf} data for {self.symbol}...")
            all_candles = []
            end_time = int(datetime.now().timestamp() * 1000)
            interval_minutes = {
                "5m": 5,
                "15m": 15,
                "1h": 60,
                "4h": 240,
                "1d": 1440,
            }[tf]
            total_candles = int((lookback_days * 24 * 60) / interval_minutes)
            limit = 1000
            while total_candles > 0:
                fetch_limit = min(limit, total_candles)
                candles = self.client.klines(
                    symbol=self.symbol,
                    interval=tf,
                    limit=fetch_limit,
                    endTime=end_time,
                )
                if not candles:
                    break
                all_candles = candles + all_candles
                end_time = candles[0][0] - 1
                total_candles -= len(candles)
                if len(candles) < fetch_limit:
                    break
            df = pd.DataFrame(
                all_candles,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            result[tf] = df
            print(f"Successfully fetched {len(df)} {tf} candles")
        return result

    async def fetch_historical(
        self, timeframe: str, lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Fetch historical data for a specific timeframe.

        Args:
            timeframe: The timeframe to fetch ('5m', '15m', '1h', '4h', '1d')
            lookback_days: Number of days of historical data to fetch

        Returns:
            DataFrame with historical price data
        """
        if timeframe not in self.timeframes:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. Must be one of {list(self.timeframes.keys())}"
            )

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=lookback_days)

        logger.info(
            f"Fetching {timeframe} data for {self.symbol} from {start_time} to {end_time}"
        )

        try:
            klines = self.client.klines(
                symbol=self.symbol,
                interval=self.timeframes[timeframe],
                startTime=int(start_time.timestamp() * 1000),
                endTime=int(end_time.timestamp() * 1000),
                limit=1000,
            )

            # Convert to DataFrame with proper column names
            df = pd.DataFrame(
                klines,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )

            # Convert types
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)

            # Add timeframe info
            df["timeframe"] = timeframe

            logger.info(f"Successfully fetched {len(df)} {timeframe} candles")

            return df

        except Exception as e:
            logger.error(f"Error fetching {timeframe} data: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error


async def main():
    # Load environment variables from .env
    load_dotenv()
    neon_conn_str = os.getenv("NEON_DATABASE_URL")

    # Initialize collector with database connection
    collector = SOLDataCollector(neon_conn_str)

    try:
        # Start WebSocket for real-time data
        await collector.start_websocket()
    except KeyboardInterrupt:
        await collector.stop_websocket()
        logger.info("Gracefully shut down")


if __name__ == "__main__":
    asyncio.run(main())
