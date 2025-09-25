import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

USER = os.getenv("user")
PASSWORD = os.getenv("password")
HOST = os.getenv("host")
PORT = os.getenv("port")
DBNAME = os.getenv("dbname")


class PostgresMarketDataAdapter:
    def __init__(self):
        self.conn = psycopg2.connect(
            user=USER,
            password=PASSWORD,
            host=HOST,
            port=PORT,
            dbname=DBNAME
        )
        self.conn.autocommit = True
        self.exchange_cache = {}
        self.market_cache = {}

    def get_or_create_exchange(self, exchange_name):
        if exchange_name in self.exchange_cache:
            return self.exchange_cache[exchange_name]
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM market_data.exchanges WHERE name = %s;",
                (exchange_name,)
            )
            row = cur.fetchone()
            if row:
                exchange_id = row[0]
            else:
                cur.execute(
                    "INSERT INTO market_data.exchanges (name) VALUES (%s) RETURNING id;",
                    (exchange_name,)
                )
                exchange_id = cur.fetchone()[0]
            self.exchange_cache[exchange_name] = exchange_id
            return exchange_id

    def get_or_create_market(self, exchange_name, symbol):
        cache_key = f"{exchange_name}:{symbol}"
        if cache_key in self.market_cache:
            return self.market_cache[cache_key]
        exchange_id = self.get_or_create_exchange(exchange_name)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM market_data.markets WHERE exchange_id = %s "
                "AND symbol = %s;",
                (exchange_id, symbol)
            )
            row = cur.fetchone()
            if row:
                market_id = row[0]
            else:
                # Parsing logic (match previous adapter)
                base_asset = symbol
                quote_asset = "UNKNOWN"
                market_type = "Spot"
                if symbol.endswith("_UMCBL"):
                    core = symbol.replace("_UMCBL", "")
                    if core.endswith("USDT"):
                        base_asset = core[:-4]
                        quote_asset = "USDT"
                        market_type = "PERP"
                    elif core.endswith("USDC"):
                        base_asset = core[:-4]
                        quote_asset = "USDC"
                        market_type = "PERP"
                    else:
                        base_asset = core
                        quote_asset = "UNKNOWN"
                        market_type = "PERP"
                elif symbol.endswith("-PERP") or symbol.endswith("_PERP"):
                    base_asset = symbol.split("-")[0].split("_")[0]
                    quote_asset = "PERP"
                    market_type = "PERP"
                elif "-" in symbol:
                    parts = symbol.split("-")
                    if len(parts) == 2:
                        base_asset, quote_asset = parts
                        market_type = "Spot"
                elif "/" in symbol:
                    parts = symbol.split("/")
                    if len(parts) == 2:
                        base_asset, quote_asset = parts
                        market_type = "Spot"
                elif symbol.endswith("USDT") or symbol.endswith("USDC") or symbol.endswith("USD"):
                    for q in ["USDT", "USDC", "USD"]:
                        if symbol.endswith(q):
                            base_asset = symbol[:-len(q)]
                            quote_asset = q
                            market_type = "Spot"
                            break
                cur.execute(
                    (
                        """
                        INSERT INTO market_data.markets (exchange_id, type, symbol, base_asset, quote_asset)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (exchange_id, symbol) DO UPDATE SET type=EXCLUDED.type
                        RETURNING id;
                        """
                    ),
                    (exchange_id, market_type, symbol, base_asset, quote_asset)
                )
                market_id = cur.fetchone()[0]
            self.market_cache[cache_key] = market_id
            return market_id

    def store_candles(self, exchange, market, resolution, df: pd.DataFrame):
        market_id = self.get_or_create_market(exchange, market)
        records = []
        for _, row in df.iterrows():
            ts = row["ts"] if "ts" in row else row["timestamp"]
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            if hasattr(ts, 'to_pydatetime'):
                ts = ts.to_pydatetime()
            ts = ts.isoformat()
            records.append(
                (
                    market_id,
                    str(resolution),
                    ts,
                    row["open"],
                    row["high"],
                    row["low"],
                    row["close"],
                    row["volume"],
                )
            )
        with self.conn.cursor() as cur:
            cur.executemany(
                (
                    """
                    INSERT INTO market_data.candles (market_id, resolution, ts, open, high, low, close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (market_id, resolution, ts) DO UPDATE
                    SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume;
                    """
                ),
                records
            )
        return True

    def close(self):
        self.conn.close() 