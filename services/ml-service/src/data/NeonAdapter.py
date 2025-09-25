import asyncpg
import pandas as pd
import logging
import dateutil.parser


class NeonAdapter:
    def __init__(self, neon_conn_str: str):
        self.neon_conn_str = neon_conn_str
        self._db_pool = None
        self.exchange_cache = {}
        self.market_cache = {}
        self.logger = logging.getLogger(__name__)

    async def init_db(self):
        if not self._db_pool:
            self._db_pool = await asyncpg.create_pool(self.neon_conn_str)

    async def get_or_create_exchange(self, exchange_name: str) -> int:
        await self.init_db()
        if exchange_name in self.exchange_cache:
            return self.exchange_cache[exchange_name]
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM exchanges WHERE name = $1", exchange_name
            )
            if row:
                exchange_id = row["id"]
            else:
                row = await conn.fetchrow(
                    "INSERT INTO exchanges (name) VALUES ($1) RETURNING id",
                    exchange_name
                )
                exchange_id = row["id"]
            self.exchange_cache[exchange_name] = exchange_id
            return exchange_id

    async def get_or_create_market(self, exchange_name: str, symbol: str) -> int:
        await self.init_db()
        cache_key = f"{exchange_name}:{symbol}"
        if cache_key in self.market_cache:
            return self.market_cache[cache_key]
        exchange_id = await self.get_or_create_exchange(exchange_name)
        async with self._db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id FROM markets WHERE exchange_id = $1 AND symbol = $2",
                exchange_id,
                symbol
            )
            if row:
                market_id = row["id"]
            else:
                # Basic parsing for base/quote/type (match SupabaseAdapter logic)
                base_asset = symbol
                quote_asset = "UNKNOWN"
                market_type = "Spot"
                if symbol.endswith("_UMCBL"):
                    core = symbol.replace("_UMCBL", "")
                    if core.endswith("USDT"):
                        base_asset = core[:-4]
                        quote_asset = "USDT"
                    elif core.endswith("USDC"):
                        base_asset = core[:-4]
                        quote_asset = "USDC"
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
                row = await conn.fetchrow(
                    """
                    INSERT INTO markets (exchange_id, type, symbol, base_asset, quote_asset)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (exchange_id, symbol) DO UPDATE SET type=EXCLUDED.type
                    RETURNING id
                    """,
                    exchange_id,
                    market_type,
                    symbol,
                    base_asset,
                    quote_asset
                )
                market_id = row["id"]
            self.market_cache[cache_key] = market_id
            return market_id

    async def store_candles(self, exchange: str, market: str, resolution: str, df: pd.DataFrame) -> bool:
        await self.init_db()
        try:
            market_id = await self.get_or_create_market(exchange, market)
            self.logger.info(
                f"NeonAdapter: Storing candles for market_id={market_id}, "
                f"exchange={exchange}, market={market}, "
                f"resolution={resolution}, "
                f"num_records={len(df)}"
            )
            records = []
            for _, row in df.iterrows():
                ts = row["ts"] if "ts" in row else row["timestamp"]
                if isinstance(ts, str):
                    try:
                        ts = pd.to_datetime(ts)
                    except Exception:
                        ts = dateutil.parser.parse(ts)
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
            async with self._db_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO candles (market_id, resolution, ts, open, high, low, close, volume)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (market_id, resolution, ts) DO UPDATE
                    SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume
                    """,
                    records
                )
            self.logger.info(
                f"NeonAdapter: Inserted {len(records)} candles."
            )
            return True
        except Exception as e:
            self.logger.error(
                f"NeonAdapter: Error storing candles: {e}"
            )
            return False 