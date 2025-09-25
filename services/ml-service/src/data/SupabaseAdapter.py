from supabase import create_client
import pandas as pd
from datetime import datetime
import dateutil.parser
import os
import traceback


class SupabaseAdapter:
    """
    Adapter class to bridge your existing storage system with Supabase.
    This allows you to keep using your existing code while adding Supabase support.
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        """Initialize the Supabase adapter with credentials."""
        self.supabase = create_client(supabase_url, supabase_key)

        # Cache for exchanges and markets to avoid repeated lookups
        self.exchange_cache: dict[str, int] = {}
        self.market_cache: dict[str, int] = {}

    async def get_or_create_exchange(self, exchange_name: str) -> int:
        """Get exchange ID or create if it doesn't exist."""
        if exchange_name in self.exchange_cache:
            return self.exchange_cache[exchange_name]

        # Check if exchange exists
        response = (
            self.supabase.table("market_data.exchanges")
            .select("id")
            .eq("name", exchange_name)
            .execute()
        )

        if response.data and len(response.data) > 0:
            exchange_id = response.data[0]["id"]
        else:
            # Create new exchange
            response = (
                self.supabase.table("market_data.exchanges")
                .insert({"name": exchange_name})
                .execute()
            )
            exchange_id = response.data[0]["id"]

        # Cache the result
        self.exchange_cache[exchange_name] = exchange_id
        return exchange_id

    async def get_or_create_market(self, exchange_name: str, symbol: str) -> int:
        """Get market ID or create if it doesn't exist."""
        cache_key = f"{exchange_name}:{symbol}"
        if cache_key in self.market_cache:
            return self.market_cache[cache_key]

        # Get exchange_id
        exchange_id = await self.get_or_create_exchange(exchange_name)

        # Check if market exists
        response = (
            self.supabase.table("market_data.markets")
            .select("id")
            .eq("exchange_id", exchange_id)
            .eq("symbol", symbol)
            .execute()
        )

        if response.data and len(response.data) > 0:
            market_id = response.data[0]["id"]
        else:
            # Handle Bitget/Perp/Spot/Other formats
            if symbol.endswith("_UMCBL"):
                # Bitget USDT Perp: e.g., SOLUSDT_UMCBL
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
                # Perpetuals: e.g., BTC-PERP
                base_asset = symbol.split("-")[0].split("_")[0]
                quote_asset = "PERP"
                market_type = "PERP"
            elif "-" in symbol:
                # e.g., BTC-USD (Coinbase)
                parts = symbol.split("-")
                if len(parts) == 2:
                    base_asset, quote_asset = parts
                    market_type = "Spot"
            elif "/" in symbol:
                # e.g., BTC/USDT
                parts = symbol.split("/")
                if len(parts) == 2:
                    base_asset, quote_asset = parts
                    market_type = "Spot"
            elif symbol.endswith("USDT") or symbol.endswith("USDC") or symbol.endswith("USD"):
                # e.g., BTCUSDT, ETHUSDC
                for q in ["USDT", "USDC", "USD"]:
                    if symbol.endswith(q):
                        base_asset = symbol[:-len(q)]
                        quote_asset = q
                        market_type = "Spot"
                        break
            # Add more parsing rules as needed for other exchanges

            # Create new market
            response = (
                self.supabase.table("market_data.markets")
                .insert(
                    {
                        "exchange_id": exchange_id,
                        "type": market_type,  # Always provide type
                        "symbol": symbol,
                        "base_asset": base_asset,
                        "quote_asset": quote_asset,
                    }
                )
                .execute()
            )
            market_id = response.data[0]["id"]

        # Cache the result
        self.market_cache[cache_key] = market_id
        return market_id

    async def store_candles(
        self, exchange: str, market: str, resolution: str, df: pd.DataFrame
    ) -> bool:
        """
        Store candles from a pandas DataFrame into Supabase.
        Compatible with your ProcessedDataStorage format.

        Args:
            exchange: Exchange name (e.g., 'binance')
            market: Market symbol (e.g., 'SOL/USDT')
            resolution: Candle timeframe (e.g., '1m', '5m', '1h')
            df: DataFrame with OHLCV data

        Returns:
            bool: Success status
        """
        try:
            # Get market_id
            market_id = await self.get_or_create_market(exchange, market)
            print(
                f"SupabaseAdapter: Storing candles for market_id={market_id}, "
                f"exchange={exchange}, market={market}, "
                f"resolution={resolution}, "
                f"num_records={len(df)}"
            )
            # Prepare records for insertion
            records = []
            for _, row in df.iterrows():
                # Ensure ts is an ISO 8601 string
                ts = row["ts"] if "ts" in row else row["timestamp"]
                if isinstance(ts, str):
                    try:
                        ts = pd.to_datetime(ts)
                    except Exception:
                        ts = dateutil.parser.parse(ts)
                if hasattr(ts, 'to_pydatetime'):
                    ts = ts.to_pydatetime()
                ts = ts.isoformat()
                record = {
                    "market_id": market_id,
                    "resolution": str(resolution),
                    "ts": ts,
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row["volume"],
                }
                records.append(record)
            print(
                f"SupabaseAdapter: First 3 records to insert: {records[:3]}"
            )
            batch_size = 400
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                print(
                    f"SupabaseAdapter: Inserting batch of {len(batch)} records "
                    f"into candles table"
                )
                try:
                    response = self.supabase.table("market_data.candles").insert(
                        batch
                    ).execute()
                    print(
                        f"SupabaseAdapter: Supabase response: {response}"
                    )
                except Exception as e:
                    print(f"SupabaseAdapter: Exception during insert: {e}")
                    print("SupabaseAdapter: Writing batch to CSV as backup.")
                    # --- CSV BACKUP LOGIC ---
                    # Prepare DataFrame for CSV
                    backup_df = pd.DataFrame(batch)
                    # Remove market_id for CSV, add market and exchange columns
                    backup_df["market"] = market
                    backup_df["exchange"] = exchange
                    # Use consistent column order
                    cols = [
                        "ts", "open", "high", "low", "close", "volume",
                        "market", "exchange", "resolution"
                    ]
                    backup_df["resolution"] = str(resolution)
                    backup_df = backup_df[cols]
                    # Prepare directory and filename
                    backup_dir = os.path.join(
                        "data", "historical", "processed", "bitget"
                    )
                    os.makedirs(backup_dir, exist_ok=True)
                    backup_file = os.path.join(
                        backup_dir, f"{market}_{resolution}.csv"
                    )
                    # If file exists, load and deduplicate
                    if os.path.exists(backup_file):
                        existing = pd.read_csv(backup_file)
                        combined = pd.concat(
                            [existing, backup_df], ignore_index=True
                        )
                        combined.drop_duplicates(subset=["ts"], inplace=True)
                        combined.sort_values("ts", inplace=True)
                        combined.to_csv(backup_file, index=False)
                        print(
                            f"SupabaseAdapter: Appended and deduplicated to "
                            f"{backup_file}"
                        )
                    else:
                        backup_df.sort_values("ts", inplace=True)
                        backup_df.to_csv(backup_file, index=False)
                        print(
                            f"SupabaseAdapter: Wrote new backup file {backup_file}"
                        )
                    # Continue to next batch
            return True

        except Exception as e:
            print(
                f"Error storing candles: {e}\n{traceback.format_exc()}"
            )
            # Try to write the whole DataFrame to CSV as a last resort
            try:
                backup_dir = os.path.join(
                    "data", "historical", "processed", "bitget"
                )
                os.makedirs(backup_dir, exist_ok=True)
                backup_file = os.path.join(
                    backup_dir,
                    f"{market}_{resolution}_FALLBACK.csv"
                )
                df["market"] = market
                df["exchange"] = exchange
                df["resolution"] = str(resolution)
                cols = [
                    "ts" if "ts" in df.columns else "timestamp", "open", "high",
                    "low", "close", "volume", "market", "exchange", "resolution"
                ]
                # Rename timestamp to ts if needed
                if "timestamp" in df.columns and "ts" not in df.columns:
                    df = df.rename(columns={"timestamp": "ts"})
                df = df[cols]
                if os.path.exists(backup_file):
                    existing = pd.read_csv(backup_file)
                    combined = pd.concat(
                        [existing, df], ignore_index=True
                    )
                    combined.drop_duplicates(subset=["ts"], inplace=True)
                    combined.sort_values("ts", inplace=True)
                    combined.to_csv(backup_file, index=False)
                    print(
                        f"SupabaseAdapter: Appended and deduplicated to "
                        f"{backup_file}"
                    )
                else:
                    df.sort_values("ts", inplace=True)
                    df.to_csv(backup_file, index=False)
                    print(
                        f"SupabaseAdapter: Wrote new fallback backup file "
                        f"{backup_file}"
                    )
            except Exception as e2:
                print(
                    f"SupabaseAdapter: FATAL: Could not write fallback CSV: {e2}"
                )
            return False

    async def load_candles(
        self,
        exchange: str,
        market: str,
        resolution: str,
        start_time: datetime,
        end_time: datetime,
    ) -> pd.DataFrame:
        """
        Load candles from Supabase.
        Compatible with your ProcessedDataStorage.load_candles method.

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Get market_id
            market_id = await self.get_or_create_market(exchange, market)

            # Format timestamps
            start_iso = start_time.isoformat()
            end_iso = end_time.isoformat()

            # Query Supabase
            response = (
                self.supabase.table("market_data.candles")
                .select("ts,open,high,low,close,volume")
                .eq("market_id", market_id)
                .eq("resolution", resolution)
                .gte("ts", start_iso)
                .lte("ts", end_iso)
                .order("ts", desc=False)
                .execute()
            )

            if response.data:
                # Convert to DataFrame
                df = pd.DataFrame(response.data)

                # Convert ts to datetime
                df["ts"] = pd.to_datetime(df["ts"])

                return df

            return pd.DataFrame()

        except Exception as e:
            print(f"Error loading candles: {e}")
            return pd.DataFrame()

    async def store_prediction(
        self,
        exchange: str,
        market: str,
        resolution: str,
        timestamp: datetime,
        model_version: str,
        direction_prediction: float,
        magnitude_prediction: float,
        signal: int,
    ) -> bool:
        """
        Store ML model prediction in Supabase.

        Args:
            exchange: Exchange name
            market: Market symbol
            resolution: Candle timeframe
            timestamp: Prediction timestamp
            model_version: ML model version (e.g., 'pytorch_v1.0')
            direction_prediction: Probability of upward movement (0-1)
            magnitude_prediction: Predicted percent change
            signal: Trading signal (1=buy, -1=sell, 0=hold)

        Returns:
            bool: Success status
        """
        try:
            # Get market_id
            market_id = await self.get_or_create_market(exchange, market)

            # Prepare record
            record = {
                "market_id": market_id,
                "resolution": resolution,
                "ts": timestamp.isoformat(),
                "model_version": model_version,
                "direction_prediction": float(direction_prediction),
                "magnitude_prediction": float(magnitude_prediction),
                "signal": int(signal),
            }

            # Insert the prediction
            self.supabase.table("predictions").insert(record).execute()

            return True

        except Exception as e:
            print(f"Error storing prediction: {e}")
            return False
