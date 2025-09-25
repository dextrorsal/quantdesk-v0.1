"""
Bitget WebSocket to Supabase Pipeline

This script is the main, robust pipeline for streaming live or demo Bitget candles
and storing them in Supabase. It supports multiple symbols and resolutions, both live and demo modes,
and handles reconnects and CLI configuration.

For minimal REST and WebSocket Bitget examples, see:
- d3x7-algo/scripts/archive/fetch_bitget_live_candles.py (REST, single-symbol, quick fetch)
- d3x7-algo/scripts/archive/fetch_bitget_demo_ws.py (WebSocket, demo, minimal subscription)

These archived scripts are useful for reference or educational purposes.
"""
import os
import sys
import argparse
import asyncio
import json
from datetime import datetime
import pandas as pd
import websockets
from shared.PostgresMarketDataAdapter import PostgresMarketDataAdapter
from dotenv import load_dotenv

# Load .env for Supabase credentials
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

BITGET_WS_URLS = {
    "live": "wss://ws.bitget.com/v2/ws/public",
    "demo": "wss://wspap.bitget.com/v2/ws/public"
}

SYMBOL_PREFIX = {
    "live": "",
    "demo": "S"
}

INST_TYPE = "USDT-FUTURES"

CHANNEL_MAP = {
    "1": "candle1m",
    "5": "candle5m",
    "15": "candle15m",
    "30": "candle30m",
    "60": "candle1H"
}

# Note: For type checking, install pandas stubs with 'pip install pandas-stubs'

async def subscribe_and_store(mode, symbols, resolutions, supabase_url, supabase_key):
    ws_url = BITGET_WS_URLS[mode]
    adapter = PostgresMarketDataAdapter()
    # Prepare subscription args
    args = []
    for symbol in symbols:
        inst_id = (
            f"{SYMBOL_PREFIX[mode]}{symbol}" if mode == "demo" else symbol
        )
        for res in resolutions:
            channel = CHANNEL_MAP.get(res, f"candle{res}m")
            args.append({
                "instType": INST_TYPE,
                "channel": channel,
                "instId": inst_id
            })
    subscribe_msg = {"op": "subscribe", "args": args}
    print(
        f"Connecting to {ws_url} and subscribing to: {args}"
    )
    while True:
        try:
            async with websockets.connect(ws_url) as ws:
                await ws.send(json.dumps(subscribe_msg))
                print("Subscribed. Waiting for data...")
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    # Only process candle data
                    if (
                        data.get("arg", {}).get("channel", "").startswith("candle")
                        and "data" in data
                    ):
                        inst_id = data["arg"]["instId"]
                        channel = data["arg"]["channel"]
                        # Map channel back to resolution
                        for k, v in CHANNEL_MAP.items():
                            if v == channel:
                                resolution = k
                                break
                        else:
                            resolution = channel
                        # Remove demo prefix for storage
                        market = (
                            inst_id[1:]
                            if mode == "demo" and inst_id.startswith("S")
                            else inst_id
                        )
                        # Each data entry is a candle
                        for candle in data["data"]:
                            # Format: [timestamp, open, high, low, close, volume, quoteVolume]
                            ts = datetime.utcfromtimestamp(
                                int(candle[0]) // 1000
                            )
                            row = {
                                "timestamp": ts,
                                "open": float(candle[1]),
                                "high": float(candle[2]),
                                "low": float(candle[3]),
                                "close": float(candle[4]),
                                "volume": float(candle[5]),
                            }
                            df = pd.DataFrame([row])
                            await adapter.store_candles(
                                exchange=f"bitget_{mode}",
                                market=market,
                                resolution=(
                                    f"{resolution}m" if resolution.isdigit() else resolution
                                ),
                                df=df,
                            )
                            print(
                                f"Stored candle: {market} {resolution} {ts}"
                            )
        except Exception as e:
            print(
                f"WebSocket error: {e}. Reconnecting in 5 seconds..."
            )
            await asyncio.sleep(5)


def main():
    parser = argparse.ArgumentParser(
        description="Bitget WebSocket to Supabase"
    )
    parser.add_argument(
        "--mode", choices=["live", "demo"], required=True,
        help="Trading mode: live or demo"
    )
    parser.add_argument(
        "--symbols", nargs="+", required=True,
        help="Symbols (e.g. BTCUSDT ETHUSDT)"
    )
    parser.add_argument(
        "--resolutions", nargs="+", required=True,
        help="Resolutions (e.g. 1 5 15 30 60)"
    )
    args = parser.parse_args()
    if not SUPABASE_URL or not SUPABASE_KEY:
        print(
            "Please set SUPABASE_URL and SUPABASE_KEY in your .env file."
        )
        sys.exit(1)
    asyncio.run(
        subscribe_and_store(
            args.mode, args.symbols, args.resolutions, SUPABASE_URL, SUPABASE_KEY
        )
    )


if __name__ == "__main__":
    main() 