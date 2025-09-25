"""
Symbol standardization for cryptocurrency exchanges.
Maps between a standard internal format and exchange-specific formats.
"""

import logging
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

class SymbolMapper:
    """
    Manages symbol mapping between a standardized internal format and exchange-specific formats.
    (Drift logic removed)
    """
    # Core supported assets - focusing only on major cryptocurrencies
    SUPPORTED_ASSETS = {
        "binance": ["SOL", "BTC", "ETH", "BNB"],  # Spot only
        "coinbase": ["SOL", "BTC", "ETH"]  # Spot only
    }
    # Quote currencies for each exchange
    QUOTE_CURRENCIES = {
        "binance": ["USDT"],  # Binance primarily uses USDT
        "coinbase": ["USD"]  # Coinbase uses USD
    }
    # Perpetual market support
    PERPETUAL_SUPPORT = {
        "binance": False,  # We're not using Binance perpetuals
        "coinbase": False
    }
    def __init__(self):
        self.to_exchange_map: Dict[str, Dict[str, str]] = {}
        self.from_exchange_map: Dict[str, Dict[str, str]] = {}
        self.supported_symbols: Dict[str, Set[str]] = {}
    def register_symbol(self, exchange: str, symbol: str, base_asset: str, 
                       quote_asset: str, is_perpetual: bool = False) -> None:
        exchange_id = exchange.lower()
        if exchange_id not in self.to_exchange_map:
            self.to_exchange_map[exchange_id] = {}
            self.from_exchange_map[exchange_id] = {}
            self.supported_symbols[exchange_id] = set()
        if is_perpetual:
            standard_symbol = f"{base_asset}-PERP"
        else:
            standard_symbol = f"{base_asset}-{quote_asset}"
        self.to_exchange_map[exchange_id][standard_symbol] = symbol
        self.from_exchange_map[exchange_id][symbol] = standard_symbol
        self.supported_symbols[exchange_id].add(standard_symbol)
        logger.debug(f"Registered symbol {symbol} → {standard_symbol} for {exchange}")
    def register_exchange(self, exchange: str, markets: List[str]) -> None:
        exchange_id = exchange.lower()
        for market in markets:
            try:
                if exchange_id == "binance":
                    for quote in self.QUOTE_CURRENCIES["binance"]:
                        if market.endswith(quote):
                            base = market[:-len(quote)]
                            if base in self.SUPPORTED_ASSETS["binance"]:
                                self.register_symbol(exchange, market, base, quote)
                                break
                elif exchange_id == "coinbase":
                    if "-" in market:
                        base, quote = market.split("-")
                        if base in self.SUPPORTED_ASSETS["coinbase"] and quote in self.QUOTE_CURRENCIES["coinbase"]:
                            self.register_symbol(exchange, market, base, quote)
            except Exception as e:
                logger.warning(f"Failed to register market {market} for {exchange}: {str(e)}")
                continue
    def to_exchange_symbol(self, exchange_name: str, standard_symbol: str) -> str:
        exchange_id = exchange_name.lower()
        if exchange_id in self.to_exchange_map and standard_symbol in self.to_exchange_map[exchange_id]:
            return self.to_exchange_map[exchange_id][standard_symbol]
        is_perp = standard_symbol.endswith("-PERP")
        if is_perp and not self.PERPETUAL_SUPPORT.get(exchange_id, False):
            raise ValueError(f"Exchange {exchange_name} does not support perpetual markets")
        try:
            if exchange_id == "binance":
                return self._standard_to_binance(standard_symbol)
            elif exchange_id == "coinbase":
                return self._standard_to_coinbase(standard_symbol)
            else:
                return standard_symbol.replace("-", "")
        except Exception as e:
            raise ValueError(f"Cannot convert {standard_symbol} to {exchange_name} format: {str(e)}")
    def from_exchange_symbol(self, exchange_name: str, exchange_symbol: str) -> str:
        exchange_id = exchange_name.lower()
        if exchange_id in self.from_exchange_map and exchange_symbol in self.from_exchange_map[exchange_id]:
            return self.from_exchange_map[exchange_id][exchange_symbol]
        try:
            if exchange_id == "binance":
                return self._binance_to_standard(exchange_symbol)
            elif exchange_id == "coinbase":
                return self._coinbase_to_standard(exchange_symbol)
            else:
                return exchange_symbol
        except Exception as e:
            raise ValueError(f"Cannot convert {exchange_symbol} to {exchange_name} format: {str(e)}")


# Example usage:
if __name__ == "__main__":
    mapper = SymbolMapper()
    
    # Register exchanges with their supported symbols
    # Binance: Spot only with USDT pairs
    mapper.register_exchange("binance", [
        "BTCUSDT", "ETHUSDT", "SOLUSDT"
    ])
    
    # Coinbase: Spot only with USD pairs
    mapper.register_exchange("coinbase", [
        "BTC-USD", "ETH-USD", "SOL-USD"
    ])
    
    # Jupiter: Spot only with USDC pairs
    mapper.register_exchange("jupiter", [
        "SOL-USDC", "BTC-USDC", "ETH-USDC"
    ])
    
    print("\nTesting standard spot symbols:")
    spot_symbols = ["BTC-USD", "ETH-USD", "SOL-USD"]
    for symbol in spot_symbols:
        print(f"\nTesting spot symbol: {symbol}")
        for exchange in ["binance", "coinbase", "jupiter"]:
            try:
                ex_sym = mapper.to_exchange_symbol(exchange, symbol)
                print(f"  {symbol} → {exchange}: {ex_sym}")
                # Convert back to standard to verify roundtrip
                std_sym = mapper.from_exchange_symbol(exchange, ex_sym)
                print(f"  {ex_sym} → standard: {std_sym}")
            except ValueError as e:
                print(f"  Error for {exchange}: {e}")
    
    print("\nTesting perpetual symbols:")
    perp_symbols = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
    for symbol in perp_symbols:
        print(f"\nTesting perp symbol: {symbol}")
        for exchange in ["binance", "coinbase", "jupiter"]:
            try:
                ex_sym = mapper.to_exchange_symbol(exchange, symbol)
                print(f"  {symbol} → {exchange}: {ex_sym}")
                # Convert back to standard to verify roundtrip
                std_sym = mapper.from_exchange_symbol(exchange, ex_sym)
                print(f"  {ex_sym} → standard: {std_sym}")
            except ValueError as e:
                print(f"  Error for {exchange}: {e}")
    
    print("\nTesting bare asset symbols:")
    assets = ["BTC", "ETH", "SOL"]
    for asset in assets:
        print(f"\nTesting asset: {asset}")
        for exchange in ["binance", "coinbase", "jupiter"]:
            try:
                ex_sym = mapper.to_exchange_symbol(exchange, asset)
                print(f"  {asset} → {exchange}: {ex_sym}")
                # Convert back to standard to verify roundtrip
                std_sym = mapper.from_exchange_symbol(exchange, ex_sym)
                print(f"  {ex_sym} → standard: {std_sym}")
            except ValueError as e:
                print(f"  Error for {exchange}: {e}")
    
    print("\nTesting exchange-specific formats:")
    exchange_symbols = {
        "binance": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "coinbase": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "jupiter": ["BTC-USDC", "ETH-USDC", "SOL-USDC"]
    }
    
    for exchange, symbols in exchange_symbols.items():
        print(f"\nTesting {exchange} symbols:")
        for symbol in symbols:
            try:
                std_sym = mapper.from_exchange_symbol(exchange, symbol)
                print(f"  {symbol} → standard: {std_sym}")
                # Convert back to exchange format to verify roundtrip
                ex_sym = mapper.to_exchange_symbol(exchange, std_sym)
                print(f"  {std_sym} → {exchange}: {ex_sym}")
            except ValueError as e:
                print(f"  Error for {symbol}: {e}")
