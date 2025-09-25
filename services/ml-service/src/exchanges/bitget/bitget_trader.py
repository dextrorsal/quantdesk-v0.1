import os
import logging
from .bitget.v2.mix.account_api import AccountApi
from .bitget.v1.mix.order_api import OrderApi  # Use v1 API for order placement
from .bitget.v2.mix.order_api import (
    OrderApi as OrderApiV2
)
from src.core.secure_config import get_exchange_config

logger = logging.getLogger(__name__)

class BitgetTrader:
    def __init__(self):
        # SECURITY FIX: Use secure configuration instead of direct env access
        config = get_exchange_config("bitget")
        
        self.api_key = config.get("api_key")
        self.api_secret = config.get("secret_key")
        self.passphrase = config.get("passphrase")
        
        # Log configuration status (without exposing secrets)
        logger.info(f"Bitget API Key: {'✓ Configured' if self.api_key else '✗ Missing'}")
        logger.info(f"Bitget Secret: {'✓ Configured' if self.api_secret else '✗ Missing'}")
        logger.info(f"Bitget Passphrase: {'✓ Configured' if self.passphrase else '✗ Missing'}")
        
        if not all([self.api_key, self.api_secret, self.passphrase]):
            raise ValueError("Missing Bitget API credentials. Check your .env file and variable names.")
        self.account_api = AccountApi(
            self.api_key, self.api_secret, self.passphrase
        )
        self.order_api = OrderApi(
            self.api_key, self.api_secret, self.passphrase
        )
        self.order_api_v2 = OrderApiV2(
            self.api_key, self.api_secret, self.passphrase
        )
        self.margin_coin = "USDT"
        self.product_type = "usdt-futures"

    def get_balance(self):
        params = {
            "productType": self.product_type,
            "marginCoin": self.margin_coin
        }
        result = self.account_api.accounts(params)
        return result

    def place_order(
        self, symbol, side, order_type, price, size, time_in_force="normal"
    ):
        params = {
            "symbol": symbol,  # e.g., "BTCUSDT_UMCBL"
            "marginCoin": self.margin_coin,
            "side": side,  # "open_long" or "close_long"
            "orderType": order_type,  # "limit" or "market"
            "size": str(size),
            "timInForceValue": time_in_force  # Note: typo matches Bitget's API
        }
        if price is not None:
            params["price"] = str(price)
        result = self.order_api.placeOrder(params)
        return result

    def place_order_v2(
        self, symbol, side, order_type, price, size, time_in_force="normal", position_mode="single"
    ):
        params = {
            "symbol": symbol,  # e.g., "BTCUSDT"
            "marginCoin": self.margin_coin,
            "side": side,  # "open_long" or "buy"
            "orderType": order_type,  # "limit" or "market"
            "size": str(size),
            "timeInForceValue": time_in_force,
            "marginMode": "crossed",
            "productType": "usdt-futures",
            "positionMode": position_mode
        }
        if price is not None:
            params["price"] = str(price)
        result = self.order_api_v2.placeOrder(params)
        return result

    def cancel_order(self, symbol, order_id):
        params = {
            "symbol": symbol,
            "marginCoin": self.margin_coin,
            "orderId": order_id
        }
        result = self.order_api.cancelOrder(params)
        return result

    def get_positions(self, symbol=None):
        params = {"productType": self.product_type}
        if symbol:
            params["symbol"] = symbol
            result = self.account_api.singlePosition(params)
        else:
            result = self.account_api.allPosition(params)
        return result

    def get_account_info_v2(self):
        combos = [
            {"productType": self.product_type, "marginCoin": self.margin_coin},
            {"productType": self.product_type},
            {"marginCoin": self.margin_coin},
            {}
        ]
        for params in combos:
            print(f"\n[Bitget v2 Account Info] Params: {params}")
            try:
                result = self.account_api.accounts(params)
                print(result)
            except Exception as e:
                print(f"Error: {e}")
        return None

    def set_leverage(self, symbol, leverage, hold_side):
        """Set leverage for a symbol and side (long/short) using v1 API."""
        params = {
            "symbol": symbol,
            "marginCoin": self.margin_coin,
            "leverage": str(leverage),
            "holdSide": hold_side  # 'long' or 'short'
        }
        result = self.account_api.setLeverage(params)
        print(f"Set leverage result for {symbol} {hold_side}: {result}")
        return result

    def reverse_position(self, symbol):
        """Reverse position: close current and open opposite (long<->short)."""
        # Fetch current position
        pos = self.get_positions(symbol)
        if not pos or not pos.get('data'):
            print(f"No open position to reverse for {symbol}.")
            return None
        data = pos['data']
        if isinstance(data, list):
            data = data[0] if data else None
        if not data or float(data.get('total', 0)) == 0:
            print(f"No open position to reverse for {symbol}.")
            return None
        side = data.get('holdSide')
        size = abs(float(data.get('total', 0)))
        if side == 'long':
            close_side = 'close_long'
            open_side = 'open_short'
        else:
            close_side = 'close_short'
            open_side = 'open_long'
        # Close current
        print(f"Closing {side} position of size {size} for {symbol}...")
        self.place_order(symbol, close_side, 'market', None, size)
        # Open opposite
        print(f"Opening opposite position...")
        self.place_order(symbol, open_side, 'market', None, size)
        return True

    def place_stop_loss(self, symbol, trigger_price, size, plan_type='pos_loss', hold_side='long'):
        """Place a stop loss (or take profit) plan order using v1 API."""
        params = {
            "symbol": symbol,
            "marginCoin": self.margin_coin,
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "planType": plan_type,  # 'pos_loss' for SL, 'pos_profit' for TP
            "holdSide": hold_side,  # 'long' or 'short'
            "orderType": "market"
        }
        result = self.order_api.placePlanOrder(params)
        print(f"Set {plan_type} result for {symbol}: {result}")
        return result

    def place_take_profit(self, symbol, trigger_price, size, hold_side='long'):
        """Place a take profit plan order using v1 API."""
        return self.place_stop_loss(symbol, trigger_price, size, plan_type='pos_profit', hold_side=hold_side)

    def place_trailing_stop(self, symbol, trigger_price, size):
        """Place a trailing stop plan order using v1 API (if supported)."""
        params = {
            "symbol": symbol,
            "marginCoin": self.margin_coin,
            "triggerPrice": str(trigger_price),
            "size": str(size),
            "planType": "moving_plan",  # Bitget's trailing stop plan type
            "holdSide": "long",  # or 'short', could be parameterized
            "orderType": "market"
        }
        try:
            result = self.order_api.placePlanOrder(params)
            print(f"Set trailing stop result for {symbol}: {result}")
            return result
        except Exception as e:
            print(f"Trailing stop not supported or error: {e}")
            return None


if __name__ == "__main__":
    trader = BitgetTrader()
    # Fetch and print v2 account info for diagnosis
    trader.get_account_info_v2()
    # Automated deep-dive: try all v2 order param combos
    sides = ["open_long", "buy"]
    margin_modes = ["crossed", "isolated"]
    product_types = ["usdt-futures", "umcbl"]
    position_modes = [None, "single"]
    for side in sides:
        for margin_mode in margin_modes:
            for product_type in product_types:
                for position_mode in position_modes:
                    params = {
                        "symbol": "BTCUSDT",
                        "marginCoin": "USDT",
                        "side": side,
                        "orderType": "market",
                        "size": 0.001,
                        "timeInForceValue": "normal",
                        "marginMode": margin_mode,
                        "productType": product_type
                    }
                    if position_mode:
                        params["positionMode"] = position_mode
                    print(f"\n[Bitget v2 Order Attempt] Params: {params}")
                    try:
                        result = trader.order_api_v2.placeOrder(params)
                        print("Result:")
                        print(result)
                    except Exception as e:
                        print(f"Error: {e}")

    print("Placing a small test order (open long)...")
    order_params = {
        "symbol": "BTCUSDT_UMCBL",  # Use v1 symbol format
        "side": "open_long",  # 'open_long' to open long in single mode
        "order_type": "market",
        "size": 0.001,  # Small size for safety
        "time_in_force": "normal"
    }
    order_result = trader.place_order(
        symbol=order_params["symbol"],
        side=order_params["side"],
        order_type=order_params["order_type"],
        price=None,  # Omit price for market order
        size=order_params["size"],
        time_in_force=order_params["time_in_force"]
    )
    print("Order result:")
    print(order_result)

    print("Fetching positions after opening...")
    positions = trader.get_positions()
    print(positions)

    print("Closing the test position...")
    # Use 'close_long' to close long in single mode
    close_result = trader.place_order(
        symbol=order_params["symbol"],
        side="close_long",
        order_type="market",
        price=None,
        size=order_params["size"],
        time_in_force="normal"
    )
    print("Close order result:")
    print(close_result)

    print("Fetching positions after closing...")
    positions = trader.get_positions()
    print(positions)

    # Example CLI/main block options to test features
    print("\n--- Bitget Feature Test ---")
    # 1. Set leverage
    trader.set_leverage("BTCUSDT_UMCBL", 5, "long")
    # 2. Reverse position
    trader.reverse_position("BTCUSDT_UMCBL")
    # 3. Place stop loss
    trader.place_stop_loss("BTCUSDT_UMCBL", 25000, 0.001)
    # 4. Place take profit
    trader.place_take_profit("BTCUSDT_UMCBL", 30000, 0.001)
    # 5. Place trailing stop
    trader.place_trailing_stop("BTCUSDT_UMCBL", 0.01, 0.001) 