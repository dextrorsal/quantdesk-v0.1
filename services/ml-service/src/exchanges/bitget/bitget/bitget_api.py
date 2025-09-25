#!/usr/bin/python
from .client import Client
from .consts import GET, POST


class BitgetApi(Client):
    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, first=False, demo=False, base_url=None):
        super().__init__(api_key, api_secret_key, passphrase, use_server_time, first, demo, base_url)

    def post(self, request_path, params):
        return self._request_with_params(POST, request_path, params)

    def get(self, request_path, params):
        return self._request_with_params(GET, request_path, params)

    def get_account_assets(self, product_type=None, margin_coin=None):
        return super().get_account_assets(product_type, margin_coin)

    def get_positions(self, symbol=None, product_type=None, margin_coin=None):
        return super().get_positions(symbol, product_type, margin_coin)

    def set_leverage(self, symbol, leverage, product_type, margin_coin, margin_mode):
        return super().set_leverage(symbol, leverage, product_type, margin_coin, margin_mode)

    def set_margin_mode(self, symbol, margin_mode, product_type, margin_coin):
        return super().set_margin_mode(symbol, margin_mode, product_type, margin_coin)

    def set_asset_mode(self, asset_mode):
        return super().set_asset_mode(asset_mode)

    def place_order(self, symbol, side, order_type, size, price=None, product_type=None, margin_coin=None, margin_mode=None, leverage=None, asset_mode=None, reduce_only=False, **kwargs):
        return super().place_order(symbol, side, order_type, size, price, product_type, margin_coin, margin_mode, leverage, asset_mode, reduce_only, **kwargs)

    def cancel_order(self, symbol, order_id, product_type=None, margin_coin=None):
        return super().cancel_order(symbol, order_id, product_type, margin_coin)
