import requests
import json
from . import consts as c, utils, exceptions


class Client(object):

    def __init__(self, api_key, api_secret_key, passphrase, use_server_time=False, first=False, demo=False, base_url=None):

        self.API_KEY = api_key
        self.API_SECRET_KEY = api_secret_key
        self.PASSPHRASE = passphrase
        self.use_server_time = use_server_time
        self.first = first
        self.demo = demo
        self.base_url = base_url or c.API_URL

    def _request(self, method, request_path, params, cursor=False):
        if method == c.GET:
            request_path = request_path + utils.parse_params_to_str(params)
        # url
        url = self.base_url + request_path

        # 获取本地时间
        timestamp = utils.get_timestamp()
        print('DEBUG: Local timestamp:', timestamp)

        # sign & header
        if self.use_server_time:
            # 获取服务器时间接口
            timestamp = self._get_timestamp()
            print('DEBUG: Server timestamp:', timestamp)

        body = json.dumps(params) if method == c.POST else ""
        sign = utils.sign(utils.pre_hash(timestamp, method, request_path, str(body)), self.API_SECRET_KEY)
        if c.SIGN_TYPE == c.RSA:
            sign = utils.signByRSA(utils.pre_hash(timestamp, method, request_path, str(body)), self.API_SECRET_KEY)
        header = utils.get_header(self.API_KEY, sign, timestamp, self.PASSPHRASE)
        if self.demo:
            header['paptrading'] = '1'
            print('DEBUG: paptrading header set for demo mode:', header)

        if self.first:
            print("url:", url)
            print("method:", method)
            print("body:", body)
            print("headers:", header)
            # print("sign:", sign)
            self.first = False

        # --- ADDED DEBUG PRINTS ---
        print("DEBUG: URL:", url)
        print("DEBUG: Method:", method)
        print("DEBUG: Headers:", header)
        print("DEBUG: Body:", body)
        # --- END DEBUG PRINTS ---

        # send request
        response = None
        if method == c.GET:
            response = requests.get(url, headers=header)
            print("response : ",response.text)
        elif method == c.POST:
            response = requests.post(url, data=body, headers=header)
            print("response : ",response.text)
            #response = requests.post(url, json=body, headers=header)
        elif method == c.DELETE:
            response = requests.delete(url, headers=header)

        # --- ADDED DEBUG PRINTS ---
        print("DEBUG: Response Text:", response.text[:200] + ("..." if len(response.text) > 200 else ""))
        # --- END DEBUG PRINTS ---

        print("status:", response.status_code)
        # exception handle
        if not str(response.status_code).startswith('2'):
            raise exceptions.BitgetAPIException(response)
        try:
            res_header = response.headers
            if cursor:
                r = dict()
                try:
                    r['before'] = res_header['OK-BEFORE']
                    r['after'] = res_header['OK-AFTER']
                except:
                    pass
                return response.json(), r
            else:
                return response.json()

        except ValueError:
            raise exceptions.BitgetRequestException('Invalid Response: %s' % response.text)

    def _request_without_params(self, method, request_path):
        return self._request(method, request_path, {})

    def _request_with_params(self, method, request_path, params, cursor=False):
        return self._request(method, request_path, params, cursor)

    def _get_timestamp(self):
        url = self.base_url + c.SERVER_TIMESTAMP_URL
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json().get('data')
            if isinstance(data, dict) and 'serverTime' in data:
                return data['serverTime']
            return ""
        else:
            return ""

    def get_account_assets(self, product_type=None, margin_coin=None):
        """Fetch account assets for spot, margin, or futures (demo/real)."""
        # For demo USDT-M futures: product_type='susdt-futures', margin_coin='SUSDT'
        # Endpoint: /api/v2/mix/account/accounts
        params = {}
        if product_type:
            params['productType'] = product_type
        if margin_coin:
            params['marginCoin'] = margin_coin
        return self._request(c.GET, '/api/v2/mix/account/accounts', params)

    def get_positions(self, product_type=None, margin_coin=None):
        """Fetch all open positions for the account (demo/real). Do NOT pass symbol to this endpoint."""
        # Endpoint: /api/v2/mix/position/all-position
        params = {}
        if product_type:
            params['productType'] = product_type
        if margin_coin:
            params['marginCoin'] = margin_coin
        return self._request(c.GET, '/api/v2/mix/position/all-position', params)

    def get_single_position(self, symbol, product_type=None, margin_coin=None):
        """Fetch open position for a specific symbol (demo/real)."""
        # Endpoint: /api/v2/mix/position/single-position
        params = {'symbol': symbol}
        if product_type:
            params['productType'] = product_type
        if margin_coin:
            params['marginCoin'] = margin_coin
        return self._request(c.GET, '/api/v2/mix/position/single-position', params)

    def set_leverage(self, symbol, leverage, product_type, margin_coin, margin_mode):
        """Set leverage for a symbol."""
        # Endpoint: /api/v2/mix/account/set-leverage
        params = {
            'symbol': symbol,
            'leverage': str(leverage),
            'marginMode': margin_mode,
            'productType': product_type,
            'marginCoin': margin_coin
        }
        return self._request(c.POST, '/api/v2/mix/account/set-leverage', params)

    def set_margin_mode(self, symbol, margin_mode, product_type, margin_coin):
        """Set margin mode for a symbol (isolated/cross)."""
        # Endpoint: /api/v2/mix/account/set-margin-mode
        params = {
            'symbol': symbol,
            'marginMode': margin_mode,
            'productType': product_type,
            'marginCoin': margin_coin
        }
        return self._request(c.POST, '/api/v2/mix/account/set-margin-mode', params)

    def set_asset_mode(self, asset_mode):
        """Switch asset mode (single/multi-asset) if supported."""
        # Endpoint: /api/v2/mix/account/set-position-mode
        params = {'positionMode': asset_mode}
        return self._request(c.POST, '/api/v2/mix/account/set-position-mode', params)

    def place_order(self, symbol, side, order_type, size, price=None, product_type=None, margin_coin=None, margin_mode=None, leverage=None, asset_mode=None, reduce_only=False, **kwargs):
        """Place an order with all relevant params (spot, margin, futures, demo/real)."""
        # Endpoint: /api/v2/mix/order/place-order
        params = {
            'symbol': symbol,
            'side': side,
            'orderType': order_type,
            'size': str(size),
            'reduceOnly': 'YES' if reduce_only else 'NO'
        }
        if price is not None:
            params['price'] = str(price)
        if product_type:
            params['productType'] = product_type
        if margin_coin:
            params['marginCoin'] = margin_coin
        if margin_mode:
            params['marginMode'] = margin_mode
        if leverage:
            params['leverage'] = str(leverage)
        if asset_mode:
            params['positionMode'] = asset_mode
        params.update(kwargs)
        return self._request(c.POST, '/api/v2/mix/order/place-order', params)

    def cancel_order(self, symbol, order_id, product_type=None, margin_coin=None):
        """Cancel an order (spot, margin, futures, demo/real)."""
        # Endpoint: /api/v2/mix/order/cancel-order
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        if product_type:
            params['productType'] = product_type
        if margin_coin:
            params['marginCoin'] = margin_coin
        return self._request(c.POST, '/api/v2/mix/order/cancel-order', params)
