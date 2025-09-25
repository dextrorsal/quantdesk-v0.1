from bitget.v2.mix.account_api import AccountApi
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("BITGET_LIVE_KEY")
api_secret = os.getenv("BITGET_LIVE_SK")
passphrase = os.getenv("BITGET_LIVE_PASSPHRASE")

account_api = AccountApi(api_key, api_secret, passphrase)
params = {
    "productType": "usdt-futures",  # For live USDT-margined futures
    "marginCoin": "USDT"
}
result = account_api.accounts(params)
print(result) 