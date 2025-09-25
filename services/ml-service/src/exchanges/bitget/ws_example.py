import os
from dotenv import load_dotenv
from .bitget.ws.bitget_ws_client import BitgetWsClient, SubscribeReq
from .bitget import consts as c

load_dotenv()
api_key = os.getenv("BITGET_DEMO_KEY")
secret_key = os.getenv("BITGET_DEMO_SK")
passphrase = os.getenv("BITGET_DEMO_PASSPHRASE")


def handle(message):
    print("handle:", message)


def handle_error(message):
    print("handle_error:", message)


if __name__ == '__main__':
    client = BitgetWsClient(c.CONTRACT_WS_URL, need_login=True) \
        .api_key(api_key) \
        .api_secret_key(secret_key) \
        .passphrase(passphrase) \
        .error_listener(handle_error) \
        .build()

    channels = [SubscribeReq("mc", "ticker", "BTCUSD")]
    client.subscribe(channels, handle) 