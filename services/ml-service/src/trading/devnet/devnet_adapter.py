#!/usr/bin/env python3
"""
Devnet Adapter for Testing
(Jupiter logic only; Drift logic removed)
"""

import asyncio
import logging
import os
import json
import time
import sys
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from dotenv import load_dotenv
from tabulate import tabulate
from decimal import Decimal

from anchorpy.provider import Provider, Wallet
from solders.keypair import Keypair as SoldersKeypair
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed

# Drift logic removed
# from driftpy.drift_client import DriftClient
# from driftpy.account_subscription_config import AccountSubscriptionConfig
# from driftpy.constants.numeric_constants import BASE_PRECISION, QUOTE_PRECISION
# from driftpy.types import TxParams

from ...utils.wallet.wallet_manager import WalletManager
from ...utils.wallet.sol_rpc import get_solana_client
from src.trading.security.security_manager import SecurityManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Token configuration for Devnet
TOKEN_INFO = {
    "SOL": {
        "mint": "So11111111111111111111111111111111111111112",
        "decimals": 9,
        "symbol": "SOL"
    },
    "USDC": {
        "mint": "4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU",  # Devnet USDC
        "decimals": 6,
        "symbol": "USDC"
    },
    "TEST": {
        "mint": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs",  # Devnet test token
        "decimals": 9,
        "symbol": "TEST"
    }
}

# Create a shim for solana.keypair
import sys
from solders.keypair import Keypair as SoldersKeypair

class MockKeypairModule:
    def __init__(self):
        self.Keypair = SoldersKeypair

sys.modules['solana.keypair'] = MockKeypairModule()

class DevnetAdapter:
    """
    Unified adapter for devnet testing (Drift logic removed)
    """
    def __init__(self):
        self.rpc_endpoint = os.getenv('DEVNET_RPC_ENDPOINT')
        if not self.rpc_endpoint:
            raise EnvironmentError("DEVNET_RPC_ENDPOINT environment variable not set")
        self.wallet_manager = WalletManager()
        self.solana_client = None
        self.security_manager = SecurityManager()
        logger.info("DevnetAdapter initialized (Drift logic removed)")
    # All Drift methods and attributes removed
    # Only Jupiter or generic logic should remain 