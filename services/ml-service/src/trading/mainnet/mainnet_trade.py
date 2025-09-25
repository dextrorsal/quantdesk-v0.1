#!/usr/bin/env python3
"""
Mainnet Trading Script
(Jupiter logic only; Drift logic removed)
"""

import asyncio
import logging
import argparse
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Import security module
from src.trading.mainnet.security_limits import SecurityLimits

# Import existing infrastructure
from src.utils.wallet.wallet_cli import WalletCLI
# from src.trading.drift.drift_adapter import DriftAdapter  # Drift logic removed
from src.trading.jup.adapter import JupiterAdapter, SOL_MINT, USDC_MINT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

MAINNET_RPC_ENDPOINT = os.getenv("MAINNET_RPC_ENDPOINT")
MAIN_KEY_PATH = os.getenv("MAIN_KEY_PATH")
KP_PATH = os.getenv("KP_PATH")
AG_PATH = os.getenv("AG_PATH")

if not MAINNET_RPC_ENDPOINT:
    logger.warning("MAINNET_RPC_ENDPOINT not set, using default public endpoint")

# All Drift logic, usage examples, and config removed
# Only Jupiter or generic mainnet logic should remain
