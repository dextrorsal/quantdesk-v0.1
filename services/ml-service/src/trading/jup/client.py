"""
Core Jupiter client implementation.
"""

import os
import logging
import json
from typing import Optional, Dict, Any
import aiohttp
from dotenv import load_dotenv

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair
from anchorpy import Wallet

from src.core.exceptions import ExchangeError, NotInitializedError

logger = logging.getLogger(__name__)

class JupiterClient:
    """
    Core Jupiter client for interacting with the Jupiter protocol.
    Handles client initialization, connection management, and token swaps.
    """
    
    def __init__(self, network: str = "mainnet", keypair_path: Optional[str] = None):
        """Initialize the Jupiter client."""
        load_dotenv()
        self.network = network.lower()
        
        # Try different keypair paths in order of priority
        self.keypair_path = (
            keypair_path or  # Explicitly provided path
            os.getenv("DEVNET_KEYPAIR_PATH") or  # Environment variable for devnet
            os.getenv("MAIN_KEY_PATH") or  # Main keypair path
            "/home/dex/.config/solana/keys/id.json"  # Default path
        )
        
        # Jupiter API endpoint
        self.api_endpoint = (
            os.getenv('JUPITER_API_ENDPOINT', "https://public.jupiterapi.com")
        )
        
        # Solana RPC endpoint
        self.rpc_url = (
            os.getenv('MAINNET_RPC_ENDPOINT', "https://api.mainnet-beta.solana.com")
            if network == 'mainnet' 
            else os.getenv('DEVNET_RPC_ENDPOINT', "https://api.devnet.solana.com")
        )
        
        self.connection: Optional[AsyncClient] = None
        self.keypair: Optional[Keypair] = None
        self.wallet: Optional[Wallet] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the Jupiter client connection."""
        if not self.keypair_path:
            raise ExchangeError("No keypair path provided and DEVNET_KEYPAIR_PATH environment variable not set")
            
        if not os.path.exists(self.keypair_path):
            raise ExchangeError(f"Keypair not found at {self.keypair_path}")
            
        try:
            # Load keypair
            with open(self.keypair_path, 'r') as f:
                keypair_bytes = bytes(json.load(f))
                
            self.keypair = Keypair.from_bytes(keypair_bytes)
            self.wallet = Wallet(self.keypair)
            logger.info(f"Loaded keypair from {self.keypair_path}")
            
            # Initialize Solana connection
            self.connection = AsyncClient(self.rpc_url)
            logger.info(f"Connected to Solana RPC at {self.rpc_url}")
            
            # Initialize HTTP session for Jupiter API
            self.session = aiohttp.ClientSession(base_url=self.api_endpoint)
            logger.info(f"Initialized Jupiter API session at {self.api_endpoint}")
            
            self.initialized = True
            logger.info("Successfully initialized Jupiter client")
            
        except Exception as e:
            logger.error(f"Failed to initialize Jupiter client: {e}")
            await self.cleanup()
            raise ExchangeError(f"Failed to initialize: {e}")
    
    async def get_quote(self, 
                       input_mint: str,
                       output_mint: str,
                       amount: int,
                       slippage_bps: int = 50) -> Dict[str, Any]:
        """
        Get a quote for swapping tokens.
        
        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount in input token's smallest units (e.g., lamports for SOL)
            slippage_bps: Slippage tolerance in basis points (default 0.5%)
            
        Returns:
            Quote details including price and route
        """
        if not self.initialized:
            raise NotInitializedError("Jupiter client not initialized")
            
        try:
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps
            }
            
            async with self.session.get("/quote", params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ExchangeError(f"Failed to get quote: {error_text}")
                    
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            raise
    
    async def get_swap_transaction(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a swap transaction for the given quote.
        
        Args:
            quote: Quote object from get_quote()
            
        Returns:
            Transaction details
        """
        if not self.initialized:
            raise NotInitializedError("Jupiter client not initialized")
            
        try:
            # Prepare swap request
            swap_request = {
                "quoteResponse": quote,
                "userPublicKey": str(self.wallet.public_key),
                "wrapUnwrapSOL": True  # Auto wrap/unwrap SOL
            }
            
            async with self.session.post("/swap", json=swap_request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ExchangeError(f"Failed to get swap transaction: {error_text}")
                    
                return await response.json()
                
        except Exception as e:
            logger.error(f"Error getting swap transaction: {str(e)}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup client resources."""
        if self.session:
            await self.session.close()
            self.session = None
            
        if self.connection:
            await self.connection.close()
            self.connection = None
        
        self.keypair = None
        self.wallet = None
        self.initialized = False 