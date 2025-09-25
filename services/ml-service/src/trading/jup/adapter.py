"""
Simple Jupiter DEX adapter focused on core functionality
"""

import asyncio
import logging
import os
from typing import Dict, Optional, Any
from dotenv import load_dotenv

from solders.pubkey import Pubkey
from solana.rpc.types import TxOpts
from solana.transaction import Transaction

from src.trading.jup.client import JupiterClient
from src.trading.jup.auth import JupiterAuth

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

class JupiterAdapter:
    """
    Simplified Jupiter DEX adapter focusing on core functionality
    """
    
    def __init__(self, network="mainnet", keypair_path=None):
        """Initialize the JupiterAdapter.
        
        Args:
            network (str): The network to connect to ("mainnet" or "devnet")
            keypair_path (str): Path to the keypair file
        """
        load_dotenv()
        self.network = network
        self.keypair_path = keypair_path or os.getenv("DEVNET_KEYPAIR_PATH")
        
        self.auth = None
        self.client = None
        self.connected = False
        
        logger.info(f"JupiterAdapter initialized for {network}")
    
    async def connect(self) -> bool:
        """Connect to Jupiter and initialize client"""
        try:
            # Initialize authentication
            self.auth = JupiterAuth(
                network=self.network,
                keypair_path=self.keypair_path
            )
            
            # Authenticate
            await self.auth.authenticate()
            
            # Get initialized client
            self.client = self.auth.get_client()
            if not self.client:
                raise Exception("Failed to get initialized client")
            
            self.connected = True
            logger.info(f"Connected to {self.network}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def get_quote(self,
                     input_token: str,
                     output_token: str,
                     amount: int,
                     slippage_bps: int = 50) -> Dict[str, Any]:
        """
        Get a quote for swapping tokens
        
        Args:
            input_token: Input token mint address
            output_token: Output token mint address
            amount: Amount in input token's smallest units
            slippage_bps: Slippage tolerance in basis points (default 0.5%)
            
        Returns:
            Quote details
        """
        if not self.client:
            logger.error("Jupiter client not initialized")
            return None
        
        try:
            quote = await self.client.get_quote(
                input_mint=input_token,
                output_mint=output_token,
                amount=amount,
                slippage_bps=slippage_bps
            )
            
            return {
                "input_token": input_token,
                "output_token": output_token,
                "amount": amount,
                "quote": quote
            }
            
        except Exception as e:
            logger.error(f"Error getting quote: {str(e)}")
            raise
    
    async def swap(self,
                quote: Dict[str, Any],
                confirm: bool = True) -> Dict[str, Any]:
        """
        Execute a token swap based on a quote
        
        Args:
            quote: Quote object from get_quote()
            confirm: Whether to wait for transaction confirmation
            
        Returns:
            Transaction details
        """
        if not self.client:
            logger.error("Jupiter client not initialized")
            return None
        
        try:
            # Get swap transaction
            swap_response = await self.client.get_swap_transaction(quote["quote"])
            
            # Extract transaction data
            tx_data = swap_response["swapTransaction"]
            
            # Deserialize and sign transaction
            tx = Transaction.deserialize(bytes.fromhex(tx_data))
            tx.sign(self.client.keypair)
            
            # Send transaction
            opts = TxOpts(skip_preflight=True, preflight_commitment=None)
            tx_sig = await self.client.connection.send_transaction(
                tx,
                self.client.keypair,
                opts=opts
            )
            
            result = {
                "transaction": str(tx_sig),
                "input_token": quote["input_token"],
                "output_token": quote["output_token"],
                "amount": quote["amount"]
            }
            
            # Wait for confirmation if requested
            if confirm:
                await self.client.connection.confirm_transaction(tx_sig)
                logger.info(f"Transaction confirmed: {tx_sig}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing swap: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.auth:
            await self.auth.cleanup()
            self.auth = None
        self.client = None
        self.connected = False
        logger.info("Cleaned up JupiterAdapter resources") 