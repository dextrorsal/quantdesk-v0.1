"""
Authentication handler for Jupiter.
"""

import logging
from typing import Optional

from src.trading.jup.client import JupiterClient
from src.core.exceptions import ExchangeError

logger = logging.getLogger(__name__)

class JupiterAuth:
    """
    Authentication handler for Jupiter.
    Manages client initialization and authentication.
    """
    
    def __init__(self, network: str = "mainnet", keypair_path: Optional[str] = None):
        """Initialize the Jupiter authentication handler."""
        self.network = network
        self.keypair_path = keypair_path
        self.client: Optional[JupiterClient] = None
        
    async def authenticate(self) -> None:
        """Initialize and authenticate the Jupiter client."""
        try:
            # Create and initialize client
            self.client = JupiterClient(
                network=self.network,
                keypair_path=self.keypair_path
            )
            
            await self.client.initialize()
            
        except Exception as e:
            logger.error(f"Failed to authenticate: {e}")
            await self.cleanup()
            raise ExchangeError(f"Authentication failed: {e}")
    
    def get_client(self) -> Optional[JupiterClient]:
        """Get the initialized client."""
        return self.client
    
    async def cleanup(self) -> None:
        """Cleanup authentication resources."""
        if self.client:
            await self.client.cleanup()
            self.client = None 