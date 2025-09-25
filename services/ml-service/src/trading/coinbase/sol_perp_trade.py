"""
Script to set up a SOL-PERP long position on Coinbase with 20x leverage.
"""

import os
import asyncio
import logging
from dotenv import load_dotenv
from src.exchanges.coinbase.coinbase_perp import CoinbasePerp

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_sol_perp_long():
    # Load environment variables
    load_dotenv()
    
    # Initialize CoinbasePerp with credentials from .env
    api_key = os.getenv("COINBASE_API_KEY")
    api_secret = os.getenv("COINBASE_API_SECRET")
    
    if not api_key or not api_secret:
        raise ValueError("Missing Coinbase API credentials in .env file")
    
    # Initialize the perpetual futures handler
    perp = CoinbasePerp(api_key, api_secret)
    
    try:
        # 1. Get account summary to check available collateral
        logger.info("Fetching account summary...")
        account = perp.get_account_summary()
        logger.info(f"Account summary: {account}")
        
        # 2. Get perpetuals portfolio
        logger.info("Fetching perpetuals portfolio...")
        portfolios = perp.get_perpetuals_portfolio()
        logger.info(f"Perpetuals portfolio: {portfolios}")
        
        # 3. Enable multi-asset collateral to use SOL as collateral
        logger.info("Enabling multi-asset collateral...")
        multi_asset = perp.opt_in_multi_asset_collateral()
        logger.info(f"Multi-asset collateral status: {multi_asset}")
        
        # 4. Place the SOL-PERP long order with 20x leverage
        # Using 1 SOL worth of position size initially
        logger.info("Placing SOL-PERP long order...")
        order = perp.create_perp_order(
            product_id="SOL-PERP",
            side="BUY",
            leverage=20.0,  # 20x leverage as requested
            size=1.0,  # Starting with 1 SOL position size
            order_type="MARKET"  # Using market order for immediate execution
        )
        logger.info(f"Order placed: {order}")
        
        # 5. Get position information to confirm
        logger.info("Fetching position information...")
        position = perp.get_position("SOL-PERP")
        logger.info(f"Current position: {position}")
        
        return {
            "status": "success",
            "order": order,
            "position": position
        }
        
    except Exception as e:
        logger.error(f"Error setting up SOL-PERP long position: {e}")
        raise

if __name__ == "__main__":
    try:
        result = asyncio.run(setup_sol_perp_long())
        logger.info("Successfully set up SOL-PERP long position!")
    except Exception as e:
        logger.error(f"Failed to set up position: {e}")