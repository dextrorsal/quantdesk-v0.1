"""
Jupiter DEX adapter for live trading
Implements the specific functionality needed to interact with Jupiter Aggregator
"""

import logging
import asyncio
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import base64
import os
from pathlib import Path
import random
import argparse
from tabulate import tabulate
from src.utils.wallet.sol_rpc import get_solana_client
import aiohttp
from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.transaction import VersionedTransaction
from solana.rpc.commitment import Commitment
from dotenv import load_dotenv
from solders.pubkey import Pubkey
from spl.token.constants import TOKEN_PROGRAM_ID
from ...utils.wallet.sol_wallet import get_wallet
import requests

# Load environment variables
load_dotenv()

# Use devnet by default
DEFAULT_NETWORK = "devnet"
NETWORK_URLS = {
    "devnet": "https://api.devnet.solana.com",
    "mainnet": "https://api.mainnet-beta.solana.com"
}

# Initialize client in connect() instead of at module level
logger = logging.getLogger(__name__)

# Token addresses for different networks
NETWORK_TOKENS = {
    "mainnet": {
        "SOL": "So11111111111111111111111111111111111111112",
        "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    },
    "devnet": {
        "SOL": "So11111111111111111111111111111111111111112",  # SOL mint is the same on all networks
        "TEST": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"  # Devnet test token
    }
}

class JupiterAdapter:
    """
    Adapter for Jupiter DEX Aggregator
    Handles the specific logic for interacting with Jupiter APIs
    """
    
    def __init__(self, config_path: str = None, keypair_path: str = None, network: str = DEFAULT_NETWORK):
        """
        Initialize Jupiter adapter
        
        Args:
            config_path: Path to configuration file
            keypair_path: Path to Solana keypair file
            network: Network to use (devnet or mainnet)
        """
        self.config_path = config_path
        self.keypair_path = keypair_path
        self.network = network
        self.connected = False
        self.client = None
        self.wallet = None
        self._session = None
        
        # Define supported token pairs based on network
        if self.network == "devnet":
            # Devnet - SOL to TEST token
            self.markets = {
                "SOL-TEST": {
                    "input_mint": NETWORK_TOKENS["devnet"]["SOL"],
                    "output_mint": NETWORK_TOKENS["devnet"]["TEST"],
                    "decimals_in": 9,
                    "decimals_out": 9
                }
            }
        else:
            # Original mainnet token pairs
            self.markets = {
                "SOL-USDC": {
                    "input_mint": NETWORK_TOKENS["mainnet"]["SOL"],
                    "output_mint": NETWORK_TOKENS["mainnet"]["USDC"],
                    "decimals_in": 9,
                    "decimals_out": 6
                }
            }
        
        # Ultra API configuration
        self.ultra_config = {
            "slippage_bps": 50,  # 0.5%
            "priority_level": "veryHigh",
            "max_priority_fee_lamports": 10000000,  # 0.01 SOL
            "restrict_intermediate_tokens": True
        }
        
        self.tokens = NETWORK_TOKENS[self.network]
        self.base_url = "https://quote-api.jup.ag/v6"
        
        # Add trade logging
        self.log_dir = Path("data/trade_logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trade_log_file = self.log_dir / f"jup_trade_log_{datetime.now().strftime('%Y%m%d')}.json"
        
    async def connect(self) -> bool:
        """
        Connect to Jupiter
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up Jupiter adapter...")
            
            # Initialize Solana client
            self.client = await get_solana_client(self.network)
            version = await self.client.get_version()
            logger.info(f"Connected to Solana {self.network} node version: {version}")
            
            # Initialize wallet
            self.wallet = await get_wallet()
            logger.info(f"Loaded wallet with pubkey: {self.wallet.pubkey}")
            
            self.connected = True
            logger.info("Connected to Jupiter API successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Jupiter: {e}")
            self.connected = False
            return False
    
    @property
    def session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def get_market_price(self, market: str) -> float:
        """
        Get current market price from Jupiter quote API
        
        Args:
            market: Market symbol (e.g., "SOL-USDC")
            
        Returns:
            Current market price
        """
        if not self.connected:
            await self.connect()
            
        try:
            if market not in self.markets:
                raise ValueError(f"Unknown market: {market}")
                
            market_config = self.markets[market]
            
            # In a real implementation, this would call the Jupiter API
            # For the price, we'd need to get a quote for a small amount
            # url = f"{self.api_url}/quote"
            # params = {
            #     "inputMint": market_config["input_mint"],
            #     "outputMint": market_config["output_mint"],
            #     "amount": 1000000,  # 1 USDC in base units
            #     "slippageBps": 50
            # }
            # async with self.session.get(url, params=params) as response:
            #     if response.status != 200:
            #         raise Exception(f"API error: {await response.text()}")
            #     quote = await response.json()
            #     # Calculate price from quote
            #     input_amount = int(quote["inputAmount"])
            #     output_amount = int(quote["outputAmount"])
            #     input_decimals = market_config["decimals_in"]
            #     output_decimals = market_config["decimals_out"]
            #     price = (output_amount / 10**output_decimals) / (input_amount / 10**input_decimals)
            #     return price
            
            # Mock implementation for demonstration
            # Use similar prices to Drift but with a small spread
            base_prices = {
                "SOL-USDC": 80.15,
                "BTC-USDC": 42450.50,
                "ETH-USDC": 2272.25,
                "USDC-SOL": 1/80.15,
                "USDC-BTC": 1/42450.50,
                "USDC-ETH": 1/2272.25
            }
            
            # Add small random variation to simulate price movement
            base_price = base_prices.get(market, 100.0)
            variation = random.uniform(-0.5, 0.5) / 100  # -0.5% to +0.5%
            return base_price * (1 + variation)
            
        except Exception as e:
            logger.error(f"Error getting price for {market}: {e}")
            raise
    
    async def get_account_balances(self) -> Dict[str, float]:
        """
        Get account balances
        
        Returns:
            Dictionary of token balances
        """
        if not self.connected:
            await self.connect()
            
        try:
            # In a real implementation, this would query the Solana wallet
            # token_accounts = await get_token_accounts(self.connection, self.wallet.pubkey)
            # balances = {}
            # for token, account in token_accounts.items():
            #     balance_info = await self.connection.get_token_account_balance(account.pubkey)
            #     balances[token] = balance_info.value.ui_amount
            # return balances
            
            # Mock implementation for demonstration
            return {
                "USDC": 1000.0,
                "SOL": 10.0,
                "BTC": 0.02,
                "ETH": 0.5
            }
            
        except Exception as e:
            logger.error(f"Error getting account balances: {e}")
            raise
    
    async def get_ultra_quote(self, 
                            market: str, 
                            input_amount: float,
                            config: Optional[Dict] = None) -> Dict:
        """
        Get a quote using Jupiter Ultra API
        
        Args:
            market: Market symbol (e.g., "SOL-USDC")
            input_amount: Amount of input token
            config: Optional configuration overrides
            
        Returns:
            Quote details from Ultra API
        """
        try:
            if market not in self.markets:
                raise ValueError(f"Unknown market: {market}")
                
            market_config = self.markets[market]
            
            # Convert input amount to proper decimals
            amount_in_base_units = int(input_amount * (10 ** market_config["decimals_in"]))
            
            # Get wallet address as string
            wallet_address = str(self.wallet.pubkey)
            
            # Prepare request parameters exactly as per docs
            params = {
                "inputMint": market_config["input_mint"],
                "outputMint": market_config["output_mint"],
                "amount": str(amount_in_base_units),
                "taker": wallet_address
            }
            
            # Make API request to the Ultra order endpoint
            url = f"{self.ultra_api_url}/order"
            logger.info(f"Requesting Ultra order with params: {params}")
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ultra API order error: {error_text}")
                
                order_data = await response.json()
                
                # Log the response for debugging
                logger.debug(f"Ultra order response: {json.dumps(order_data, indent=2)}")
                
                return order_data
                
        except Exception as e:
            logger.error(f"Error getting Ultra order for {market}: {e}")
            raise

    async def execute_ultra_swap(self,
                               market: str,
                               input_amount: float,
                               config: Optional[Dict] = None) -> Dict:
        """
        Execute a swap using Jupiter Ultra API
        
        Args:
            market: Market symbol (e.g., "SOL-USDC")
            input_amount: Amount of input token to swap
            config: Optional configuration overrides
            
        Returns:
            Swap execution details
        """
        try:
            # Get quote first
            quote = await self.get_ultra_quote(market, input_amount, config)
            
            # Prepare swap request
            swap_request = {
                "quoteResponse": quote,
                "userPublicKey": str(self.wallet.pubkey),
                "priorityLevel": config.get("priority_level", self.ultra_config["priority_level"]),
                "maxPriorityFeeLamports": config.get("max_priority_fee_lamports",
                                                   self.ultra_config["max_priority_fee_lamports"])
            }
            
            # Get swap transaction
            url = f"{self.ultra_api_url}/execute"
            async with self.session.post(url, json=swap_request) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ultra API execute error: {error_text}")
                
                execute_response = await response.json()
                
                # Extract and deserialize the transaction
                transaction_base64 = execute_response["transaction"]
                transaction = VersionedTransaction.deserialize(
                    base64.b64decode(transaction_base64)
                )
                
                # Sign the transaction
                transaction.sign([self.wallet])
                
                # Serialize the signed transaction
                signed_transaction = base64.b64encode(
                    transaction.serialize()
                ).decode('utf-8')
                
                # Submit the signed transaction
                submit_request = {
                    "signedTransaction": signed_transaction,
                    "requestId": execute_response["requestId"]
                }
                
                # Submit to Jupiter's execution endpoint
                async with self.session.post(f"{self.ultra_api_url}/execute", json=submit_request) as submit_response:
                    if submit_response.status != 200:
                        error_text = await submit_response.text()
                        raise Exception(f"Transaction submission error: {error_text}")
                    
                    result = await submit_response.json()
                    
                    if result["status"] == "Success":
                        logger.info(f"Swap successful: {result['signature']}")
                        return {
                            "status": "success",
                            "signature": result["signature"],
                            "input_amount": result["inputAmountResult"],
                            "output_amount": result["outputAmountResult"],
                            "market": market,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    else:
                        raise Exception(f"Swap failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error executing Ultra swap for {market}: {e}")
            raise

    async def execute_swap(self, 
                          market: str, 
                          input_amount: float,
                          slippage_bps: int = 50) -> Dict:
        """
        Execute a token swap on Jupiter (using Ultra API)
        
        Args:
            market: Market symbol (e.g., "SOL-USDC")
            input_amount: Amount of input token to swap
            slippage_bps: Slippage tolerance in basis points (1 bps = 0.01%)
            
        Returns:
            Swap details
        """
        # Update to use Ultra API
        config = {**self.ultra_config, "slippage_bps": slippage_bps}
        return await self.execute_ultra_swap(market, input_amount, config)
    
    async def buy_with_usdc(self, token: str, usdc_amount: float) -> Dict:
        """
        Buy a token with USDC
        
        Args:
            token: Token to buy (e.g., "SOL", "BTC", "ETH")
            usdc_amount: Amount of USDC to spend
            
        Returns:
            Swap details
        """
        market = f"USDC-{token}"
        return await self.execute_swap(market, usdc_amount)
    
    async def sell_to_usdc(self, token: str, token_amount: float) -> Dict:
        """
        Sell a token for USDC
        
        Args:
            token: Token to sell (e.g., "SOL", "BTC", "ETH")
            token_amount: Amount of token to sell
            
        Returns:
            Swap details
        """
        market = f"{token}-USDC"
        return await self.execute_swap(market, token_amount)
    
    async def get_route_options(self, market: str, input_amount: float) -> List[Dict]:
        """
        Get available swap route options
        
        Args:
            market: Market symbol (e.g., "SOL-USDC")
            input_amount: Amount of input token
            
        Returns:
            List of route options with pricing
        """
        if not self.connected:
            await self.connect()
            
        try:
            if market not in self.markets:
                raise ValueError(f"Unknown market: {market}")
                
            market_config = self.markets[market]
            
            # In a real implementation, this would call the Jupiter API
            # url = f"{self.api_url}/quote"
            # params = {
            #     "inputMint": market_config["input_mint"],
            #     "outputMint": market_config["output_mint"],
            #     "amount": int(input_amount * 10**market_config["decimals_in"]),
            #     "slippageBps": 50,
            #     "onlyDirectRoutes": False
            # }
            # async with self.session.get(url, params=params) as response:
            #     if response.status != 200:
            #         raise Exception(f"API error: {await response.text()}")
            #     quote = await response.json()
            #     return quote.get("routesInfos", [])
            
            # Mock implementation for demonstration
            base_price = await self.get_market_price(market)
            
            # Generate some mock route options with slight variations
            routes = []
            for i in range(3):
                # Simulate different routes with different pricing
                price_variation = random.uniform(-0.2, 0.5) / 100  # Between -0.2% and +0.5%
                route_price = base_price * (1 + price_variation)
                
                # Generate mock route
                route = {
                    "routeIdx": i,
                    "inAmount": str(int(input_amount * 10**market_config["decimals_in"])),
                    "outAmount": str(int(input_amount * route_price * 10**market_config["decimals_out"])),
                    "outAmountWithSlippage": str(int(input_amount * route_price * 0.995 * 10**market_config["decimals_out"])),
                    "priceImpactPct": abs(price_variation) * 100,
                    "marketInfos": [
                        {
                            "id": f"mock-amm-{i+1}",
                            "label": f"Mock AMM {i+1}",
                            "inputMint": market_config["input_mint"],
                            "outputMint": market_config["output_mint"],
                            "inAmount": str(int(input_amount * 10**market_config["decimals_in"])),
                            "outAmount": str(int(input_amount * route_price * 10**market_config["decimals_out"])),
                            "lpFee": {"amount": "0.3%", "percent": 0.3}
                        }
                    ],
                    "amount": str(int(input_amount * 10**market_config["decimals_in"])),
                    "slippageBps": 50,
                    "otherAmountThreshold": str(int(input_amount * route_price * 0.995 * 10**market_config["decimals_out"]))
                }
                routes.append(route)
                
            return routes
            
        except Exception as e:
            logger.error(f"Error getting route options for {market}: {e}")
            raise
    
    async def disconnect(self):
        """
        Disconnect from Jupiter
        """
        if self.connected:
            logger.info("Disconnecting from Jupiter API")
            self.connected = False
            self.client = None
            if self._session and not self._session.closed:
                await self._session.close()
            self._session = None
    
    async def close(self):
        """Close the adapter and cleanup resources"""
        if hasattr(self, '_session') and self._session is not None:
            await self._session.close()
            self._session = None
        logger.info("Closed Jupiter adapter session")
    
    def __del__(self):
        """
        Clean up resources when the adapter is garbage collected
        """
        if self._session and not self._session.closed:
            asyncio.create_task(self._session.close())
        self.connected = False
        self.client = None
        self._session = None

    async def log_trade(self, trade_details: Dict):
        """Log trade details to file"""
        trade_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **trade_details
        }
        
        with open(self.trade_log_file, "a") as f:
            f.write(json.dumps(trade_log) + "\n")
    
    async def get_route_metrics(self, market: str, amount: float) -> Dict:
        """
        Get comprehensive route metrics for a potential swap
        
        Args:
            market: Market symbol (e.g., "SOL-USDC")
            amount: Input amount
            
        Returns:
            Dictionary with route metrics
        """
        routes = await self.get_route_options(market, amount)
        if not routes:
            return {}
            
        best_route = routes[0]
        return {
            "price": best_route["price"],
            "price_impact_pct": best_route.get("priceImpactPct", 0),
            "min_output": best_route.get("minOutputAmount", 0),
            "route_count": len(routes),
            "best_route_hops": len(best_route.get("routePlan", [])),
            "fee_estimates": {
                "platform_fee": best_route.get("platformFee", 0),
                "network_fee": best_route.get("networkFee", 0)
            }
        }
    
    @classmethod
    async def cli(cls):
        """CLI interface for Jupiter adapter"""
        parser = argparse.ArgumentParser(description="Jupiter DEX CLI")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Price command
        price_parser = subparsers.add_parser("price", help="Get market price")
        price_parser.add_argument("market", help="Market symbol (e.g., SOL-USDC)")
        
        # Balance command
        balance_parser = subparsers.add_parser("balance", help="Get account balances")
        
        # Quote command
        quote_parser = subparsers.add_parser("quote", help="Get swap quote")
        quote_parser.add_argument("market", help="Market symbol (e.g., SOL-USDC)")
        quote_parser.add_argument("amount", type=float, help="Input amount")
        
        # Route command
        route_parser = subparsers.add_parser("route", help="Get route metrics")
        route_parser.add_argument("market", help="Market symbol (e.g., SOL-USDC)")
        route_parser.add_argument("amount", type=float, help="Input amount")
        
        # Swap command
        swap_parser = subparsers.add_parser("swap", help="Execute token swap")
        swap_parser.add_argument("market", help="Market symbol (e.g., SOL-USDC)")
        swap_parser.add_argument("amount", type=float, help="Input amount")
        swap_parser.add_argument("--slippage", type=int, default=50, help="Slippage in basis points (default: 50)")
        
        # Monitor command
        monitor_parser = subparsers.add_parser("monitor", help="Monitor market prices")
        monitor_parser.add_argument("--markets", nargs="+", default=["SOL-USDC", "BTC-USDC", "ETH-USDC"],
                                  help="Markets to monitor")
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
            
        adapter = cls()
        await adapter.connect()
        
        try:
            if args.command == "price":
                price = await adapter.get_market_price(args.market)
                print(f"\nMarket: {args.market}")
                print(f"Price: ${price:,.2f}")
                
            elif args.command == "balance":
                balances = await adapter.get_account_balances()
                balance_data = [[
                    token,
                    f"{amount:.6f}",
                    f"${await adapter.get_market_price(f'{token}-USDC') * amount:.2f}" if token != "USDC" else f"${amount:.2f}"
                ] for token, amount in balances.items()]
                print("\nAccount Balances:")
                print(tabulate(balance_data, headers=['Token', 'Amount', 'Value (USD)'], tablefmt='simple'))
                
            elif args.command == "quote":
                routes = await adapter.get_route_options(args.market, args.amount)
                print(f"\nQuotes for {args.amount} {args.market.split('-')[0]}:")
                for i, route in enumerate(routes[:3], 1):
                    print(f"\nRoute {i}:")
                    print(f"Price: ${route['price']:.2f}")
                    print(f"Price Impact: {route['priceImpactPct']:.2f}%")
                    print(f"Min Output: {route['minOutputAmount']}")
                    
            elif args.command == "route":
                metrics = await adapter.get_route_metrics(args.market, args.amount)
                print(f"\nRoute Metrics for {args.amount} {args.market.split('-')[0]}:")
                print(f"Best Price: ${metrics['price']:.2f}")
                print(f"Price Impact: {metrics['price_impact_pct']:.2f}%")
                print(f"Available Routes: {metrics['route_count']}")
                print(f"Best Route Hops: {metrics['best_route_hops']}")
                print("\nFee Estimates:")
                print(f"Platform Fee: {metrics['fee_estimates']['platform_fee']}")
                print(f"Network Fee: {metrics['fee_estimates']['network_fee']}")
                
            elif args.command == "swap":
                print(f"\nExecuting swap: {args.amount} {args.market.split('-')[0]}")
                print(f"Slippage: {args.slippage} bps")
                
                confirm = input("\nConfirm swap? [y/N]: ")
                if confirm.lower() != 'y':
                    print("Swap cancelled")
                    return
                    
                result = await adapter.execute_swap(
                    market=args.market,
                    input_amount=args.amount,
                    slippage_bps=args.slippage
                )
                
                print("\nSwap executed successfully:")
                print(f"Input: {args.amount} {args.market.split('-')[0]}")
                print(f"Output: {result['outputAmount']}")
                print(f"Transaction: {result['transaction']}")
                
            elif args.command == "monitor":
                print(f"\nMonitoring prices for: {', '.join(args.markets)}")
                print("Press Ctrl+C to exit...")
                
                last_prices = {}
                while True:
                    prices = {}
                    for market in args.markets:
                        prices[market] = await adapter.get_market_price(market)
                    
                    print("\033[2J\033[H")  # Clear screen
                    price_data = [[
                        market,
                        f"${price:,.2f}",
                        "🔼" if price > last_prices.get(market, 0) else "🔽"
                    ] for market, price in prices.items()]
                    
                    print(tabulate(price_data, headers=['Market', 'Price', 'Move'], tablefmt='simple'))
                    last_prices = prices.copy()
                    await asyncio.sleep(1)
                    
        except KeyboardInterrupt:
            print("\nExiting...")
        except Exception as e:
            logger.error(f"Error: {str(e)}")
        finally:
            await adapter.close()

if __name__ == "__main__":
    asyncio.run(JupiterAdapter.cli())