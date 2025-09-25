#!/usr/bin/env python3
"""
Devnet Trading CLI - Wrapper for devnet testing operations.

Handles all devnet-specific operations including:
- Account creation and funding
- Airdrop requests
- Test token minting
- Market simulation
- Test trade execution
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from argparse import _SubParsersAction
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.cli.base import BaseCLI
from src.core.config import Config
from src.trading.devnet.devnet_adapter import DevnetAdapter
from src.utils.wallet.wallet_manager import WalletManager

console = Console()
logger = logging.getLogger(__name__)

class DevnetCLI(BaseCLI):
    """CLI wrapper for devnet testing operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Devnet CLI wrapper."""
        super().__init__(config)
        self.adapter: Optional[DevnetAdapter] = None
        self.wallet_manager = WalletManager()
    
    async def start(self) -> None:
        """Initialize the Devnet adapter."""
        self.adapter = DevnetAdapter(self.config)
        await self.adapter.initialize()
    
    async def stop(self) -> None:
        """Cleanup Devnet adapter."""
        if self.adapter:
            await self.adapter.cleanup()
    
    def add_arguments(self, parser: _SubParsersAction) -> None:
        """Add Devnet-specific arguments to the parser."""
        devnet_parser = parser.add_parser("devnet", help="Devnet testing operations")
        subparsers = devnet_parser.add_subparsers(dest="devnet_command")
        
        # Airdrop command
        airdrop_parser = subparsers.add_parser("airdrop", help="Request SOL airdrop")
        airdrop_parser.add_argument("--wallet", required=True, help="Wallet to receive airdrop")
        airdrop_parser.add_argument("--amount", type=float, default=1.0, help="Amount of SOL to request")
        
        # Create test token command
        token_parser = subparsers.add_parser("create-token", help="Create test token")
        token_parser.add_argument("--name", required=True, help="Token name")
        token_parser.add_argument("--symbol", required=True, help="Token symbol")
        token_parser.add_argument("--decimals", type=int, default=9, help="Token decimals")
        token_parser.add_argument("--wallet", required=True, help="Wallet to create token with")
        
        # Mint tokens command
        mint_parser = subparsers.add_parser("mint", help="Mint test tokens")
        mint_parser.add_argument("--token", required=True, help="Token address or symbol")
        mint_parser.add_argument("--amount", type=float, required=True, help="Amount to mint")
        mint_parser.add_argument("--to-wallet", required=True, help="Recipient wallet")
        mint_parser.add_argument("--authority-wallet", required=True, help="Mint authority wallet")
        
        # Create market command
        market_parser = subparsers.add_parser("create-market", help="Create test market")
        market_parser.add_argument("--base-token", required=True, help="Base token address or symbol")
        market_parser.add_argument("--quote-token", required=True, help="Quote token address or symbol")
        market_parser.add_argument("--wallet", required=True, help="Wallet to create market with")
        
        # Test trade command
        trade_parser = subparsers.add_parser("test-trade", help="Execute test trade")
        trade_parser.add_argument("--market", required=True, help="Market address or symbol")
        trade_parser.add_argument("--side", choices=["buy", "sell"], required=True, help="Trade side")
        trade_parser.add_argument("--amount", type=float, required=True, help="Trade amount")
        trade_parser.add_argument("--wallet", required=True, help="Trading wallet")
    
    async def handle_command(self, args: Any) -> None:
        """Handle Devnet CLI commands."""
        if not hasattr(args, 'devnet_command'):
            console.print("[red]No devnet command specified[/red]")
            return

        try:
            if args.devnet_command == "airdrop":
                await self._handle_airdrop(args)
            elif args.devnet_command == "create-token":
                await self._handle_create_token(args)
            elif args.devnet_command == "mint":
                await self._handle_mint(args)
            elif args.devnet_command == "create-market":
                await self._handle_create_market(args)
            elif args.devnet_command == "test-trade":
                await self._handle_test_trade(args)
        except Exception as e:
            console.print(f"[red]Error in devnet command: {str(e)}[/red]")
            logger.error(f"Error in devnet command: {str(e)}")
            raise

    async def _handle_airdrop(self, args: Any) -> None:
        """Handle airdrop command with rich output."""
        if not self.adapter:
            await self.start()
            
        wallet = self.wallet_manager.get_wallet(args.wallet)
        if not wallet:
            console.print(f"[red]Error: Wallet '{args.wallet}' not found[/red]")
            return
            
        console.print(f"[yellow]Requesting {args.amount} SOL airdrop for {args.wallet}...[/yellow]")
        result = await self.adapter.request_airdrop(wallet, args.amount)
        
        console.print(Panel.fit(
            f"[green]Airdrop successful![/green]\nTransaction: {result['signature']}",
            title="Airdrop Result"
        ))

    async def _handle_create_token(self, args: Any) -> None:
        """Handle create-token command with rich output."""
        if not self.adapter:
            await self.start()
            
        wallet = self.wallet_manager.get_wallet(args.wallet)
        if not wallet:
            console.print(f"[red]Error: Wallet '{args.wallet}' not found[/red]")
            return
            
        console.print(f"[yellow]Creating test token {args.symbol}...[/yellow]")
        result = await self.adapter.create_test_token(
            args.name,
            args.symbol,
            args.decimals,
            wallet
        )
        
        table = Table(title="Token Created")
        table.add_column("Detail", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Token Name", args.name)
        table.add_row("Symbol", args.symbol)
        table.add_row("Decimals", str(args.decimals))
        table.add_row("Mint Address", result["mint_address"])
        table.add_row("Transaction", result["signature"])
        
        console.print(table)

    async def _handle_mint(self, args: Any) -> None:
        """Handle mint command with rich output."""
        if not self.adapter:
            await self.start()
            
        to_wallet = self.wallet_manager.get_wallet(args.to_wallet)
        authority_wallet = self.wallet_manager.get_wallet(args.authority_wallet)
        if not to_wallet or not authority_wallet:
            console.print("[red]Error: Wallet not found[/red]")
            return
            
        console.print(f"[yellow]Minting {args.amount} {args.token} to {args.to_wallet}...[/yellow]")
        result = await self.adapter.mint_test_tokens(
            args.token,
            args.amount,
            to_wallet,
            authority_wallet
        )
        
        console.print(Panel.fit(
            f"[green]Tokens minted successfully![/green]\nTransaction: {result['signature']}",
            title="Mint Result"
        ))

    async def _handle_create_market(self, args: Any) -> None:
        """Handle create-market command with rich output."""
        if not self.adapter:
            await self.start()
            
        wallet = self.wallet_manager.get_wallet(args.wallet)
        if not wallet:
            console.print(f"[red]Error: Wallet '{args.wallet}' not found[/red]")
            return
            
        console.print(f"[yellow]Creating market {args.base_token}/{args.quote_token}...[/yellow]")
        result = await self.adapter.create_test_market(
            args.base_token,
            args.quote_token,
            wallet
        )
        
        table = Table(title="Market Created")
        table.add_column("Detail", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Market Address", result["market_address"])
        table.add_row("Base Token", args.base_token)
        table.add_row("Quote Token", args.quote_token)
        table.add_row("Transaction", result["signature"])
        
        console.print(table)

    async def _handle_test_trade(self, args: Any) -> None:
        """Handle test-trade command with rich output."""
        if not self.adapter:
            await self.start()
            
        wallet = self.wallet_manager.get_wallet(args.wallet)
        if not wallet:
            console.print(f"[red]Error: Wallet '{args.wallet}' not found[/red]")
            return
            
        console.print(f"[yellow]Executing test trade on {args.market}...[/yellow]")
        result = await self.adapter.execute_test_trade(
            args.market,
            args.side,
            args.amount,
            wallet
        )
        
        table = Table(title="Test Trade Executed")
        table.add_column("Detail", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Market", args.market)
        table.add_row("Side", args.side.upper())
        table.add_row("Amount", str(args.amount))
        table.add_row("Price", str(result["price"]))
        table.add_row("Transaction", result["signature"])
        
        console.print(table) 