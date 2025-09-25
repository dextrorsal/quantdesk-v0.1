#!/usr/bin/env python3
"""
Jupiter Trading CLI - Wrapper for Jupiter adapter functionality.

Handles all Jupiter trading operations including:
- Token swaps
- Route finding
- Price quotes
- Balance checking
- Market analysis
- Token verification
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
from src.trading.jup.jup_adapter import JupiterAdapter
from src.utils.wallet.wallet_manager import WalletManager

console = Console()
logger = logging.getLogger(__name__)

class JupiterCLI(BaseCLI):
    """CLI wrapper for Jupiter trading operations."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize Jupiter CLI wrapper."""
        super().__init__(config)
        self.adapter: Optional[JupiterAdapter] = None
        self.wallet_manager = WalletManager()
    
    async def start(self) -> None:
        """Initialize the Jupiter adapter."""
        self.adapter = JupiterAdapter(self.config)
        await self.adapter.initialize()
    
    async def stop(self) -> None:
        """Cleanup Jupiter adapter."""
        if self.adapter:
            await self.adapter.cleanup()
    
    def add_arguments(self, parser: _SubParsersAction) -> None:
        """Add Jupiter-specific arguments to the parser."""
        jupiter_parser = parser.add_parser("jupiter", help="Jupiter trading operations")
        subparsers = jupiter_parser.add_subparsers(dest="jupiter_command")
        
        # Quote command
        quote_parser = subparsers.add_parser("quote", help="Get a swap quote")
        quote_parser.add_argument("--from", dest="input_token", required=True, help="Input token")
        quote_parser.add_argument("--to", dest="output_token", required=True, help="Output token")
        quote_parser.add_argument("--amount", type=float, required=True, help="Amount to swap")
        quote_parser.add_argument("--wallet", help="Wallet to use for the quote")
        
        # Swap command
        swap_parser = subparsers.add_parser("swap", help="Execute a token swap")
        swap_parser.add_argument("--from", dest="input_token", required=True, help="Input token")
        swap_parser.add_argument("--to", dest="output_token", required=True, help="Output token")
        swap_parser.add_argument("--amount", type=float, required=True, help="Amount to swap")
        swap_parser.add_argument("--slippage", type=float, default=0.5, help="Maximum slippage percentage")
        swap_parser.add_argument("--wallet", required=True, help="Wallet to use for the swap")
        
        # Balance command
        balance_parser = subparsers.add_parser("balance", help="Check token balances")
        balance_parser.add_argument("--wallet", required=True, help="Wallet to check balances for")
        balance_parser.add_argument("--token", help="Specific token to check (optional)")
        
        # Routes command
        routes_parser = subparsers.add_parser("routes", help="Find available swap routes")
        routes_parser.add_argument("--from", dest="input_token", required=True, help="Input token")
        routes_parser.add_argument("--to", dest="output_token", required=True, help="Output token")
        routes_parser.add_argument("--amount", type=float, required=True, help="Amount to swap")
        routes_parser.add_argument("--limit", type=int, default=3, help="Maximum number of routes to show")
        
        # Market command
        market_parser = subparsers.add_parser("market", help="Get market information")
        market_parser.add_argument("--pair", required=True, help="Trading pair (e.g., SOL/USDC)")
        
        # Verify command
        verify_parser = subparsers.add_parser("verify", help="Verify token information")
        verify_parser.add_argument("--token", required=True, help="Token address or symbol to verify")
    
    async def handle_command(self, args: Any) -> None:
        """Handle Jupiter CLI commands."""
        if not hasattr(args, 'jupiter_command'):
            console.print("[red]No Jupiter command specified[/red]")
            return

        try:
            if args.jupiter_command == "quote":
                await self._handle_quote(args)
            elif args.jupiter_command == "swap":
                await self._handle_swap(args)
            elif args.jupiter_command == "balance":
                await self._handle_balance(args)
            elif args.jupiter_command == "routes":
                await self._handle_routes(args)
            elif args.jupiter_command == "market":
                await self._handle_market(args)
            elif args.jupiter_command == "verify":
                await self._handle_verify(args)
        except Exception as e:
            console.print(f"[red]Error in Jupiter command: {str(e)}[/red]")
            logger.error(f"Error in Jupiter command: {str(e)}")
            raise

    async def _handle_quote(self, args: Any) -> None:
        """Handle quote command with rich output."""
        if not self.adapter:
            await self.start()
            
        quote = await self.adapter.get_quote(args.input_token, args.output_token, args.amount)
        
        # Create a rich table for the quote
        table = Table(title=f"Swap Quote: {args.input_token} → {args.output_token}")
        table.add_column("Detail", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Input Amount", f"{args.amount} {args.input_token}")
        table.add_row("Output Amount", f"{quote['outAmount']} {args.output_token}")
        table.add_row("Price Impact", f"{quote.get('priceImpact', 'N/A')}%")
        table.add_row("Route Type", quote.get('routeType', 'N/A'))
        
        console.print(table)

    async def _handle_swap(self, args: Any) -> None:
        """Handle swap command with rich output."""
        if not self.adapter:
            await self.start()
            
        wallet = self.wallet_manager.get_wallet(args.wallet)
        if not wallet:
            console.print(f"[red]Error: Wallet '{args.wallet}' not found[/red]")
            return
            
        # First get a quote
        console.print("[yellow]Getting quote...[/yellow]")
        quote = await self.adapter.get_quote(args.input_token, args.output_token, args.amount)
        
        # Show quote and ask for confirmation
        table = Table(title="Swap Preview")
        table.add_column("Detail", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("From", f"{args.amount} {args.input_token}")
        table.add_row("To", f"{quote['outAmount']} {args.output_token}")
        table.add_row("Price Impact", f"{quote.get('priceImpact', 'N/A')}%")
        table.add_row("Slippage Tolerance", f"{args.slippage}%")
        
        console.print(table)
        console.print("\nExecuting swap...")
        
        result = await self.adapter.execute_swap(
            args.input_token,
            args.output_token,
            args.amount,
            args.slippage,
            wallet
        )
        
        console.print(Panel.fit(
            f"[green]Swap executed successfully![/green]\nTransaction: {result['signature']}",
            title="Swap Result"
        ))

    async def _handle_balance(self, args: Any) -> None:
        """Handle balance command with rich output."""
        if not self.adapter:
            await self.start()
            
        wallet = self.wallet_manager.get_wallet(args.wallet)
        if not wallet:
            console.print(f"[red]Error: Wallet '{args.wallet}' not found[/red]")
            return
            
        if args.token:
            balance = await self.adapter.get_balance(args.token, wallet)
            console.print(Panel(f"{balance} {args.token}", title=f"Balance for {args.wallet}"))
        else:
            balances = await self.adapter.get_all_balances(wallet)
            table = Table(title=f"Token Balances for {args.wallet}")
            table.add_column("Token", style="cyan")
            table.add_column("Amount", style="green", justify="right")
            
            for token, amount in balances.items():
                if amount > 0:  # Only show non-zero balances
                    table.add_row(token, f"{amount:,.8f}")
            
            console.print(table)

    async def _handle_routes(self, args: Any) -> None:
        """Handle routes command with rich output."""
        if not self.adapter:
            await self.start()
            
        routes = await self.adapter.get_routes(
            args.input_token,
            args.output_token,
            args.amount,
            args.limit
        )
        
        table = Table(title=f"Swap Routes: {args.input_token} → {args.output_token}")
        table.add_column("Route", style="cyan")
        table.add_column("Output", style="green", justify="right")
        table.add_column("Price Impact", style="yellow", justify="right")
        
        for route in routes:
            table.add_row(
                " → ".join(route['path']),
                f"{route['outAmount']} {args.output_token}",
                f"{route.get('priceImpact', 'N/A')}%"
            )
        
        console.print(table)

    async def _handle_market(self, args: Any) -> None:
        """Handle market command with rich output."""
        if not self.adapter:
            await self.start()
            
        market_info = await self.adapter.get_market_info(args.pair)
        
        table = Table(title=f"Market Info: {args.pair}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in market_info.items():
            table.add_row(key, str(value))
        
        console.print(table)

    async def _handle_verify(self, args: Any) -> None:
        """Handle verify command with rich output."""
        if not self.adapter:
            await self.start()
            
        token_info = await self.adapter.verify_token(args.token)
        
        panel = Panel(
            "\n".join([f"{k}: {v}" for k, v in token_info.items()]),
            title=f"Token Verification: {args.token}",
            border_style="green"
        )
        console.print(panel) 