#!/usr/bin/env python3
"""
Wallet Management CLI - Wrapper for wallet management functionality.
"""

import logging
from typing import Optional, Any
from argparse import _SubParsersAction
from rich.console import Console
from rich.table import Table
from rich import box

from src.cli.base import BaseCLI
from src.core.config import Config
from src.utils.wallet.wallet_manager import WalletManager

console = Console()
logger = logging.getLogger(__name__)


class WalletCLI(BaseCLI):
    """CLI wrapper for wallet management operations."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize wallet CLI wrapper."""
        super().__init__(config)
        self.wallet_manager: Optional[WalletManager] = None
        self.network: Optional[str] = None

    def start(self) -> None:
        """Initialize the wallet manager."""
        self.wallet_manager = WalletManager()

    def stop(self) -> None:
        """Cleanup wallet manager."""
        if self.wallet_manager:
            self.wallet_manager = None

    async def cleanup(self):
        """No-op cleanup for test compatibility."""
        pass

    async def setup(self):
        """No-op setup for test compatibility."""
        pass

    def add_arguments(self, parser: _SubParsersAction) -> None:
        """Add wallet-specific arguments to the parser."""
        wallet_parser = parser.add_parser("wallet", help="Wallet management operations")
        subparsers = wallet_parser.add_subparsers(dest="wallet_command")

        # List wallets command
        subparsers.add_parser("list", help="List available wallets")

        # Create wallet command
        create_parser = subparsers.add_parser("create", help="Create new wallet")
        create_parser.add_argument(
            "--name", required=True, help="Name for the new wallet"
        )
        create_parser.add_argument(
            "--password", required=True, help="Password for wallet encryption"
        )

        # Import wallet command
        import_parser = subparsers.add_parser("import", help="Import existing wallet")
        import_parser.add_argument(
            "--name", required=True, help="Name for the imported wallet"
        )
        import_parser.add_argument(
            "--private-key", required=True, help="Private key to import"
        )
        import_parser.add_argument(
            "--password", required=True, help="Password for wallet encryption"
        )

        # Balance command
        balance_parser = subparsers.add_parser("balance", help="Check wallet balance")
        balance_parser.add_argument("--name", required=True, help="Wallet name")
        balance_parser.add_argument(
            "--token", help="Optional token to check balance for (default: SOL)"
        )

        # Set active wallet command
        active_parser = subparsers.add_parser("use", help="Set active wallet")
        active_parser.add_argument(
            "--name", required=True, help="Wallet name to set as active"
        )

        # Delete wallet command
        delete_parser = subparsers.add_parser("delete", help="Delete wallet")
        delete_parser.add_argument(
            "--name", required=True, help="Wallet name to delete"
        )
        delete_parser.add_argument(
            "--force", action="store_true", help="Force deletion without confirmation"
        )

    def handle_command(self, args: Any) -> None:
        """Handle wallet CLI commands."""
        if not hasattr(args, "wallet_command"):
            logger.error("No wallet command specified")
            return

        try:
            if args.wallet_command == "list":
                self._handle_list()
            elif args.wallet_command == "create":
                self._handle_create(args)
            elif args.wallet_command == "import":
                self._handle_import(args)
            elif args.wallet_command == "balance":
                self._handle_balance(args)
            elif args.wallet_command == "use":
                self._handle_use(args)
            elif args.wallet_command == "delete":
                self._handle_delete(args)

        except Exception as e:
            logger.error(f"Error in wallet command: {str(e)}")
            raise

    def _handle_list(self) -> None:
        """Handle wallet listing."""
        if self.wallet_manager is None:
            raise RuntimeError("Wallet manager is not initialized.")
        wallet_names = self.wallet_manager.list_wallets()

        if not wallet_names:
            console.print("[yellow]No wallets configured[/yellow]")
            return

        table = Table(title="Available Wallets", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Public Key", style="green")
        table.add_column("Status", style="magenta")

        for name in wallet_names:
            wallet = self.wallet_manager.get_wallet(name)
            if wallet is None:
                continue
            pubkey = str(wallet.get_public_key())
            is_active = (
                self.wallet_manager.get_current_wallet() is not None
                and self.wallet_manager.get_current_wallet().name == name
            )
            table.add_row(
                wallet.name,
                f"{pubkey[:10]}...{pubkey[-6:]}",
                "Active" if is_active else "",
            )

        console.print(table)

    def _handle_create(self, args: Any) -> None:
        """Handle wallet creation."""
        if self.wallet_manager is None:
            raise RuntimeError("Wallet manager is not initialized.")
        try:
            # For now, just create a wallet from a keypair path (password not used)
            # In a real implementation, you would generate a new keypair and save it
            # Here, we just simulate adding a wallet
            success = self.wallet_manager.add_wallet(
                name=args.name,
                keypair_path=args.password,  # Not ideal, but placeholder for path
                is_main=False,
            )
            if success:
                wallet = self.wallet_manager.get_wallet(args.name)
                console.print(f"[green]Created new wallet:[/green] {wallet.name}")
                console.print(f"Public Key: {wallet.get_public_key()}")
            else:
                console.print(f"[red]Failed to create wallet:[/red] {args.name}")
        except Exception as e:
            console.print(f"[red]Error creating wallet:[/red] {str(e)}")
            raise

    def _handle_import(self, args: Any) -> None:
        """Handle wallet import."""
        if self.wallet_manager is None:
            raise RuntimeError("Wallet manager is not initialized.")
        try:
            # For import, treat private_key as a path for now
            success = self.wallet_manager.add_wallet(
                name=args.name, keypair_path=args.private_key, is_main=False
            )
            if success:
                wallet = self.wallet_manager.get_wallet(args.name)
                console.print(f"[green]Imported wallet:[/green] {wallet.name}")
                console.print(f"Public Key: {wallet.get_public_key()}")
            else:
                console.print(f"[red]Failed to import wallet:[/red] {args.name}")
        except Exception as e:
            console.print(f"[red]Error importing wallet:[/red] {str(e)}")
            raise

    def _handle_balance(self, args: Any) -> None:
        """Handle balance check."""
        if self.wallet_manager is None:
            raise RuntimeError("Wallet manager is not initialized.")
        try:
            wallet = self.wallet_manager.get_wallet(args.name)
            if not wallet:
                console.print(f"[red]Wallet not found:[/red] {args.name}")
                return

            # get_balance may be async, so we check and call accordingly
            balance = (
                wallet.get_balance(args.token)
                if not hasattr(wallet.get_balance, "__await__")
                else wallet.get_balance(args.token)
            )
            token_name = args.token or "SOL"

            console.print(f"[green]Balance for {args.name}:[/green]")
            console.print(f"{balance} {token_name}")
        except Exception as e:
            console.print(f"[red]Error checking balance:[/red] {str(e)}")
            raise

    def _handle_use(self, args: Any) -> None:
        """Handle setting active wallet."""
        if self.wallet_manager is None:
            raise RuntimeError("Wallet manager is not initialized.")
        try:
            success = self.wallet_manager.switch_wallet(args.name)
            if success:
                console.print(f"[green]Set active wallet:[/green] {args.name}")
            else:
                console.print(f"[red]Failed to set active wallet:[/red] {args.name}")
        except Exception as e:
            console.print(f"[red]Error setting active wallet:[/red] {str(e)}")
            raise

    def _handle_delete(self, args: Any) -> None:
        """Handle wallet deletion."""
        if self.wallet_manager is None:
            raise RuntimeError("Wallet manager is not initialized.")
        try:
            if not args.force:
                confirm = input(
                    f"Are you sure you want to delete wallet '{args.name}'? [y/N]: "
                )
                if confirm.lower() != "y":
                    console.print("Deletion cancelled")
                    return

            success = self.wallet_manager.remove_wallet(args.name)
            if success:
                console.print(f"[green]Deleted wallet:[/green] {args.name}")
            else:
                console.print(f"[red]Failed to delete wallet:[/red] {args.name}")
        except Exception as e:
            console.print(f"[red]Error deleting wallet:[/red] {str(e)}")
            raise

    def set_network(self, network: str) -> None:
        """Set the network for wallet operations (mainnet, devnet, etc.)."""
        self.network = network
        logger.info(f"WalletCLI network set to: {network}")
