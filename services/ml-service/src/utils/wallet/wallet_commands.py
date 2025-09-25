#!/usr/bin/env python3
"""
Unified Solana wallet management system.
Combines wallet operations, information display, and configuration management.
"""

import click
import logging
import asyncio
import os
import json
import base58
import stat
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
from dotenv import load_dotenv

# Solana imports
from solders.pubkey import Pubkey
from solders.keypair import Keypair
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.system_program import transfer, TransferParams
from solana.transaction import Transaction

# Local imports
from .wallet_manager import WalletManager
from .sol_wallet import SolanaWallet
from .price_service import get_price_service, PriceService
from .wallet_migration import WalletMigration
from .watch_mode_gui import launch_watch_mode
from .sol_rpc import get_solana_client, set_network, get_network, NETWORK_URLS

# Load environment variables
load_dotenv()

class WalletCommands:
    """Unified wallet management system"""
    
    def __init__(self):
        """Initialize wallet management system"""
        # Core components
        self.wallet_manager = WalletManager()
        self.migration = WalletMigration()
        self.price_service = get_price_service()
        
        # Configuration
        self.setup_logging()
        
        # Token configuration from wallet_cli.py
        self.TOKEN_INFO = {
            "SOL": {
                "decimals": 9,
                "emoji": "◎ ",
                "color": "bright_cyan",
                "name": "Solana",
                "mint": "So11111111111111111111111111111111111111112"
            },
            "USDC": {
                "decimals": 6,
                "emoji": "💵",
                "color": "bright_cyan",
                "name": "USD Coin",
                "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
            },
            "BTC": {
                "decimals": 8,
                "emoji": "₿",
                "color": "bright_yellow",
                "name": "Bitcoin",
                "mint": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E"
            },
            "ETH": {
                "decimals": 8,
                "emoji": "Ξ",
                "color": "blue",
                "name": "Ethereum",
                "mint": "7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs"
            }
        }
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def validate_keypair_path(self, path: str) -> bool:
        """Validate keypair file exists and has secure permissions"""
        try:
            path = os.path.expanduser(path)
            if not os.path.exists(path):
                self.logger.error(f"Keypair file not found: {path}")
                return False
                
            # Check file permissions (should be 600)
            st = os.stat(path)
            if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
                self.logger.error(f"Insecure keypair file permissions: {path}")
                self.logger.error("Please run: chmod 600 <keypair_file>")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error validating keypair path: {e}")
            return False
            
    def format_address(self, address: str) -> str:
        """Format wallet address for display"""
        if len(address) <= 12:
            return address
        return click.style(f"{address[:6]}", fg="bright_white") + \
               click.style("...", fg="white") + \
               click.style(f"{address[-6:]}", fg="bright_white")
               
    def format_amount(self, amount: float, token: str = "SOL", 
                     usd_price: Optional[float] = None, 
                     cad_price: Optional[float] = None) -> str:
        """Format token amount with color and fiat values"""
        if token == "SOL":
            base_text = click.style(
                f"{self.TOKEN_INFO[token]['emoji']}{amount:.4f} ",
                fg=self.TOKEN_INFO[token]['color'],
                bold=True
            ) + click.style(self.TOKEN_INFO[token]['name'], 
                          fg=self.TOKEN_INFO[token]['color'])
            
            if usd_price is not None and cad_price is not None:
                usd_value = amount * usd_price
                cad_value = amount * cad_price
                
                usd_text = (
                    click.style("(", fg="white") +
                    click.style(f"${usd_value:,.2f} USD", fg="white", bold=True) +
                    click.style(")", fg="white")
                )
                
                cad_text = (
                    click.style("(", fg="white") +
                    click.style(f"${cad_value:,.2f}", fg="white", bold=True, italic=True) +
                    " 🍁" +
                    click.style(")", fg="white")
                )
                
                return base_text + " " + usd_text + click.style(" | ", fg="white") + cad_text
                
            return base_text
        return f"{amount:.4f} {token}"
        
    async def get_token_balance(self, client: AsyncClient, wallet_pubkey: Pubkey, 
                              token_mint: str, decimals: int) -> float:
        """Get balance for a specific token"""
        try:
            if token_mint == self.TOKEN_INFO["SOL"]["mint"]:
                response = await client.get_balance(wallet_pubkey, commitment=Confirmed)
                return response.value / 10**decimals
            else:
                response = await client.get_token_accounts_by_owner_json_parsed(
                    wallet_pubkey,
                    {'mint': token_mint}
                )
                
                total_balance = 0
                if response.value:
                    for account in response.value:
                        try:
                            parsed_info = account.account.data.parsed['info']
                            if 'tokenAmount' in parsed_info:
                                amount = float(parsed_info['tokenAmount']['uiAmount'] or 0)
                                total_balance += amount
                        except (KeyError, TypeError, ValueError) as e:
                            self.logger.debug(f"Error parsing token amount: {str(e)}")
                            continue
                            
                return total_balance
                
        except Exception as e:
            self.logger.debug(f"Error getting token balance: {str(e)}")
            return 0
            
    # Migration Commands
    def migrate_to_encrypted(self, password: str):
        """Encrypt wallet configurations"""
        if self.migration.backup_existing_configs():
            successful, total = self.migration.migrate_to_encrypted(password)
            if successful == total:
                click.echo("✅ All configurations encrypted successfully")
            else:
                click.echo(f"⚠️  {successful}/{total} configurations encrypted")
                
    def backup_configs(self):
        """Create backup of configurations"""
        if self.migration.backup_existing_configs():
            click.echo("✅ Backup created successfully")
            
    def restore_from_backup(self, timestamp: str):
        """Restore configurations from backup"""
        if self.migration.restore_backup(timestamp):
            click.echo("✅ Configurations restored successfully")
        else:
            click.echo("❌ Failed to restore configurations")
            
    def verify_encryption(self, password: str):
        """Verify encrypted configurations"""
        problems = self.migration.verify_migration(password)
        if not problems:
            click.echo("✅ All configurations verified successfully")
        else:
            click.echo("❌ Problems found with these wallets:")
            for wallet in problems:
                click.echo(f"  - {wallet}")

    def show_wallet_formats(self, wallet_name: str, keypair_path: str):
        """Display wallet information in multiple formats"""
        if not os.path.exists(keypair_path):
            click.echo(f"Error: Keypair not found at {keypair_path}")
            return
        
        click.echo(f"\n{'='*20} {wallet_name} {'='*20}")
        click.echo(f"Using keypair from: {keypair_path}")
        
        try:
            # Method 1: Extract private key as base58
            with open(keypair_path, 'r') as f:
                keypair_json = json.load(f)
            
            keypair_bytes = bytes(keypair_json)
            private_key_bytes = keypair_bytes[:32]
            base58_private_key = base58.b58encode(private_key_bytes).decode('utf-8')
            
            click.echo("\n=== METHOD 1: First 32 bytes as base58 (for Phantom) ===")
            click.echo(f"Private key: {base58_private_key}")
            
            # Method 2: Get BIP39 seed phrase
            temp_file = f"/tmp/phantom_temp_{wallet_name.lower().replace(' ', '_')}.json"
            with open(keypair_path, 'r') as src, open(temp_file, 'w') as dst:
                dst.write(src.read())
                
            result = subprocess.run(
                ["solana-keygen", "recover", "--force", f"--outfile={temp_file}"],
                capture_output=True,
                text=True,
                input="\n"
            )
            
            output_lines = result.stderr.split('\n')
            seed_phrase = None
            for line in output_lines:
                if "recover: " in line:
                    seed_phrase = line.replace("recover: ", "").strip()
                    break
            
            if seed_phrase:
                click.echo("\n=== METHOD 2: BIP39 Seed Phrase ===")
                click.echo(f"Seed phrase: {seed_phrase}")
                click.echo("This is the most reliable way to import into Phantom.")
            
            os.remove(temp_file)
            
        except Exception as e:
            click.echo(f"Error processing wallet formats: {e}")

    async def check_wallet_balance(self, wallet_name: str):
        """Enhanced balance checking with token information"""
        try:
            self.wallet_manager.load_wallets()
            
            if wallet_name not in self.wallet_manager.wallets:
                click.echo(f"Error: Wallet '{wallet_name}' not found")
                return
                
            wallet = self.wallet_manager.wallets[wallet_name]
            network = get_network()
            
            # Enhanced header with network colors
            network_colors = {
                "devnet": "bright_yellow",
                "mainnet": "bright_green",
                "testnet": "bright_blue",
                "unknown": "white"
            }
            
            border = "═" * 70
            click.echo("\n" + click.style(border, fg=network_colors.get(network, "white")))
            click.echo(
                click.style("💫 ", fg="bright_yellow") + 
                click.style(f"SOLANA {network.upper()}", fg=network_colors.get(network, "white"), bold=True) + 
                click.style(" WALLET BALANCE", fg="bright_white", bold=True)
            )
            click.echo(click.style(border, fg=network_colors.get(network, "white")) + "\n")
            
            # Get price information
            sol_price = self.price_service.get_sol_price() if self.price_service else None
            
            if sol_price:
                click.echo(
                    click.style("💰 ", fg="bright_yellow") + 
                    click.style("Current SOL Price: ", fg="white") + 
                    click.style(f"(${sol_price['usd']:.2f} USD)", fg="bright_white", bold=True) + 
                    click.style(" | ", fg="white") + 
                    click.style(f"(${sol_price['cad']:.2f} 🍁)", fg="bright_white", bold=True)
                )
                click.echo(click.style(border, fg=network_colors.get(network, "white")) + "\n")
            
            # Connect to RPC
            async with AsyncClient(NETWORK_URLS[network]) as client:
                # Check balances for each token
                for token, info in self.TOKEN_INFO.items():
                    balance = await self.get_token_balance(
                        client,
                        wallet.public_key,
                        info['mint'],
                        info['decimals']
                    )
                    if balance > 0:
                        formatted_balance = self.format_amount(
                            balance,
                            token,
                            sol_price['usd'] if sol_price and token == "SOL" else None,
                            sol_price['cad'] if sol_price and token == "SOL" else None
                        )
                        click.echo(formatted_balance)
                        
        except Exception as e:
            click.echo(f"Error checking balance: {e}")

# CLI Interface
@click.group()
def cli():
    """Unified Solana wallet management system"""
    pass

# Wallet Commands
@cli.group()
def wallet():
    """Wallet management commands"""
    pass
    
@wallet.command()
@click.option('--name', '-n', required=True, help='Wallet name')
def balance(name):
    """Check wallet balance"""
    cmd = WalletCommands()
    asyncio.run(cmd.check_wallet_balance(name))
    
@wallet.command()
@click.option('--from-wallet', '-f', required=True, help='Source wallet name')
@click.option('--to-address', '-t', required=True, help='Destination address')
@click.option('--amount', '-a', required=True, type=float, help='Amount in SOL')
def transfer(from_wallet, to_address, amount):
    """Transfer SOL between wallets"""
    cmd = WalletCommands()
    asyncio.run(cmd.transfer_sol(from_wallet, to_address, amount))

# Configuration Commands
@cli.group()
def config():
    """Configuration management commands"""
    pass
    
@config.command()
@click.option('--password', prompt=True, hide_input=True,
              confirmation_prompt=True)
def encrypt(password):
    """Encrypt wallet configurations"""
    cmd = WalletCommands()
    cmd.migrate_to_encrypted(password)
    
@config.command()
def backup():
    """Backup wallet configurations"""
    cmd = WalletCommands()
    cmd.backup_configs()
    
@config.command()
@click.argument('timestamp')
def restore(timestamp):
    """Restore from backup"""
    cmd = WalletCommands()
    cmd.restore_from_backup(timestamp)
    
@config.command()
@click.option('--password', prompt=True, hide_input=True)
def verify(password):
    """Verify encrypted configurations"""
    cmd = WalletCommands()
    cmd.verify_encryption(password)

# Network Commands
@cli.group()
def network():
    """Network management commands"""
    pass
    
@network.command()
def status():
    """Show current network status"""
    network = get_network()
    click.echo(f"Current network: {network}")
    click.echo(f"RPC URL: {NETWORK_URLS[network]}")
    
@network.command()
@click.argument('name', type=click.Choice(['mainnet', 'devnet', 'testnet']))
def switch(name):
    """Switch between networks"""
    set_network(name)
    click.echo(f"Switched to {name}")

# Watch Mode Commands
@cli.group()
def watch():
    """Real-time wallet monitoring commands"""
    pass

@watch.command(name="start")
@click.option('--wallet', '-w', help='Specific wallet to monitor (optional)')
def start_watch_mode(wallet):
    """Launch real-time wallet monitoring GUI"""
    cmd = WalletCommands()
    
    # Get wallet configurations
    cmd.wallet_manager.load_wallets()
    wallet_configs = {}
    
    if wallet:
        # Monitor specific wallet if provided
        if wallet in cmd.wallet_manager.wallets:
            w = cmd.wallet_manager.wallets[wallet]
            wallet_configs[wallet] = {
                "pubkey": str(w.public_key),
                "name": wallet
            }
        else:
            click.echo(f"Error: Wallet '{wallet}' not found")
            return
    else:
        # Monitor all wallets
        for name, w in cmd.wallet_manager.wallets.items():
            wallet_configs[name] = {
                "pubkey": str(w.public_key),
                "name": name
            }
    
    if not wallet_configs:
        click.echo("No wallets configured for monitoring!")
        return
    
    click.echo(f"Starting watch mode for {len(wallet_configs)} wallet(s)...")
    launch_watch_mode(wallet_configs)

# Wallet Format Commands
@wallet.group()
def format():
    """Wallet format commands"""
    pass

@format.command()
@click.option('--wallet-name', '-w', help='Specific wallet to show formats for (optional)')
def show(wallet_name):
    """Show wallet formats for Phantom import"""
    cmd = WalletCommands()
    cmd.wallet_manager.load_wallets()
    
    if wallet_name:
        if wallet_name in cmd.wallet_manager.wallets:
            wallet = cmd.wallet_manager.wallets[wallet_name]
            cmd.show_wallet_formats(wallet_name, wallet.keypair_path)
        else:
            click.echo(f"Error: Wallet '{wallet_name}' not found")
    else:
        for name, wallet in cmd.wallet_manager.wallets.items():
            cmd.show_wallet_formats(name, wallet.keypair_path)

# Enhanced Wallet Management Commands
@wallet.command()
@click.argument('name')
@click.argument('keypair_path')
def add(name, keypair_path):
    """Add a new wallet"""
    cmd = WalletCommands()
    if cmd.validate_keypair_path(keypair_path):
        cmd.wallet_manager.add_wallet(name, keypair_path)
        click.echo(f"Added wallet: {name}")

@wallet.command()
@click.argument('name')
def remove(name):
    """Remove a wallet"""
    cmd = WalletCommands()
    cmd.wallet_manager.load_wallets()
    if name in cmd.wallet_manager.wallets:
        cmd.wallet_manager.remove_wallet(name)
        click.echo(f"Removed wallet: {name}")
    else:
        click.echo(f"Error: Wallet '{name}' not found")

@wallet.command(name="list")
def list_wallets():
    """List configured wallets"""
    cmd = WalletCommands()
    cmd.wallet_manager.load_wallets()
    
    if not cmd.wallet_manager.wallets:
        click.echo("No wallets configured")
        return
        
    click.echo("\n=== Configured Wallets ===")
    for name, wallet in cmd.wallet_manager.wallets.items():
        click.echo(f"\n🔑 {name}")
        click.echo(f"  Address: {cmd.format_address(str(wallet.public_key))}")
        click.echo(f"  Path: {wallet.keypair_path}")

@wallet.command()
@click.argument('name')
@click.argument('new_keypair_path')
def update(name, new_keypair_path):
    """Update wallet keypair path"""
    cmd = WalletCommands()
    cmd.wallet_manager.load_wallets()
    
    if name not in cmd.wallet_manager.wallets:
        click.echo(f"Error: Wallet '{name}' not found")
        return
        
    if cmd.validate_keypair_path(new_keypair_path):
        cmd.wallet_manager.update_wallet(name, new_keypair_path)
        click.echo(f"Updated wallet: {name}")

if __name__ == "__main__":
    cli() 