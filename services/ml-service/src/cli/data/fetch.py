#!/usr/bin/env python3
"""
Data Fetching CLI Component

Handles all data operations including:
- Historical data fetching
- Live data streaming
- Data conversion for ML (TensorFlow/PyTorch)
- Market information
"""

from typing import Optional, Dict, Any
from argparse import _SubParsersAction
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich import box

from src.cli.base import BaseCLI
from src.core.config import Config
from src.core.models import TimeRange
from src.ultimate_fetcher import UltimateDataFetcher

console = Console()

class DataFetchCLI(BaseCLI):
    """CLI component for data operations"""
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.fetcher: Optional[UltimateDataFetcher] = None
    
    async def setup(self) -> None:
        """Initialize data fetcher"""
        self.fetcher = UltimateDataFetcher(self.config)
        await self.fetcher.start()
    
    async def cleanup(self) -> None:
        """Cleanup data fetcher"""
        if self.fetcher:
            await self.fetcher.stop()
    
    def add_arguments(self, parser: _SubParsersAction) -> None:
        """Add data fetching arguments"""
        data_parser = parser.add_parser("data", help="Data operations")
        subparsers = data_parser.add_subparsers(dest="data_command")
        
        # List available markets
        list_parser = subparsers.add_parser("list", help="List available markets")
        list_parser.add_argument(
            "--exchange",
            help="Filter by exchange"
        )
        
        # Historical data fetching
        historical_parser = subparsers.add_parser("historical", help="Fetch historical data")
        historical_parser.add_argument(
            "--markets",
            nargs="+",
            required=True,
            help="Markets to fetch (e.g., BTC-PERP ETH-PERP)"
        )
        historical_parser.add_argument(
            "--exchanges",
            nargs="+",
            help="Exchanges to use (default: all configured)"
        )
        historical_parser.add_argument(
            "--resolution",
            default="1",
            choices=["1", "5", "15", "30", "60", "240", "1D", "1W"],
            help="Candle resolution"
        )
        historical_parser.add_argument(
            "--start-date",
            required=True,
            help="Start date (YYYY-MM-DD)"
        )
        historical_parser.add_argument(
            "--end-date",
            required=True,
            help="End date (YYYY-MM-DD)"
        )
        
        # Live data streaming
        live_parser = subparsers.add_parser("live", help="Stream live data")
        live_parser.add_argument(
            "--markets",
            nargs="+",
            required=True,
            help="Markets to stream"
        )
        live_parser.add_argument(
            "--exchanges",
            nargs="+",
            help="Exchanges to use (default: all configured)"
        )
        live_parser.add_argument(
            "--resolution",
            default="1",
            choices=["1", "5", "15", "30", "60"],
            help="Candle resolution"
        )
        
        # ML data conversion
        ml_parser = subparsers.add_parser("convert", help="Convert data for machine learning")
        ml_parser.add_argument(
            "--format",
            choices=["tfrecord", "pytorch"],
            required=True,
            help="Output format"
        )
        ml_parser.add_argument(
            "--markets",
            nargs="+",
            required=True,
            help="Markets to convert"
        )
        ml_parser.add_argument(
            "--exchanges",
            nargs="+",
            help="Exchanges to use"
        )
        ml_parser.add_argument(
            "--resolution",
            default="1",
            choices=["1", "5", "15", "30", "60", "240", "1D"],
            help="Candle resolution"
        )
        ml_parser.add_argument(
            "--start-date",
            required=True,
            help="Start date (YYYY-MM-DD)"
        )
        ml_parser.add_argument(
            "--end-date",
            required=True,
            help="End date (YYYY-MM-DD)"
        )
        ml_parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for ML datasets"
        )
    
    async def handle_command(self, args: Any) -> None:
        """Handle data commands"""
        if args.data_command == "list":
            await self._handle_list(args)
        elif args.data_command == "historical":
            await self._handle_historical(args)
        elif args.data_command == "live":
            await self._handle_live(args)
        elif args.data_command == "convert":
            await self._handle_convert(args)
    
    async def _handle_list(self, args: Any) -> None:
        """Handle market listing"""
        try:
            # Get available markets from symbol mapper
            markets = self.fetcher.symbol_mapper.get_markets(args.exchange)
            
            if not markets:
                console.print("[yellow]No markets found[/yellow]")
                return
            
            table = Table(title="Available Markets", box=box.ROUNDED)
            table.add_column("Market", style="cyan")
            table.add_column("Exchanges", style="green")
            
            for market, exchanges in markets.items():
                table.add_row(
                    market,
                    ", ".join(exchanges)
                )
            
            console.print(table)
            
        except Exception as e:
            self.logger.error(f"Error listing markets: {str(e)}")
            raise
    
    async def _handle_historical(self, args: Any) -> None:
        """Handle historical data fetching"""
        try:
            time_range = TimeRange(
                start=datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                end=datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            )
            
            console.print(f"[cyan]Fetching historical data for {len(args.markets)} markets...[/cyan]")
            
            await self.fetcher.fetch_historical_data(
                markets=args.markets,
                time_range=time_range,
                resolution=args.resolution,
                exchanges=args.exchanges
            )
            
            console.print("[green]Historical data fetching completed![/green]")
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    async def _handle_live(self, args: Any) -> None:
        """Handle live data streaming"""
        try:
            console.print(f"[cyan]Starting live data stream for {len(args.markets)} markets...[/cyan]")
            console.print("[yellow]Press Ctrl+C to stop streaming[/yellow]")
            
            await self.fetcher.start_live_fetching(
                markets=args.markets,
                resolution=args.resolution,
                exchanges=args.exchanges
            )
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Live data streaming stopped by user[/yellow]")
        except Exception as e:
            self.logger.error(f"Error in live data streaming: {str(e)}")
            raise
    
    async def _handle_convert(self, args: Any) -> None:
        """Handle ML data conversion"""
        try:
            time_range = TimeRange(
                start=datetime.strptime(args.start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                end=datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            )
            
            console.print(f"[cyan]Converting data to {args.format.upper()} format...[/cyan]")
            
            if args.format == "tfrecord":
                await self.fetcher.convert_historical_to_tfrecord(
                    markets=args.markets,
                    time_range=time_range,
                    resolution=args.resolution,
                    exchanges=args.exchanges
                )
            elif args.format == "pytorch":
                for market in args.markets:
                    for exchange in (args.exchanges or self.fetcher.exchange_handlers.keys()):
                        dataloader = self.fetcher.get_pytorch_dataloader(
                            market=market,
                            resolution=args.resolution,
                            exchange=exchange,
                            batch_size=args.batch_size
                        )
                        console.print(f"[green]Created PyTorch DataLoader for {exchange}/{market}[/green]")
            
            console.print("[green]Data conversion completed![/green]")
            
        except Exception as e:
            self.logger.error(f"Error converting data: {str(e)}")
            raise 