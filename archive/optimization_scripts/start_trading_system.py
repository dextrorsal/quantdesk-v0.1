#!/usr/bin/env python3
"""
Start Trading System - Launch the complete trading system with dashboard

This script starts both the combined model trader and the trading dashboard,
providing a complete trading system with real-time monitoring.

USE CASES:
- **Complete trading system**: Launch both trader and dashboard together
- **Real-time monitoring**: Monitor trading signals and performance in real-time
- **System orchestration**: Manage multiple processes (trader + dashboard)
- **Production deployment**: Deploy the complete trading system
- **Development testing**: Test the full system during development
- **Process management**: Handle startup, shutdown, and cleanup of processes

DIFFERENCES FROM OTHER TRADING SCRIPTS:
- start_trading_system.py: Complete system orchestration (trader + dashboard)
- combined_model_trader.py: Multi-timeframe model combination and trading
- run_comparison.py: Model comparison and backtesting
- final_comparison.py: Final model evaluation and comparison
- fresh_backtest_pipeline.py: Fresh backtesting pipeline

WHEN TO USE:
- When you want to run the complete trading system
- For real-time trading with dashboard monitoring
- When you need process management and orchestration
- For production deployment of the trading system
- When you want to test the full system

FEATURES:
- Process orchestration and management
- Real-time dashboard monitoring
- Configurable confidence thresholds
- Automatic process cleanup on shutdown
- ASCII art startup messages
- Error handling and process monitoring

EXAMPLES:
    # Start with default settings
    python scripts/start_trading_system.py
    
    # Custom confidence thresholds
    python scripts/start_trading_system.py --confidence-threshold 0.7 --combined-threshold 0.6
    
    # Custom model paths
    python scripts/start_trading_system.py --model-5m path/to/5m_model.pt --model-15m path/to/15m_model.pt
"""
import os
import logging
import asyncio
import signal
import atexit
import argparse
import time
import subprocess

from src.trading.mainnet.mainnet_trade import MainnetTrader
from src.core.config import Config
from src.utils.log_setup import setup_logging

# The project should be installed in editable mode, so sys.path manipulation is not needed.
# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# pass # sys.path.append(ROOT_DIR)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Start the trading system with dashboard")
    
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                       help="Confidence threshold for signals (default: 0.3)")
    
    parser.add_argument("--combined-threshold", type=float, default=0.25,
                       help="Combined confidence threshold (default: 0.25)")
    
    parser.add_argument("--model-5m", type=str, 
                       default="models/trained/extended/5m_20250402_1614/final_model_5m.pt",
                       help="Path to 5m model")
    
    parser.add_argument("--model-15m", type=str,
                       default="models/trained/extended/15m_20250402_1614/final_model_15m.pt",
                       help="Path to 15m model")
    
    parser.add_argument("--neon-connection", type=str,
                       default="postgresql://dex:testpassword@localhost:5432/solana_trading_test",
                       help="Neon database connection string")
    
    return parser.parse_args()

def start_model_trader(args):
    """Start the combined model trader process"""
    print("Starting Combined Model Trader...")
    
    cmd = [
        "python", 
        os.path.join(SCRIPT_DIR, "combined_model_trader.py"),
        "--live",
        f"--confidence-threshold={args.confidence_threshold}",
        f"--combined-threshold={args.combined_threshold}",
        f"--model-5m={args.model_5m}",
        f"--model-15m={args.model_15m}",
        f"--neon-connection={args.neon_connection}"
    ]
    
    trader_process = subprocess.Popen(cmd)
    return trader_process

def start_dashboard(args):
    """Start the dashboard process"""
    print("Starting Trading Dashboard...")
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(SCRIPT_DIR, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set environment variable for Neon connection
    env = os.environ.copy()
    env["NEON_CONNECTION_STRING"] = args.neon_connection
    
    cmd = [
        "python", 
        os.path.join(SCRIPT_DIR, "dashboard", "trader_dashboard.py")
    ]
    
    dashboard_process = subprocess.Popen(cmd, env=env)
    return dashboard_process

def cleanup_processes(processes):
    """Clean up processes on exit"""
    print("\nShutting down trading system...")
    
    for process in processes:
        if process and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()

def print_startup_message(dashboard_url):
    """Print startup message with ASCII art"""
    print("\n" + "="*80)
    print("""
    ███████╗ ██████╗ ██╗         ████████╗██████╗  █████╗ ██████╗ ███████╗██████╗ 
    ██╔════╝██╔═══██╗██║         ╚══██╔══╝██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
    ███████╗██║   ██║██║            ██║   ██████╔╝███████║██║  ██║█████╗  ██████╔╝
    ╚════██║██║   ██║██║            ██║   ██╔══██╗██╔══██║██║  ██║██╔══╝  ██╔══██╗
    ███████║╚██████╔╝███████╗       ██║   ██║  ██║██║  ██║██████╔╝███████╗██║  ██║
    ╚══════╝ ╚═════╝ ╚══════╝       ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝
                                                                                  
      ML-POWERED TRADING SYSTEM WITH NEON DB
    """)
    print("="*80)
    print(f"\n🚀 Trading model is running! The system is now monitoring SOL price in real-time.")
    print(f"\n🌐 Dashboard is available at: {dashboard_url}")
    print("\n📊 The dashboard will show:")
    print("   - Real-time price chart")
    print("   - Trading signals as they occur")
    print("   - Performance statistics")
    print("\n🗄️  All data is being stored in your Neon database")
    print("\n⚠️  Press Ctrl+C to stop the trading system")
    print("\n" + "="*80 + "\n")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Create data directory for dashboard if it doesn't exist
    dashboard_data_dir = os.path.join(SCRIPT_DIR, "dashboard", "data")
    os.makedirs(dashboard_data_dir, exist_ok=True)
    
    # Start processes
    processes = []
    
    try:
        # Start dashboard first
        dashboard_process = start_dashboard(args)
        processes.append(dashboard_process)
        
        # Wait for dashboard to start
        time.sleep(2)
        
        # Start model trader
        trader_process = start_model_trader(args)
        processes.append(trader_process)
        
        # Register cleanup function
        atexit.register(cleanup_processes, processes)
        
        # Print startup message
        dashboard_url = "http://127.0.0.1:5000"
        print_startup_message(dashboard_url)
        
        # Keep running until interrupted
        while all(p.poll() is None for p in processes):
            time.sleep(1)
        
        # Check if any process died
        for i, process in enumerate(processes):
            if process.poll() is not None:
                print(f"Process {i} exited with code {process.returncode}")
        
    except KeyboardInterrupt:
        pass
    finally:
        cleanup_processes(processes)

if __name__ == "__main__":
    main() 