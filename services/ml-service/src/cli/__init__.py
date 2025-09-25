"""
Quantify CLI Package

A unified command-line interface for all Quantify components:
- Data fetching and processing
- Trading execution (Drift, Jupiter, etc.)
- Wallet management
- ML model interaction
"""

from src.cli.main import main

__all__ = ['main'] 