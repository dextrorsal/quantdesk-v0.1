#!/usr/bin/env python3
"""
Base CLI Component

Provides common functionality for all CLI components including:
- Configuration management
- Logging setup
- Error handling
- Output formatting
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from argparse import _SubParsersAction
from rich.console import Console

from src.core.config import Config
from src.core.logging import setup_enhanced_logging

console = Console()


class BaseCLI(ABC):
    """Base class for all CLI components"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize base CLI component"""
        self.config = config or Config()
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup component-specific logging"""
        # Get component name from class name (e.g., DataFetchCLI -> data_fetch)
        component_name = self.__class__.__name__.replace('CLI', '').lower()
        
        # Setup enhanced logging with component-specific configuration
        setup_enhanced_logging(
            log_level=self.config.log_level if hasattr(self.config, 'log_level') else logging.INFO,
            log_file=f"logs/{component_name}.log",
            module_levels={
                f"src.cli.{component_name}": logging.DEBUG,
                "src": logging.INFO
            }
        )
        
        # Get logger for this component
        return logging.getLogger(f"src.cli.{component_name}")
    
    @abstractmethod
    async def setup(self) -> None:
        """Initialize component resources"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        pass
    
    @abstractmethod
    def add_arguments(self, parser: _SubParsersAction) -> None:
        """Add component-specific arguments"""
        pass
    
    @abstractmethod
    async def handle_command(self, args: Any) -> None:
        """Handle component-specific commands"""
        pass
    
    def format_output(self, data: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary data for console output"""
        output = []
        for key, value in data.items():
            if isinstance(value, dict):
                output.append(f"{'  ' * indent}{key}:")
                output.append(self.format_output(value, indent + 1))
            else:
                output.append(f"{'  ' * indent}{key}: {value}")
        return "\n".join(output)
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with context"""
        if context:
            self.logger.error(f"{context}: {str(error)}")
        else:
            self.logger.error(str(error))
        self.logger.debug("Error details:", exc_info=error)
    
    def log_warning(self, message: str, context: str = "") -> None:
        """Log warning with context"""
        if context:
            self.logger.warning(f"{context}: {message}")
        else:
            self.logger.warning(message)
    
    def log_info(self, message: str, context: str = "") -> None:
        """Log info with context"""
        if context:
            self.logger.info(f"{context}: {message}")
        else:
            self.logger.info(message)
    
    def log_debug(self, message: str, context: str = "") -> None:
        """Log debug with context"""
        if context:
            self.logger.debug(f"{context}: {message}")
        else:
            self.logger.debug(message) 