"""
Unified logging module providing both basic and enhanced logging functionality.
"""

import logging
import sys
from typing import Optional

def setup_basic_logging(level: int = logging.INFO) -> None:
    """
    Sets up basic logging to console.
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)

def setup_enhanced_logging(log_level: int = logging.INFO, 
                         log_file: Optional[str] = None,
                         module_levels: Optional[dict] = None) -> None:
    """
    Enhanced logging setup with additional features.
    
    Args:
        log_level: Base logging level (default: logging.INFO)
        log_file: Optional file path to also log to file
        module_levels: Optional dict of module names and their specific log levels
    """
    # Set up basic logging first
    setup_basic_logging(log_level)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logging.getLogger().addHandler(file_handler)
    
    # Set specific levels for main modules
    default_modules = ['src', 'ultimate_data_fetcher']
    module_levels = module_levels or {}
    
    # Combine default modules with any provided ones
    for module in default_modules:
        if module not in module_levels:
            module_levels[module] = log_level
    
    # Set levels for all specified modules
    for module_name, level in module_levels.items():
        logging.getLogger(module_name).setLevel(level)

# For backward compatibility
setup_logging = setup_enhanced_logging 