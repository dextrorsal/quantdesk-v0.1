# 🎯 Core Components

This directory contains the core functionality and models used throughout the Quantify platform.

## 🔍 Overview

The core components provide:
- Standardized data models
- Common interfaces
- Base classes
- Core exceptions
- Configuration management
- Logging system
- Time utilities
- Symbol mapping

## 📁 Structure

```
core/
├── models.py           # Core data models (Candle, TimeRange, etc.)
├── exceptions.py       # Custom exceptions
├── logging.py         # Logging configuration and utilities
├── config.py          # Configuration management
├── time_utils.py      # Time-related utilities
├── symbol_mapper.py   # Exchange symbol mapping
├── __init__.py        # Package initialization
└── README.md          # This file
```

## 📊 Data Models

### StandardizedCandle
The fundamental data structure for market data:
```python
from src.core.models import StandardizedCandle

candle = StandardizedCandle(
    timestamp=datetime.now(timezone.utc),
    open=100.0,
    high=105.0,
    low=98.0,
    close=103.0,
    volume=1000.0,
    market="SOL",
    resolution="1h",
    source="drift"
)
```

### TimeRange
Represents a time period for data queries:
```python
from src.core.models import TimeRange
from datetime import datetime, timezone, timedelta

now = datetime.now(timezone.utc)
time_range = TimeRange(
    start=now - timedelta(days=1),
    end=now
)
```

## ⚙️ Configuration

### Loading Configuration
```python
from src.core.config import Config

# Load configuration
config = Config.load_config()

# Access values
rpc_url = config.get("solana.rpc_url")
api_key = config.get("exchanges.binance.api_key")
```

## 📝 Logging

### Setting up Logging
```python
from src.core.logging import setup_logger

# Create logger
logger = setup_logger(
    name="trading",
    log_level="INFO",
    log_file="trading.log"
)

# Use logger
logger.info("Starting trading session")
logger.error("Error executing trade", extra={"market": "SOL"})
```

## ⏰ Time Utilities

### Time Operations
```python
from src.core.time_utils import TimeUtils

# Get current time in UTC
now = TimeUtils.get_current_time()

# Format timestamp
formatted = TimeUtils.format_timestamp(now)

# Parse timestamp
parsed = TimeUtils.parse_timestamp("2024-03-25T12:00:00Z")
```

## 🔄 Symbol Mapping

### Exchange Symbol Conversion
```python
from src.core.symbol_mapper import SymbolMapper

# Initialize mapper
mapper = SymbolMapper()

# Convert symbols between exchanges
drift_symbol = mapper.to_drift_symbol("SOLUSDC")
binance_symbol = mapper.to_binance_symbol("SOL")
```

## 🎯 Core Functionality

### Exception Handling
Custom exceptions for better error management:
```python
from src.core.exceptions import (
    ExchangeError,
    ValidationError,
    ConfigurationError
)

try:
    # Your code here
except ValidationError as e:
    logger.error("Validation failed", exc_info=e)
except ExchangeError as e:
    logger.error("Exchange error", exc_info=e)
except ConfigurationError as e:
    logger.error("Configuration error", exc_info=e)
```

## 🧪 Testing

The core components have comprehensive test coverage:
```bash
# Run core tests
python -m pytest tests/core/

# Run specific component tests
python -m pytest tests/core/test_models.py
python -m pytest tests/core/test_config.py
```

## 📚 Best Practices

1. Always use the standardized models for consistency
2. Handle exceptions appropriately
3. Use type hints for better code clarity
4. Document any changes to core functionality
5. Add tests for new features

## 🔄 Integration

When integrating with core components:

1. Import models from `src.core.models`
2. Use appropriate exceptions from `src.core.exceptions`
3. Configure logging using `src.core.logging`
4. Use time utilities from `src.core.time_utils`
5. Handle symbol mapping with `src.core.symbol_mapper`

Example:
```python
from src.core.models import StandardizedCandle, TimeRange
from src.core.exceptions import ValidationError
from src.core.logging import setup_logger
from src.core.time_utils import TimeUtils
from src.core.symbol_mapper import SymbolMapper

class MyNewComponent:
    def __init__(self):
        self.logger = setup_logger("my_component")
        self.symbol_mapper = SymbolMapper()
        
    def process_data(self, time_range: TimeRange) -> List[StandardizedCandle]:
        try:
            if not isinstance(time_range, TimeRange):
                raise ValidationError("Invalid time range")
            # Your processing logic here
        except Exception as e:
            self.logger.error("Processing failed", exc_info=e)
            raise
``` 