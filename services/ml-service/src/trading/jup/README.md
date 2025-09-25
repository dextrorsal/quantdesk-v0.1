# Jupiter DEX Integration

This directory contains the consolidated adapter for interacting with Jupiter, Solana's leading DEX aggregator for optimal token swaps and trading.

## Features

- Token swaps with optimal routing
- Real-time price quotes
- Route analysis and optimization
- Balance checking
- Market information
- Token verification
- Slippage protection
- Multi-wallet support

## CLI Usage

The Jupiter module is integrated into the main D3X7-ALGO CLI. Here are the available commands:

### Basic Commands

```bash
# Get a Quote
d3x7 jupiter quote --from SOL --to USDC --amount 1.0 [--wallet my_wallet]

# Execute a Swap
d3x7 jupiter swap --from SOL --to USDC --amount 1.0 --wallet my_wallet [--slippage 0.5]

# Check Balances
d3x7 jupiter balance --wallet my_wallet [--token SOL]

# Find Available Routes
d3x7 jupiter routes --from SOL --to USDC --amount 1.0 [--limit 3]

# Get Market Information
d3x7 jupiter market --pair SOL/USDC

# Verify Token
d3x7 jupiter verify --token SOL
```

### Example Usage

1. Getting a quote for swapping SOL to USDC:
```bash
d3x7 jupiter quote --from SOL --to USDC --amount 1.0
```
Output will show:
- Input amount and token
- Expected output amount
- Price impact
- Route type

2. Executing a swap with a specific wallet:
```bash
d3x7 jupiter swap --from SOL --to USDC --amount 1.0 --wallet trading_wallet --slippage 0.5
```
The command will:
- Show a preview of the swap
- Display price impact and slippage settings
- Execute the swap and show the transaction signature

3. Checking token balances:
```bash
# Check all token balances
d3x7 jupiter balance --wallet my_wallet

# Check specific token balance
d3x7 jupiter balance --wallet my_wallet --token SOL
```

## Python API Usage

You can also use the Jupiter adapter directly in your Python code:

```python
from d3x7_algo.trading.jup import JupiterAdapter

async def example():
    adapter = JupiterAdapter()
    await adapter.initialize()
    
    try:
        # Get a quote
        quote = await adapter.get_quote("SOL", "USDC", 1.0)
        print(f"Expected output: {quote['outAmount']} USDC")
        
        # Execute a swap
        result = await adapter.execute_swap(
            "SOL", "USDC", 1.0,
            slippage=0.5,
            wallet=wallet
        )
        print(f"Swap executed: {result['signature']}")
        
    finally:
        await adapter.cleanup()
```

## Configuration

Configure Jupiter through environment variables or the config file:

```env
JUPITER_RPC_ENDPOINT=https://api.mainnet-beta.solana.com
JUPITER_MAX_SLIPPAGE=1.0
JUPITER_DEFAULT_WALLET=trading_wallet
```

## Security Features

- Slippage protection
- Price impact warnings
- Route optimization
- Transaction simulation
- Wallet validation

## Error Handling

The CLI provides clear error messages for common issues:
- Insufficient balance
- Invalid token addresses
- Network connectivity issues
- Route unavailability
- High price impact warnings

## Logging

Comprehensive logging is available:

```python
import logging
logging.getLogger("d3x7_algo.trading.jup").setLevel(logging.DEBUG)
```

## Dependencies

- Python 3.10+
- solana-py
- anchorpy
- aiohttp
- rich (for CLI output)

## Contributing

When adding features:
1. Follow existing code structure
2. Add appropriate tests
3. Update documentation
4. Include example usage
5. Consider security implications

## Support

For issues and feature requests, please use the issue tracker on our repository.

## Directory Structure

```
/src/trading/
├── jup/                   # Jupiter integration
│   ├── __init__.py       # Module exports
│   ├── jup_adapter.py    # Consolidated Jupiter adapter
│   ├── examples/         # Example scripts
│   │   └── basic_usage.py # Basic usage examples
│   └── README.md        # This file
```

## Quick Start

### Using the CLI Tools

The CLI interface provides comprehensive functionality for interacting with Jupiter:

```bash
# Price Information
python3 -m src.trading.jup.jup_adapter price SOL-USDC

# Account Balances
python3 -m src.trading.jup.jup_adapter balance

# Route Analysis
python3 -m src.trading.jup.jup_adapter quote SOL-USDC 1.0
python3 -m src.trading.jup.jup_adapter route SOL-USDC 1.0

# Execute Swaps
python3 -m src.trading.jup.jup_adapter swap SOL-USDC 1.0 --slippage 50

# Price Monitoring
python3 -m src.trading.jup.jup_adapter monitor
python3 -m src.trading.jup.jup_adapter monitor --markets SOL-USDC BTC-USDC ETH-USDC
```

### Using the Python API

```python
from src.trading.jup.jup_adapter import JupiterAdapter

async def example():
    adapter = JupiterAdapter()
    await adapter.connect()
    
    try:
        # Get market price
        price = await adapter.get_market_price("SOL-USDC")
        print(f"SOL-USDC Price: ${price:.2f}")
        
        # Get route metrics
        metrics = await adapter.get_route_metrics("SOL-USDC", 1.0)
        print(f"Price Impact: {metrics['price_impact_pct']:.2f}%")
        
        # Execute swap
        result = await adapter.execute_swap(
            market="SOL-USDC",
            input_amount=1.0,
            slippage_bps=50  # 0.5% slippage
        )
        print(f"Swap executed: {result['transaction']}")
        
    finally:
        await adapter.close()
```

## Component Details

### Jupiter Adapter (`jup_adapter.py`)

Our consolidated adapter provides comprehensive functionality for Jupiter DEX operations:

**Key Features:**
- Market Operations
  - Real-time price fetching
  - Route optimization
  - Slippage protection
  - Fee estimation
  
- Trading Operations
  - Token swaps
  - Route analysis
  - Transaction monitoring
  - Trade logging
  
- Account Management
  - Balance tracking
  - Token management
  - Transaction history

## Market Support

### Supported Token Pairs
- SOL-USDC
- BTC-USDC
- ETH-USDC
- USDC-SOL (reverse)
- USDC-BTC (reverse)
- USDC-ETH (reverse)

## Environment Setup

### Dependencies
```bash
pip install solana-py anchorpy aiohttp tabulate
```

### Configuration
The system uses environment variables for configuration:

1. **Required Environment Variables:**
   ```bash
   SOLANA_NETWORK=mainnet  # or devnet
   WALLET_PATH=/path/to/wallet.json
   ```

2. **Optional Environment Variables:**
   ```bash
   JUP_SLIPPAGE_BPS=50     # Default slippage (0.5%)
   JUP_LOG_LEVEL=INFO      # Logging level
   ```

## Security Best Practices

1. **Slippage Protection:**
   - Always set appropriate slippage tolerance
   - Monitor price impact
   - Use route analysis before swaps

2. **Transaction Safety:**
   ```python
   # Get route metrics before swap
   metrics = await adapter.get_route_metrics("SOL-USDC", 1.0)
   if metrics["price_impact_pct"] > 1.0:
       logger.warning("High price impact detected!")
   ```

3. **Error Handling:**
   ```python
   try:
       await adapter.execute_swap(market, amount)
   except Exception as e:
       logger.error(f"Swap failed: {e}")
   finally:
       await adapter.close()
   ```

## Route Metrics Example

The `route` command provides comprehensive metrics:

```json
{
  "price": 80.25,
  "price_impact_pct": 0.15,
  "min_output": 79.85,
  "route_count": 3,
  "best_route_hops": 1,
  "fee_estimates": {
    "platform_fee": 0.001,
    "network_fee": 0.000005
  }
}
```

## Examples

The `examples/` directory contains practical examples of using the Jupiter adapter:

### Basic Usage Example

Run the basic usage example to see the adapter in action:
```bash
python3 -m src.trading.jup.examples.basic_usage
```

This example demonstrates:
- Account overview with token balances
- Route analysis and swap execution
- Real-time price monitoring
- Error handling best practices

For more examples and detailed usage patterns, check the `examples/` directory.

## Troubleshooting

Common issues and solutions:

1. **Connection Issues:**
   - Verify RPC endpoint
   - Check network selection
   - Ensure wallet is accessible

2. **Swap Failures:**
   - Check token balances
   - Verify slippage settings
   - Monitor route availability

3. **Price Impact:**
   - Use route analysis
   - Monitor market conditions
   - Adjust trade size

For additional support:
1. Check the detailed logs
2. Review the [Jupiter API Documentation](https://docs.jup.ag/)
3. Ensure all environment variables are properly set