# Trading Mainnet Directory

This directory contains components for secure mainnet trading on Solana, with comprehensive security controls, position monitoring, and trade execution capabilities.

## Directory Structure

```
mainnet/
├── __init__.py                 - Module exports
├── mainnet_trade.py           - Main trading interface
├── security_limits.py         - Trading security controls
└── README.md                  - This documentation
```

## Component Details

### `mainnet_trade.py`

Comprehensive trading interface for executing trades on Solana mainnet with security controls.

**Key Features:**
1. **Trade Execution:**
   - Jupiter token swaps
   - Position monitoring integration

2. **Security Integration:**
   - Size limits enforcement
   - Risk controls
   - Trade logging and analytics

**Usage Example:**
```bash
# Execute Jupiter swap
python3 mainnet_trade.py jupiter --market SOL-USDC --amount 1.0 --slippage 100

# Start position monitoring
python3 mainnet_trade.py monitor
```

### `security_limits.py` 🛡️

Enhanced security framework for mainnet trading with comprehensive controls.

**Key Security Features:**
1. **Position Management:**
   - Market-specific position size limits
   - Default limits for new markets
   - Real-time position validation

2. **Risk Controls:**
   - Leverage limits per market
   - Daily volume tracking
   - Emergency shutdown triggers
   - Loss threshold monitoring

3. **Emergency Controls:**
   - Automatic trading suspension
   - Volume spike detection
   - Maximum drawdown protection
   - Manual override capabilities

## Configuration

### Environment Variables
```bash
# Required
MAINNET_RPC_ENDPOINT="https://..."  # Mainnet RPC endpoint
WALLET_PASSWORD="..."               # Wallet encryption password
```

### Security Configuration
```json
{
    "max_position_size": {
        "SOL-PERP": 2.0,
        "BTC-PERP": 0.05
    },
    "max_leverage": {
        "SOL-PERP": 3,
        "BTC-PERP": 2
    },
    "daily_volume_limit": 5.0,
    "emergency_shutdown_triggers": {
        "loss_threshold_pct": 3.0,
        "volume_spike_multiplier": 2.0
    }
}
```

## Integration Points

### With Jupiter Aggregator
- Token swaps
- Price discovery
- Route optimization

## Development Guidelines

1. **Testing:**
   - Use devnet for testing
   - Never test with real funds
   - Verify security limits

2. **Deployment:**
   - Double-check RPC endpoints
   - Verify security settings
   - Monitor initial trades

3. **Maintenance:**
   - Regular security audits
   - Update limits as needed
   - Monitor system health

## Error Handling

The system implements comprehensive error handling:
```python
try:
    # Execute trade with security checks
    result = await trading.execute_jupiter_swap(
        market="SOL-USDC",
        input_amount=1.0
    )
except SecurityLimitError as e:
    logger.error(f"Security limit exceeded: {e}")
except TradeExecutionError as e:
    logger.error(f"Trade execution failed: {e}")
```

## Monitoring and Alerts

1. **Position Alerts:**
   - Size limit breaches
   - Leverage warnings
   - PnL thresholds

2. **System Alerts:**
   - Emergency shutdowns
   - Volume spikes
   - Connection issues

3. **Performance Monitoring:**
   - Trade execution times
   - Slippage analysis
   - Fee tracking

## Notes

- All operations target Solana mainnet
- Uses real funds - handle with care
- Implements comprehensive security
- Requires proper key management
- Regular monitoring recommended

## Troubleshooting

1. **Trade Rejections:**
   - Check security limits
   - Verify wallet balance
   - Check market status

2. **Monitor Issues:**
   - Verify RPC connection
   - Check update interval